from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
import uuid
import zipfile
from pathlib import Path
from urllib import error, request

from openclaw_skill.config import SkillConfig
from openclaw_skill.models.schemas import LabelDocument


class LabelParserService:
    """Parser service with txt/docx + Aliyun OCR image support."""

    def __init__(self, config: SkillConfig | None = None):
        self.config = config or SkillConfig()
        self.last_ocr_raw_payload: str | None = None
        self.last_parsed_data: dict = {}

    def parse(self, source_path: str) -> LabelDocument:
        path = Path(source_path)

        # reset debug state
        self.last_ocr_raw_payload = None
        self.last_parsed_data = {
            "source": str(path),
            "source_type": path.suffix.lower(),
            "full_text": "",
            "text_blocks": [],
            "source_images": [],
            "raw_records": [],
        }

        if not path.exists():
            self.last_parsed_data["error"] = "source_not_found"
            return LabelDocument(source_path=str(path), raw_text="", fields={})

        suffix = path.suffix.lower()

        if suffix == ".txt":
            full_text = path.read_text(encoding="utf-8", errors="ignore")
            self.last_parsed_data["full_text"] = full_text
            self.last_parsed_data["text_blocks"] = [
                {"source": str(path), "kind": "txt", "text": full_text}
            ]

        elif suffix == ".docx":
            full_text = self._read_docx(path)
            self.last_parsed_data["full_text"] = full_text

        elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            full_text = self._ocr_image(path)
            self.last_parsed_data["full_text"] = full_text

        else:
            full_text = f"[UNSUPPORTED_FORMAT] {suffix}"
            self.last_parsed_data["full_text"] = full_text
            self.last_parsed_data["error"] = "unsupported_format"

        fields = self._naive_extract(full_text)
        # 通用轻结构输出（不强依赖固定标签模板）
        fields["full_text"] = self.last_parsed_data.get("full_text", "")
        fields["text_blocks"] = self.last_parsed_data.get("text_blocks", [])
        fields["source_images"] = self.last_parsed_data.get("source_images", [])
        fields["raw_records"] = self.last_parsed_data.get("raw_records", [])

        return LabelDocument(source_path=str(path), raw_text=full_text, fields=fields)

    def _read_docx(self, file_path: Path) -> str:
        """Read paragraph text + OCR images embedded in docx (word/media/*)."""
        text_lines: list[str] = []
        image_ocr_lines: list[str] = []
        ocr_debug_records: list[dict] = []

        try:
            import importlib

            docx_module = importlib.import_module("docx")
            doc = docx_module.Document(str(file_path))
            text_lines = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
            if text_lines:
                self.last_parsed_data["text_blocks"].append(
                    {
                        "source": str(file_path),
                        "kind": "docx_paragraphs",
                        "text": "\n".join(text_lines),
                    }
                )
        except Exception as exc:
            ocr_debug_records.append(
                {"error": str(exc), "source": str(file_path), "type": "docx_parse_error"}
            )

        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                media_names = [
                    n
                    for n in zf.namelist()
                    if n.startswith("word/media/")
                    and n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
                ]

                for media_name in media_names:
                    img_bytes = zf.read(media_name)
                    text, raw_payload = self._ocr_image_bytes(img_bytes)
                    if text.strip():
                        image_ocr_lines.append(text.strip())
                        self.last_parsed_data["text_blocks"].append(
                            {
                                "source": media_name,
                                "kind": "docx_embedded_image_ocr",
                                "text": text.strip(),
                            }
                        )

                    self.last_parsed_data["source_images"].append(media_name)
                    ocr_debug_records.append(
                        {
                            "image": media_name,
                            "text_preview": text[:120],
                            "raw": raw_payload,
                        }
                    )
        except Exception as exc:
            ocr_debug_records.append(
                {"error": str(exc), "source": str(file_path), "type": "docx_media_extract_error"}
            )

        if ocr_debug_records:
            self.last_parsed_data["raw_records"] = ocr_debug_records
            self.last_ocr_raw_payload = json.dumps(
                {
                    "source": str(file_path),
                    "type": "docx_embedded_images_ocr",
                    "records": ocr_debug_records,
                },
                ensure_ascii=False,
            )

        merged = []
        if text_lines:
            merged.append("\n".join(text_lines))
        if image_ocr_lines:
            merged.append("\n".join(image_ocr_lines))
        return "\n".join(merged)

    def _ocr_image(self, image_path: Path) -> str:
        if not image_path.exists():
            return ""

        img_bytes = image_path.read_bytes()
        text, raw_payload = self._ocr_image_bytes(img_bytes)
        self.last_ocr_raw_payload = raw_payload

        self.last_parsed_data["source_images"].append(str(image_path))
        self.last_parsed_data["raw_records"].append(
            {
                "image": str(image_path),
                "text_preview": text[:120],
                "raw": raw_payload,
            }
        )
        self.last_parsed_data["text_blocks"].append(
            {
                "source": str(image_path),
                "kind": "image_ocr",
                "text": text,
            }
        )

        return text

    def _ocr_image_bytes(self, img_bytes: bytes) -> tuple[str, str]:
        image_b64 = base64.b64encode(img_bytes).decode("utf-8")
        body_dict = {"img": image_b64}
        body_bytes = json.dumps(body_dict).encode("utf-8")

        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json",
        }

        if self.config.ocr_auth_mode.upper() == "APPCODE":
            if not self.config.ocr_appcode:
                payload = json.dumps({"error": "OCR_APPCODE is empty"}, ensure_ascii=False)
                return "", payload
            headers["Authorization"] = f"APPCODE {self.config.ocr_appcode}"
        else:
            sign_headers = self._build_signature_headers(body_bytes)
            headers.update(sign_headers)

        req = request.Request(
            self.config.ocr_endpoint,
            data=body_bytes,
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.config.ocr_timeout_seconds) as resp:
                payload = resp.read().decode("utf-8", errors="ignore")
                return self._extract_text_from_response(payload), payload
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            payload = json.dumps(
                {"error": str(exc), "endpoint": self.config.ocr_endpoint},
                ensure_ascii=False,
            )
            return "", payload

    def _build_signature_headers(self, body_bytes: bytes) -> dict[str, str]:
        if not self.config.ocr_app_key or not self.config.ocr_app_secret:
            return {}

        nonce = str(uuid.uuid4())
        timestamp = str(int(time.time() * 1000))
        body_md5 = base64.b64encode(hashlib.md5(body_bytes).digest()).decode("utf-8")

        string_to_sign = "\n".join(
            [
                "POST",
                "application/json; charset=UTF-8",
                body_md5,
                "application/json",
                timestamp,
                nonce,
                "/ocrservice/advanced",
            ]
        )

        signature = base64.b64encode(
            hmac.new(
                self.config.ocr_app_secret.encode("utf-8"),
                string_to_sign.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        return {
            "X-Ca-Key": self.config.ocr_app_key,
            "X-Ca-Nonce": nonce,
            "X-Ca-Timestamp": timestamp,
            "X-Ca-Signature": signature,
            "X-Ca-Signature-Headers": "x-ca-key,x-ca-nonce,x-ca-timestamp",
            "Content-MD5": body_md5,
        }

    def _extract_text_from_response(self, payload: str) -> str:
        """Best-effort extraction for varying OCR JSON formats."""
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return payload

        candidates: list[str] = []

        if isinstance(data, dict):
            for key in ("content", "text", "result", "data", "words_result"):
                if key in data:
                    candidates.extend(self._flatten_text(data[key]))
        else:
            candidates.extend(self._flatten_text(data))

        merged = "\n".join([x for x in candidates if x.strip()])
        return merged.strip()

    def _flatten_text(self, obj) -> list[str]:
        out: list[str] = []
        if obj is None:
            return out
        if isinstance(obj, str):
            out.append(obj)
            return out
        if isinstance(obj, (int, float)):
            out.append(str(obj))
            return out
        if isinstance(obj, list):
            for item in obj:
                out.extend(self._flatten_text(item))
            return out
        if isinstance(obj, dict):
            for k in ("words", "text", "content", "value"):
                if k in obj:
                    out.extend(self._flatten_text(obj[k]))
            if not out:
                for v in obj.values():
                    out.extend(self._flatten_text(v))
            return out
        return out

    def _naive_extract(self, text: str) -> dict:
        text = text or ""
        fields: dict = {
            "product_name": None,
            "ingredients": text,
            "claims": [],
            "nutrition": {},
        }

        if "低糖" in text:
            fields["claims"].append("低糖")

        if "产品名称" in text:
            fields["product_name"] = "已识别(占位)"

        sugar_match = re.search(r"糖\s*([0-9]+(?:\.[0-9]+)?)\s*g", text)
        if sugar_match:
            fields["nutrition"]["sugar_g_per_100g"] = float(sugar_match.group(1))

        if "能量" in text:
            fields["nutrition"]["energy_kj"] = "detected"

        return fields
