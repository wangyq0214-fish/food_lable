from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Literal

from openai import OpenAI
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

from openclaw_skill.config import SkillConfig

SplitMode = Literal["law", "standard", "table_aware", "additive"]

CHAPTER_PATTERN = re.compile(r"^\s*(第[一二三四五六七八九十百千0-9]+章)")
SECTION_PATTERN = re.compile(r"^\s*(第[一二三四五六七八九十百千0-9]+节)")
ARTICLE_PATTERN = re.compile(r"^\s*(第[一二三四五六七八九十百千0-9]+条)")
NUMERIC_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+){0,4})(?:\s+|$)")
APPENDIX_PATTERN = re.compile(r"^\s*附\s*录\s*([A-Z])\b")
APPENDIX_ITEM_PATTERN = re.compile(r"^\s*([A-Z])\.(\d+(?:\.\d+)*)\b")
CN_ENUM_PATTERN = re.compile(r"^\s*([一二三四五六七八九十百千]+)、")
CN_PAREN_ENUM_PATTERN = re.compile(r"^\s*[（\(]([一二三四五六七八九十百千]+)[）\)]")
TABLE_PATTERN = re.compile(r"^\s*表\s*([0-9]+)")
TABLE_ANY_PATTERN = re.compile(r"^\s*表\s*([A-Z]\.[0-9]+|[0-9]+)")
FOOD_CLASS_PATTERN = re.compile(r"^\s*食品分类号")
SERIAL_ROW_PATTERN = re.compile(r"序号\s*\d+\s*[：:]")
INLINE_ADDITIVE_START_PATTERN = re.compile(r"CNS\s*号\s*[：:].*INS\s*号\s*[：:]")
CNS_TOKEN_PATTERN = re.compile(r"\s*CNS\s*号\s*[：:]")
ADDITIVE_NAME_PATTERN = re.compile(
    r"[0-9A-Za-z\u4e00-\u9fff\u0370-\u03FF'’\-\+、，,·\(\)\s]+(?:\([^\)]{1,80}\))?（[^）]{1,160}）"
)
ADDITIVE_NAME_START_PATTERN = re.compile(
    r"^\s*[0-9A-Za-z\u4e00-\u9fff\u0370-\u03FF'’\-\+、，,·\(\)\s]+(?:\([^\)]{1,80}\))?（[^）]{1,160}）"
)


def _split_line_by_additive_names(line: str) -> list[str]:
    """Only treat additive names at line-start as valid entry starts."""
    return [line]


def _classify_heading(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s:
        return None

    for pat, typ in (
        (CHAPTER_PATTERN, "chapter"),
        (SECTION_PATTERN, "section"),
        (ARTICLE_PATTERN, "article"),
    ):
        m = pat.match(s)
        if m:
            return typ, m.group(1)

    m = APPENDIX_PATTERN.match(s)
    if m:
        return "appendix", f"附录{m.group(1)}"

    m = APPENDIX_ITEM_PATTERN.match(s)
    if m:
        return "appendix_item", f"{m.group(1)}.{m.group(2)}"

    m = CN_ENUM_PATTERN.match(s)
    if m:
        return "cn_enum", f"{m.group(1)}、"

    m = CN_PAREN_ENUM_PATTERN.match(s)
    if m:
        return "cn_paren_enum", f"（{m.group(1)}）"

    m = NUMERIC_PATTERN.match(s)
    if m:
        level = m.group(1).count(".") + 1
        return f"numeric_l{level}", m.group(1)

    return None


def _base_meta() -> dict:
    return {
        "heading_type": "preamble",
        "heading_key": "导言",
        "chapter": "",
        "section": "",
        "article": "",
        "clause": "",
        "item": "",
        "appendix": "",
        "appendix_item": "",
        "cn_enum": "",
        "cn_paren_enum": "",
        "table_id": "",
        "table_title": "",
        "additive_name": "",
    }


def _split_structured(text: str, keep_enum_in_article: bool) -> list[dict]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    chunks: list[dict] = []
    current_lines: list[str] = []
    current = _base_meta()

    def flush() -> None:
        if current_lines:
            chunks.append({**current, "text": "\n".join(current_lines).strip()})

    for line in lines:
        heading = _classify_heading(line)
        if heading is None:
            current_lines.append(line)
            continue

        h_type, h_key = heading
        if keep_enum_in_article and h_type in {"cn_enum", "cn_paren_enum"} and current.get("article"):
            current_lines.append(line)
            continue

        flush()
        current_lines = [line]

        if h_type in {"chapter", "section", "article", "appendix", "appendix_item", "cn_enum", "cn_paren_enum"}:
            current["heading_type"] = h_type
            current["heading_key"] = h_key
            if h_type in current:
                current[h_type] = h_key
            if h_type == "chapter":
                current.update({"chapter": h_key, "section": "", "article": "", "clause": "", "item": ""})
            elif h_type == "section":
                current.update({"section": h_key, "article": "", "clause": "", "item": ""})
            elif h_type == "article":
                current.update({"article": h_key, "clause": "", "item": ""})
            elif h_type == "appendix":
                current.update({"appendix": h_key, "appendix_item": ""})
            elif h_type == "appendix_item":
                current.update({"appendix_item": h_key})
        elif h_type.startswith("numeric_l"):
            lvl = int(h_type.split("l", 1)[1])
            current.update({"heading_type": h_type, "heading_key": h_key})
            if lvl == 1:
                current.update({"article": h_key, "clause": "", "item": ""})
            elif lvl == 2:
                current.update({"clause": h_key, "item": ""})
            else:
                current.update({"item": h_key})

    flush()
    return [c for c in chunks if c.get("text")]


def _split_table_aware(text: str, rows_per_chunk: int = 10) -> list[dict]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    result: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = TABLE_ANY_PATTERN.match(line)
        if not m:
            start = i
            i += 1
            while i < len(lines) and not TABLE_PATTERN.match(lines[i]):
                i += 1
            block = "\n".join(lines[start:i])
            result.extend(_split_structured(block, keep_enum_in_article=False))
            continue

        table_id = f"表{m.group(1)}"
        table_title = line.strip()
        i += 1
        rows: list[str] = []
        while i < len(lines) and not TABLE_PATTERN.match(lines[i]):
            rows.append(lines[i])
            i += 1

        if rows:
            for idx in range(0, len(rows), rows_per_chunk):
                part = rows[idx: idx + rows_per_chunk]
                result.append(
                    {
                        **_base_meta(),
                        "heading_type": "table_rows",
                        "heading_key": table_id,
                        "table_id": table_id,
                        "table_title": table_title,
                        "text": "\n".join([table_title, *part]).strip(),
                    }
                )
        else:
            result.append(
                {
                    **_base_meta(),
                    "heading_type": "table",
                    "heading_key": table_id,
                    "table_id": table_id,
                    "table_title": table_title,
                    "text": table_title,
                }
            )

    return result


def _merge_numeric_siblings(sections: list[dict], max_merged_chars: int = 2800) -> list[dict]:
    """Merge contiguous numeric_l2/l3 siblings like 4.1~4.5 into a single chunk."""
    if not sections:
        return sections

    merged: list[dict] = []
    i = 0
    while i < len(sections):
        cur = sections[i]
        htype = str(cur.get("heading_type", ""))
        hkey = str(cur.get("heading_key", ""))

        if not htype.startswith("numeric_l") or "." not in hkey:
            merged.append(cur)
            i += 1
            continue

        parent = hkey.split(".", 1)[0]
        group = [cur]
        j = i + 1
        total_len = len(cur.get("text", ""))

        while j < len(sections):
            nxt = sections[j]
            ntype = str(nxt.get("heading_type", ""))
            nkey = str(nxt.get("heading_key", ""))
            if not ntype.startswith("numeric_l") or "." not in nkey:
                break
            if nkey.split(".", 1)[0] != parent:
                break
            ntext = nxt.get("text", "")
            if total_len + len(ntext) > max_merged_chars:
                break
            group.append(nxt)
            total_len += len(ntext)
            j += 1

        if len(group) == 1:
            merged.append(cur)
            i += 1
            continue

        base = dict(group[0])
        base["heading_type"] = "numeric_group"
        base["heading_key"] = f"{parent}.*"
        base["clause"] = f"{parent}.*"
        base["text"] = "\n".join(g.get("text", "") for g in group if g.get("text"))
        merged.append(base)
        i = j

    return merged


def _merge_heading_only_chunks(sections: list[dict], max_heading_chars: int = 30) -> list[dict]:
    """Merge heading-only chunks (e.g. '2 术语和定义') into the next chunk."""
    if not sections:
        return sections

    out: list[dict] = []
    i = 0
    while i < len(sections):
        cur = sections[i]
        text = str(cur.get("text", "")).strip()
        if i + 1 < len(sections) and len(text) <= max_heading_chars and "\n" not in text:
            nxt = dict(sections[i + 1])
            nxt_text = str(nxt.get("text", "")).strip()
            nxt["text"] = f"{text}\n{nxt_text}" if nxt_text else text
            out.append(nxt)
            i += 2
            continue

        out.append(cur)
        i += 1

    return out


def _split_long_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return out


def _split_additive_blocks(text: str, max_block_chars: int = 1800) -> list[dict]:
    """For GB2760-style additive entries, keep one additive as one chunk; split only if too long."""
    # First, force-split compressed paragraphs where multiple additive entries are on one line.
    normalized = CNS_TOKEN_PATTERN.sub("\nCNS号：", text)
    normalized = re.sub(r"\s*INS\s*号\s*[：:]", "\nINS号：", normalized)
    normalized = re.sub(r"\s*功能\s*[：:]", "\n功能：", normalized)
    normalized = re.sub(r"\s*食品分类号", "\n食品分类号", normalized)
    raw_lines = normalized.splitlines()
    if not raw_lines:
        return []

    sections: list[dict] = []
    current_lines: list[str] = []

    def flush() -> None:
        if not current_lines:
            return
        block_text = "\n".join(current_lines).strip()
        additive_name = current_lines[0].strip() if current_lines else ""
        if re.match(r"^\s*CNS\s*号", additive_name):
            for ln in current_lines[1:]:
                if ADDITIVE_NAME_START_PATTERN.match(ln):
                    additive_name = ln.strip()
                    break
        if len(block_text) <= max_block_chars:
            sections.append({**_base_meta(), "heading_type": "additive_entry", "heading_key": "additive", "additive_name": additive_name, "text": block_text})
        else:
            rows = block_text.splitlines()
            head_rows: list[str] = []
            food_rows: list[str] = []
            for r in rows:
                if FOOD_CLASS_PATTERN.match(r):
                    food_rows.append(r)
                elif not food_rows:
                    head_rows.append(r)
                else:
                    food_rows.append(r)

            if food_rows:
                chunk: list[str] = head_rows.copy()
                cur_len = sum(len(x) for x in chunk)
                for fr in food_rows:
                    if cur_len + len(fr) > max_block_chars and len(chunk) > len(head_rows):
                        sections.append({**_base_meta(), "heading_type": "additive_entry_part", "heading_key": "additive", "additive_name": additive_name, "text": "\n".join(chunk)})
                        chunk = head_rows.copy() + [fr]
                        cur_len = sum(len(x) for x in chunk)
                    else:
                        chunk.append(fr)
                        cur_len += len(fr)
                if chunk:
                    sections.append({**_base_meta(), "heading_type": "additive_entry_part", "heading_key": "additive", "additive_name": additive_name, "text": "\n".join(chunk)})
                return

            serial_parts = [p.strip() for p in SERIAL_ROW_PATTERN.split(block_text) if p.strip()]
            serial_hits = SERIAL_ROW_PATTERN.findall(block_text)
            if serial_hits and serial_parts:
                rebuilt_rows = [f"{serial_hits[idx]}{serial_parts[idx]}" for idx in range(min(len(serial_hits), len(serial_parts)))]
                head = []
                if not block_text.lstrip().startswith("序号"):
                    first_serial_pos = block_text.find(serial_hits[0])
                    if first_serial_pos > 0:
                        head_text = block_text[:first_serial_pos].strip()
                        if head_text:
                            head = head_text.splitlines()
                chunk: list[str] = head.copy()
                cur_len = sum(len(x) for x in chunk)
                for row in rebuilt_rows:
                    if cur_len + len(row) > max_block_chars and len(chunk) > len(head):
                        sections.append({**_base_meta(), "heading_type": "additive_table_part", "heading_key": "additive", "additive_name": additive_name, "text": "\n".join(chunk)})
                        chunk = head.copy() + [row]
                        cur_len = sum(len(x) for x in chunk)
                    else:
                        chunk.append(row)
                        cur_len += len(row)
                if chunk:
                    sections.append({**_base_meta(), "heading_type": "additive_table_part", "heading_key": "additive", "additive_name": additive_name, "text": "\n".join(chunk)})
                return

            # fallback: for single-line long entries, split by food-class items separated with '；食品分类号'
            if "食品分类号" in block_text and "；" in block_text:
                pieces = [p.strip() for p in re.split(r"；\s*(?=食品分类号)", block_text) if p.strip()]
                if len(pieces) > 1:
                    head = pieces[0]
                    rest = pieces[1:]
                    chunk = head
                    for it in rest:
                        cand = f"{chunk}；{it}"
                        if len(cand) > max_block_chars and chunk != head:
                            sections.append({**_base_meta(), "heading_type": "additive_entry_part", "heading_key": "additive", "additive_name": additive_name, "text": chunk})
                            chunk = f"{head}；{it}"
                        else:
                            chunk = cand
                    if chunk:
                        sections.append({**_base_meta(), "heading_type": "additive_entry_part", "heading_key": "additive", "additive_name": additive_name, "text": chunk})
                    return

            for part in _split_long_text(block_text, chunk_size=1000, overlap=140):
                sections.append({**_base_meta(), "heading_type": "additive_entry_part", "heading_key": "additive", "additive_name": additive_name, "text": part})

    for raw in raw_lines:
        line = raw.rstrip()
        if not line.strip():
            if current_lines:
                flush()
                current_lines = []
            continue

        for piece in _split_line_by_additive_names(line):
            piece = piece.strip()
            if not piece:
                continue

            heading = _classify_heading(piece)
            if heading and current_lines:
                flush()
                current_lines = [piece]
                continue

            # Start a new chunk only when additive-name appears at line start.
            if current_lines and ADDITIVE_NAME_START_PATTERN.match(piece):
                has_cns_already = any(re.search(r"CNS\s*号", x) for x in current_lines)
                if has_cns_already:
                    flush()
                    current_lines = [piece]
                    continue

            if current_lines and INLINE_ADDITIVE_START_PATTERN.search(piece):
                has_cns_already = any(re.search(r"CNS\s*号", x) for x in current_lines)
                if has_cns_already:
                    flush()
                    current_lines = [piece]
                    continue

            if current_lines and not FOOD_CLASS_PATTERN.match(piece) and FOOD_CLASS_PATTERN.match(current_lines[-1]):
                # If next line is actually metadata continuation (CNS/INS/功能), keep in same additive.
                if re.match(r"^\s*CNS\s*号", piece) or re.match(r"^\s*INS\s*号", piece) or piece.startswith("功能"):
                    current_lines.append(piece)
                    continue
                flush()
                current_lines = [piece]
                continue

            current_lines.append(piece)

    flush()
    return sections


def _build_documents_from_file(file_path: Path, split_mode: SplitMode = "law") -> list[Document]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []

    if split_mode == "table_aware":
        sections = _split_table_aware(text)
    elif split_mode == "standard":
        sections = _split_structured(text, keep_enum_in_article=False)
        sections = _merge_numeric_siblings(sections)
        sections = _merge_heading_only_chunks(sections)
    elif split_mode == "additive":
        preface_sections = _split_structured(text, keep_enum_in_article=True)
        preface_sections = _merge_numeric_siblings(preface_sections)
        preface_sections = _merge_heading_only_chunks(preface_sections)

        # Keep only real preface chunks; drop appendix/detail blocks that contain additive lists.
        preface_sections = [
            s for s in preface_sections
            if str(s.get("heading_type", "")) not in {"appendix", "appendix_item"}
            and not re.search(r"CNS\s*号", str(s.get("text", "")))
        ]

        additive_sections = _split_additive_blocks(text)

        sections = preface_sections
        if additive_sections:
            sections.extend(
                s
                for s in additive_sections
                if str(s.get("heading_type", "")).startswith("additive")
                and (
                    bool(re.search(r"CNS\s*号", str(s.get("text", ""))))
                    or "食品分类号" in str(s.get("text", ""))
                    or "序号" in str(s.get("text", ""))
                )
            )
    else:
        sections = _split_structured(text, keep_enum_in_article=True)
        sections = _merge_heading_only_chunks(sections)

    if not sections:
        sections = [{**_base_meta(), "heading_type": "fulltext", "heading_key": "全文", "text": text}]

    docs: list[Document] = []
    for idx, sec in enumerate(sections, start=1):
        text = sec["text"]
        if split_mode == "additive" and str(sec.get("heading_type", "")).startswith("additive"):
            parts = [text]
        else:
            parts = _split_long_text(text)

        for part_idx, chunk in enumerate(parts, start=1):
            docs.append(
                Document(
                    text=chunk,
                    metadata={
                        "source_file": str(file_path),
                        "file_name": file_path.name,
                        **{k: v for k, v in sec.items() if k != "text"},
                        "section_index": idx,
                        "part_index": part_idx,
                        "split_mode": split_mode,
                    },
                )
            )
    return docs


def _write_chunks_preview(preview_file: Path, docs: Iterable[Document]) -> None:
    preview_file.parent.mkdir(parents=True, exist_ok=True)
    with preview_file.open("w", encoding="utf-8") as f:
        for i, d in enumerate(docs, start=1):
            md = d.metadata or {}
            f.write(f"===== CHUNK {i} =====\n")
            for k in [
                "file_name",
                "split_mode",
                "heading_type",
                "heading_key",
                "chapter",
                "section",
                "article",
                "clause",
                "item",
                "appendix",
                "appendix_item",
                "cn_enum",
                "cn_paren_enum",
                "table_id",
                "table_title",
                "additive_name",
            ]:
                f.write(f"{k}: {md.get(k, '')}\n")
            f.write("--- text ---\n")
            f.write((d.text or "").strip())
            f.write("\n\n")


def export_chunks_only(
    source_file: str,
    chunks_output_file: str = "outputs/chunks_preview.txt",
    split_mode: SplitMode = "law",
) -> str:
    file_path = Path(source_file)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    docs = _build_documents_from_file(file_path, split_mode=split_mode)
    if not docs:
        raise ValueError(f"No readable chunks from file: {file_path}")

    output_path = Path(chunks_output_file)
    _write_chunks_preview(output_path, docs)
    return str(output_path)


class DashScopeTextEmbedding(BaseEmbedding):
    """LlamaIndex embedding adapter for DashScope OpenAI-compatible embeddings."""

    model_name: str = "text-embedding-v4"
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @property
    def _client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _embed_text(self, text: str) -> Embedding:
        resp = self._client.embeddings.create(model=self.model_name, input=text)
        return list(resp.data[0].embedding)

    def _embed_texts(self, texts: List[str]) -> List[Embedding]:
        resp = self._client.embeddings.create(model=self.model_name, input=texts)
        return [list(x.embedding) for x in resp.data]

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._embed_text(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._embed_text(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._embed_text(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return self._embed_texts(texts)


def _parse_chunks_preview_file(chunks_file: Path) -> list[Document]:
    text = chunks_file.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []

    docs: list[Document] = []
    blocks = [b.strip() for b in text.split("===== CHUNK ") if b.strip()]
    for b in blocks:
        lines = b.splitlines()
        if not lines:
            continue

        meta: dict[str, str] = {"source_file": str(chunks_file), "file_name": chunks_file.name}
        i = 1  # skip first line like "12 ====="
        while i < len(lines) and lines[i].strip() != "--- text ---":
            ln = lines[i]
            if ":" in ln:
                k, v = ln.split(":", 1)
                meta[k.strip()] = v.strip()
            i += 1

        text_body = ""
        if i < len(lines) and lines[i].strip() == "--- text ---":
            text_body = "\n".join(lines[i + 1:]).strip()

        if text_body:
            docs.append(Document(text=text_body, metadata=meta))

    return docs


def _setup_embed_model(embed_model: str = "mock") -> None:
    if embed_model == "mock":
        Settings.embed_model = MockEmbedding(embed_dim=384)
        return

    if embed_model == "text-embedding-v4":
        cfg = SkillConfig()
        api_key = os.getenv("DASHSCOPE_API_KEY") or cfg.dashscope_api_key
        if not api_key or api_key == "你的key":
            raise ValueError("DASHSCOPE_API_KEY is required for text-embedding-v4")
        base_url = os.getenv("DASHSCOPE_BASE_URL") or cfg.dashscope_base_url
        Settings.embed_model = DashScopeTextEmbedding(
            model_name="text-embedding-v4",
            api_key=api_key,
            base_url=base_url,
        )
        return

    raise ValueError(f"Unsupported embed model: {embed_model}")


def build_index(
    source_dir: str = "data/cleaned",
    persist_dir: str = "vector_store/food_label_rules",
    chunks_preview_file: str = "outputs/chunks_preview.txt",
    split_mode: SplitMode = "law",
    embed_model: str = "mock",
    chunks_files: list[str] | None = None,
    append: bool = False,
) -> str:
    src = Path(source_dir)
    dst = Path(persist_dir)
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    _setup_embed_model(embed_model)

    docs: list[Document] = []
    if chunks_files:
        for cf in chunks_files:
            p = Path(cf)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"Chunks file not found: {p}")
            docs.extend(_parse_chunks_preview_file(p))
    else:
        for f in src.rglob("*"):
            if f.is_file() and f.suffix.lower() in {".txt", ".md"}:
                docs.extend(_build_documents_from_file(f, split_mode=split_mode))

    if not docs:
        raise ValueError(f"No readable regulation documents found in: {src}")

    _write_chunks_preview(Path(chunks_preview_file), docs)

    if append and (dst / "index_store.json").exists():
        storage_context = StorageContext.from_defaults(persist_dir=str(dst))
        index = load_index_from_storage(storage_context)
        for d in docs:
            index.insert(d)
    else:
        index = VectorStoreIndex.from_documents(docs)

    index.storage_context.persist(persist_dir=str(dst))
    return str(dst)


def main() -> None:
    path = build_index()
    print(f"Index built at: {path}")


if __name__ == "__main__":
    main()
