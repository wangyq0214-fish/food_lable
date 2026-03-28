from __future__ import annotations

import json
from pathlib import Path

from openclaw_skill.agents.basic_agents import (
    AdditivesComplianceAgent,
    BasicLabelComplianceAgent,
    ClaimComplianceAgent,
    NutritionComplianceAgent,
    TextNormalizationAgent,
)
from openclaw_skill.config import SkillConfig
from openclaw_skill.models.schemas import AuditResult
from openclaw_skill.retrieval.retriever import RuleRetriever
from openclaw_skill.services.parser import LabelParserService
from openclaw_skill.services.reporter import DocxReportService


class FoodLabelAuditSkill:
    """Minimal OpenClaw skill workflow scaffold."""

    def __init__(self, config: SkillConfig | None = None):
        self.config = config or SkillConfig()
        self.parser = LabelParserService(self.config)
        self.reporter = DocxReportService()
        self.retriever = RuleRetriever(
            persist_dir=self.config.retrieval_index_dir,
            embed_model=self.config.embedding_model,
        )
        self.agents = [
            BasicLabelComplianceAgent(),
            NutritionComplianceAgent(),
            ClaimComplianceAgent(),
            AdditivesComplianceAgent(),
            TextNormalizationAgent(),
        ]

    def run(
        self,
        source_path: str,
        debug_ocr: bool = False,
        debug_ocr_file: str = "outputs/ocr_raw.json",
        debug_parsed_file: str | None = None,
    ) -> tuple[AuditResult, str]:
        doc = self.parser.parse(source_path)

        result = AuditResult(
            document_id=Path(source_path).stem,
            product_name=doc.fields.get("product_name") or "",
        )

        for agent in self.agents:
            agent.run(doc, result)

        # Attach hybrid retrieval references to each issue for downstream report traceability.
        for issue in result.issues:
            query = " ".join(x for x in [issue.title, issue.description] if x).strip()
            if not query:
                continue
            try:
                retrieval = self.retriever.search(query, top_k=self.config.top_k_references)
                if retrieval:
                    issue.evidence = issue.evidence or {}
                    issue.evidence["retrieval"] = retrieval[0].get("sources", [])
            except Exception:
                # Retrieval failure should not break main audit flow.
                pass

        result.summary = {
            "issue_count": len(result.issues),
            "high_risk_count": sum(1 for i in result.issues if i.severity == "high"),
        }

        if debug_ocr:
            debug_path = Path(debug_ocr_file)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self.parser.last_ocr_raw_payload
            if payload is None:
                payload = (
                    '{"message":"No OCR payload generated. Input may be non-image (e.g. .docx/.txt) or OCR was not called."}'
                )
            debug_path.write_text(payload, encoding="utf-8")

        if debug_parsed_file:
            parsed_path = Path(debug_parsed_file)
            parsed_path.parent.mkdir(parents=True, exist_ok=True)
            parsed_path.write_text(
                json.dumps(self.parser.last_parsed_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        output_docx = str(self.config.output_dir / f"{result.document_id}_审核报告.docx")
        output_path = self.reporter.render(result, output_docx)
        return result, output_path
