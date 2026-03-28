from __future__ import annotations

from openclaw_skill.agents.base import BaseAuditAgent
from openclaw_skill.models.schemas import AuditResult, Issue, LabelDocument


class BasicLabelComplianceAgent(BaseAuditAgent):
    name = "basic_label_compliance"

    def run(self, doc: LabelDocument, result: AuditResult) -> None:
        # TODO: replace with GB7718 real checks
        if not doc.fields.get("product_name"):
            result.issues.append(
                Issue(
                    issue_id="BASIC-001",
                    title="缺少产品名称",
                    severity="high",
                    category="basic_label",
                    description="未识别到产品名称字段",
                    suggestion="请在标签显著位置标示产品名称",
                )
            )


class NutritionComplianceAgent(BaseAuditAgent):
    name = "nutrition_compliance"

    def run(self, doc: LabelDocument, result: AuditResult) -> None:
        # TODO: replace with GB28050 real checks
        nutrition = doc.fields.get("nutrition", {})
        if nutrition and "energy_kj" not in nutrition:
            result.issues.append(
                Issue(
                    issue_id="NUTR-001",
                    title="营养成分表缺少能量",
                    severity="medium",
                    category="nutrition",
                    description="识别到营养成分表但缺少能量项",
                    suggestion="补充每100g/100mL的能量值并标注单位kJ",
                )
            )


class ClaimComplianceAgent(BaseAuditAgent):
    name = "claim_compliance"

    def run(self, doc: LabelDocument, result: AuditResult) -> None:
        # TODO: replace with RAG + threshold checks
        claims = doc.fields.get("claims", [])
        sugars = doc.fields.get("nutrition", {}).get("sugar_g_per_100g")
        if "低糖" in claims and sugars is not None and sugars > 5:
            result.issues.append(
                Issue(
                    issue_id="CLM-001",
                    title="低糖声称可能不合规",
                    severity="high",
                    category="claim",
                    description=f"糖含量为 {sugars} g/100g，高于低糖阈值示例 5 g/100g",
                    suggestion="删除低糖声称或调整配方后再标示",
                    evidence={"sugar_g_per_100g": sugars},
                )
            )


class AdditivesComplianceAgent(BaseAuditAgent):
    name = "additives_compliance"

    def run(self, doc: LabelDocument, result: AuditResult) -> None:
        # TODO: replace with GB2760 + ingredients detail checks
        ingredients = doc.fields.get("ingredients", "")
        if "泡打粉" in ingredients:
            result.issues.append(
                Issue(
                    issue_id="ADD-001",
                    title="复配添加剂标示需核查",
                    severity="medium",
                    category="additives",
                    description="检测到“泡打粉”，建议核查是否需要拆分具体添加剂名称",
                    suggestion="按法规要求标示终产品中具有功能作用的每种食品添加剂",
                )
            )


class TextNormalizationAgent(BaseAuditAgent):
    name = "text_normalization"

    def run(self, doc: LabelDocument, result: AuditResult) -> None:
        # TODO: replace with richer typo/traditional-char checks
        raw_text = doc.raw_text or ""
        if "碳水化物" in raw_text:
            result.issues.append(
                Issue(
                    issue_id="TXT-001",
                    title="术语不规范",
                    severity="low",
                    category="text",
                    description="发现“碳水化物”表述，建议使用“碳水化合物”",
                    suggestion="将“碳水化物”更正为“碳水化合物”",
                )
            )
