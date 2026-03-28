from __future__ import annotations

import re
from datetime import datetime

from openclaw_skill.models.schemas import AuditResult, Issue, LabelDocument


def run_rule_checks(doc: LabelDocument, result: AuditResult) -> None:
    text = doc.raw_text or ""
    _check_trans_fat_zero(text, result)
    _check_sodium_unit(text, result)
    _check_vitamin_c_case(text, result)
    _check_carbohydrate_term(text, result)
    _check_invalid_date(text, result)


def _append(result: AuditResult, issue_id: str, title: str, severity: str, category: str, description: str, suggestion: str) -> None:
    result.issues.append(
        Issue(
            issue_id=issue_id,
            title=title,
            severity=severity,
            category=category,
            description=description,
            suggestion=suggestion,
        )
    )


def _check_sodium_unit(text: str, result: AuditResult) -> None:
    if re.search(r"钠\s*\d+(?:\.\d+)?\s*ng\b", text, flags=re.IGNORECASE):
        _append(
            result,
            "RULE-UNIT-SODIUM",
            "营养成分单位错误-钠",
            "high",
            "nutrition",
            "钠使用了ng单位，营养标签中钠应以mg标示。",
            "将钠单位修正为mg，并复核数值。",
        )


def _check_vitamin_c_case(text: str, result: AuditResult) -> None:
    if "维生素c" in text:
        _append(
            result,
            "RULE-VITC-CASE",
            "术语大小写不规范-维生素C",
            "medium",
            "text",
            "发现“维生素c”写法，建议使用“维生素C”。",
            "统一改为“维生素C”。",
        )


def _check_carbohydrate_term(text: str, result: AuditResult) -> None:
    if "碳水化物" in text:
        _append(
            result,
            "RULE-CARB-TERM",
            "术语不规范-碳水化合物",
            "medium",
            "text",
            "发现“碳水化物”表述，标准术语应为“碳水化合物”。",
            "将“碳水化物”更正为“碳水化合物”。",
        )


def _check_trans_fat_zero(text: str, result: AuditResult) -> None:
    m = re.search(r"反式脂肪酸\s*(\d+(?:\.\d+)?)\s*g", text)
    if not m:
        return
    val = float(m.group(1))
    if 0 < val <= 0.3:
        _append(
            result,
            "RULE-TRANSFAT-ZERO",
            "反式脂肪酸标示可按0处理",
            "medium",
            "nutrition",
            f"反式脂肪酸为{val}g，落在≤0.3g区间，通常可标示为0。",
            "按适用标准核查后可将反式脂肪酸标示为0。",
        )


def _check_invalid_date(text: str, result: AuditResult) -> None:
    for y, m, d in re.findall(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text):
        try:
            datetime(int(y), int(m), int(d))
        except ValueError:
            _append(
                result,
                "RULE-INVALID-DATE",
                "日期标示无效",
                "high",
                "basic_label",
                f"发现无效日期：{y}年{m}月{d}日。",
                "修正为真实存在的日历日期。",
            )
            return
