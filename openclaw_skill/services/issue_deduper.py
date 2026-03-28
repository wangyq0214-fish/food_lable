from __future__ import annotations

from openclaw_skill.models.schemas import Issue


TITLE_MAP = {
    "钠标示单位错误": "营养成分单位错误-钠",
    "钠含量单位错误": "营养成分单位错误-钠",
    "营养成分表钠单位错误": "营养成分单位错误-钠",
    "食品名称虚假误导": "食品名称误导风险",
}

SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1}


def normalize_title(title: str) -> str:
    t = (title or "").strip()
    return TITLE_MAP.get(t, t)


def dedupe_issues(issues: list[Issue]) -> list[Issue]:
    merged: dict[tuple[str, str], Issue] = {}
    for issue in issues:
        key = (normalize_title(issue.title), (issue.category or "").strip())
        old = merged.get(key)
        if old is None:
            issue.title = key[0]
            merged[key] = issue
            continue

        old_rank = SEVERITY_ORDER.get(old.severity, 0)
        new_rank = SEVERITY_ORDER.get(issue.severity, 0)
        if new_rank > old_rank:
            issue.title = key[0]
            merged[key] = issue
        elif new_rank == old_rank and len(issue.description or "") > len(old.description or ""):
            issue.title = key[0]
            merged[key] = issue

    return list(merged.values())

