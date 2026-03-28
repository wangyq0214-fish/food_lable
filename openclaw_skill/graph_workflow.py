from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_RULES_DIR = Path("rules")


def load_rules(rule_files: list[str] | None = None, rules_dir: str | Path = DEFAULT_RULES_DIR) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []

    files: list[Path] = []
    if rule_files:
        files = [Path(file) for file in rule_files]
    else:
        rules_root = Path(rules_dir)
        if rules_root.exists():
            files = sorted(
                [
                    p
                    for p in rules_root.glob("*.json")
                    if p.name != "rule_schema.json"
                ]
            )

    for path in files:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for r in data:
                    if isinstance(r, dict) and r.get("enabled", True):
                        r.setdefault("source_file", str(path).replace("\\", "/"))
                        rules.append(r)
        except Exception:
            continue

    rules.sort(key=lambda r: int(r.get("priority", 100)))
    return rules


def run_rules(text: str, fields: dict[str, Any], rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for rule in rules:
        rtype = rule.get("type")
        if rtype == "regex":
            issue = _eval_regex_rule(text, rule)
        elif rtype == "threshold":
            issue = _eval_threshold_rule(text, fields, rule)
        else:
            issue = None

        if issue is not None:
            issues.append(issue)
    return issues


def _eval_regex_rule(text: str, rule: dict[str, Any]) -> dict[str, Any] | None:
    cond = rule.get("condition", {})
    pattern = cond.get("pattern", "")
    if not pattern:
        return None

    flags = 0
    for f in cond.get("flags", []):
        if f == "IGNORECASE":
            flags |= re.IGNORECASE

    if not re.search(pattern, text, flags=flags):
        return None

    ev_pattern = cond.get("evidence_pattern", pattern)
    m = re.search(ev_pattern, text, flags=flags)
    evidence = m.group(0) if m else ""

    return {
        "title": rule.get("title", "未命名规则"),
        "severity": rule.get("severity", "medium"),
        "category": rule.get("category", "general"),
        "description": rule.get("description_template", ""),
        "suggestion": rule.get("suggestion_template", ""),
        "evidence": evidence,
        "retrieval": [],
        "standard_refs": rule.get("standard_refs", []),
        "rule_id": rule.get("id", ""),
        "source_file": rule.get("source_file", ""),
    }


def _eval_threshold_rule(text: str, fields: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    cond = rule.get("condition", {})
    claim_keyword = str(cond.get("claim_keyword", "")).strip()
    if claim_keyword and claim_keyword not in text:
        return None

    field_path = str(cond.get("field_path", ""))
    value = _deep_get(fields, field_path)
    if value is None:
        return None

    try:
        value_f = float(value)
        threshold = float(cond.get("threshold", 0))
    except Exception:
        return None

    op = cond.get("operator", ">")
    hit = (value_f > threshold) if op == ">" else (value_f >= threshold)
    if not hit:
        return None

    desc = str(rule.get("description_template", "")).format(
        value=value_f,
        threshold=threshold,
        unit=cond.get("unit", ""),
    )
    sug = str(rule.get("suggestion_template", "")).format(
        value=value_f,
        threshold=threshold,
        unit=cond.get("unit", ""),
    )

    return {
        "title": rule.get("title", "未命名规则"),
        "severity": rule.get("severity", "medium"),
        "category": rule.get("category", "general"),
        "description": desc,
        "suggestion": sug,
        "evidence": f"{field_path}={value_f}",
        "retrieval": [],
        "standard_refs": rule.get("standard_refs", []),
        "rule_id": rule.get("id", ""),
        "source_file": rule.get("source_file", ""),
    }


def _deep_get(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur
