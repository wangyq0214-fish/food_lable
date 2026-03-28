from dataclasses import dataclass, field
from typing import Any


@dataclass
class LabelDocument:
    source_path: str
    raw_text: str = ""
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class Issue:
    issue_id: str
    title: str
    severity: str
    category: str
    description: str
    suggestion: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    standard_refs: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AuditResult:
    document_id: str
    product_name: str
    issues: list[Issue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

