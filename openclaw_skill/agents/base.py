from abc import ABC, abstractmethod

from openclaw_skill.models.schemas import AuditResult, LabelDocument


class BaseAuditAgent(ABC):
    """Base class for all audit agents."""

    name: str = "base"

    @abstractmethod
    def run(self, doc: LabelDocument, result: AuditResult) -> None:
        """Run audit and mutate result in place."""
        raise NotImplementedError
