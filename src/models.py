"""Shared domain models used by the UI, API, and analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Requirement:
    """A single extracted requirement."""

    id: int
    section: str
    text: str
    category: str  # technical, sla, legal, commercial, security, other
    tables: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "section": self.section,
            "text": self.text,
            "category": self.category,
            "tables": self.tables,
        }


@dataclass
class RequirementVerdict:
    """Verdict for a single requirement."""

    requirement_id: int
    section: str
    requirement_text: str
    category: str
    verdict: str  # match, partial, mismatch, needs_clarification
    confidence: float
    reasoning: str
    evidence: str
    recommendation: str
    source_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "requirement_id": self.requirement_id,
            "section": self.section,
            "requirement_text": self.requirement_text,
            "category": self.category,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "source_urls": self.source_urls,
        }


@dataclass
class AnalysisReport:
    """Full analysis report."""

    document_name: str
    verdicts: list[RequirementVerdict] = field(default_factory=list)
    summary: str = ""

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def match_count(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "match")

    @property
    def partial_count(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "partial")

    @property
    def mismatch_count(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "mismatch")

    @property
    def clarification_count(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "needs_clarification")

    @property
    def score(self) -> int:
        s = 0
        for v in self.verdicts:
            if v.verdict == "match":
                s += 2
            elif v.verdict == "partial":
                s += 1
        return s

    @property
    def max_score(self) -> int:
        return len(self.verdicts) * 2

    @property
    def compliance_percentage(self) -> float:
        if not self.verdicts:
            return 0.0
        return round(self.score / self.max_score * 100, 1)

    def to_dict(self) -> dict:
        return {
            "document_name": self.document_name,
            "summary": self.summary,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "total": self.total,
            "match_count": self.match_count,
            "partial_count": self.partial_count,
            "mismatch_count": self.mismatch_count,
            "clarification_count": self.clarification_count,
            "score": self.score,
            "max_score": self.max_score,
            "compliance_percentage": self.compliance_percentage,
        }
