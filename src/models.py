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
class PlatformAssessment:
    """Requirement verdict for one Cloud.ru platform or external service group."""

    platform_name: str
    verdict: str  # match, partial, mismatch, needs_clarification
    confidence: float
    reasoning: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    source_titles: list[str] = field(default_factory=list)
    source_type: str = "platform"  # platform, external_service, unknown
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "platform_name": self.platform_name,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence_refs": self.evidence_refs,
            "source_urls": self.source_urls,
            "source_titles": self.source_titles,
            "source_type": self.source_type,
            "recommendation": self.recommendation,
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
    platform_assessments: list[PlatformAssessment] = field(default_factory=list)
    requires_external_service: bool = False
    external_service_notes: str = ""

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
            "platform_assessments": [item.to_dict() for item in self.platform_assessments],
            "requires_external_service": self.requires_external_service,
            "external_service_notes": self.external_service_notes,
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

    @property
    def platform_summary(self) -> list[dict]:
        stats: dict[str, dict] = {}
        for verdict in self.verdicts:
            for item in verdict.platform_assessments:
                name = item.platform_name or "Не определено"
                row = stats.setdefault(
                    name,
                    {
                        "platform_name": name,
                        "source_type": item.source_type,
                        "match_count": 0,
                        "partial_count": 0,
                        "mismatch_count": 0,
                        "clarification_count": 0,
                        "total": 0,
                    },
                )
                row["total"] += 1
                if item.verdict == "match":
                    row["match_count"] += 1
                elif item.verdict == "partial":
                    row["partial_count"] += 1
                elif item.verdict == "mismatch":
                    row["mismatch_count"] += 1
                else:
                    row["clarification_count"] += 1
        return sorted(
            stats.values(),
            key=lambda item: (
                0 if item.get("source_type") == "platform" else 1,
                item.get("platform_name", ""),
            ),
        )

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
            "platform_summary": self.platform_summary,
        }
