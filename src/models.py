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
    evidence_status: str = "unchecked"  # confirmed, weak, missing, downgraded, unchecked
    evidence_contract_notes: list[str] = field(default_factory=list)
    trace: dict = field(default_factory=dict)

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
            "evidence_status": self.evidence_status,
            "evidence_contract_notes": self.evidence_contract_notes,
            "trace": self.trace,
        }


@dataclass
class AnalysisReport:
    """Full analysis report."""

    document_name: str
    verdicts: list[RequirementVerdict] = field(default_factory=list)
    summary: str = ""
    # Раздельные резюме для UI-переключателя «По платформе / Best-case».
    # Заполняются генератором (`_generate_summary`) — оба создаются одним
    # прогоном анализа. `summary` остаётся для обратной совместимости
    # (старые отчёты и API-клиенты, которые не знают про новые поля) —
    # туда дублируется portfolio-вариант.
    summary_platform: str = ""
    summary_portfolio: str = ""
    extraction_summary: dict = field(default_factory=dict)

    # Процедурные пункты закупки (ОКПД, цена, обеспечение заявки и т.п.) не
    # участвуют в технической оценке Cloud.ru. Они помечены в парсере
    # category='procedural', а в анализаторе получают verdict='out_of_scope'.
    # Все счётчики ниже их исключают, чтобы compliance% отражал реальный
    # уровень соответствия именно технической части ТЗ.
    @staticmethod
    def _is_in_scope(verdict) -> bool:
        return (
            (verdict.category or "").lower() != "procedural"
            and verdict.verdict != "out_of_scope"
        )

    @property
    def total(self) -> int:
        return sum(1 for v in self.verdicts if self._is_in_scope(v))

    @property
    def total_with_procedural(self) -> int:
        """Общее число извлечённых требований, включая процедурные."""
        return len(self.verdicts)

    @property
    def procedural_count(self) -> int:
        return sum(1 for v in self.verdicts if not self._is_in_scope(v))

    @property
    def match_count(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "match" and self._is_in_scope(v))

    @property
    def partial_count(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "partial" and self._is_in_scope(v))

    @property
    def mismatch_count(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "mismatch" and self._is_in_scope(v))

    @property
    def clarification_count(self) -> int:
        return sum(
            1 for v in self.verdicts
            if v.verdict == "needs_clarification" and self._is_in_scope(v)
        )

    @property
    def score(self) -> int:
        s = 0
        for v in self.verdicts:
            if not self._is_in_scope(v):
                continue
            if v.verdict == "match":
                s += 2
            elif v.verdict == "partial":
                s += 1
        return s

    @property
    def max_score(self) -> int:
        return self.total * 2

    @property
    def compliance_percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return round(self.score / self.max_score * 100, 1)

    @property
    def recommended_platform(self) -> str:
        """Имя канонической платформы Cloud.ru, на которой максимум match'ей.

        Используется как «главная» рекомендация для пресейла. Учитываются
        только канонические имена (ГосОблако / Облако VMware / Advanced /
        Evolution), не «внешние услуги» и не «не определено». Если по всем
        канонам 0 match'ей — возвращаем пустую строку (рекомендации нет).
        """
        canonical = ("ГосОблако", "Облако VMware", "Advanced", "Evolution")
        scores: dict[str, int] = {name: 0 for name in canonical}
        for verdict in self.verdicts:
            if not self._is_in_scope(verdict):
                continue
            for item in verdict.platform_assessments:
                name = (item.platform_name or "").strip()
                if name not in scores:
                    continue
                if item.verdict == "match":
                    scores[name] += 2
                elif item.verdict == "partial":
                    scores[name] += 1
        # Сортируем по баллам, при равенстве — по приоритету в списке canonical.
        priority = {name: idx for idx, name in enumerate(canonical)}
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], priority[kv[0]]))
        if not ordered or ordered[0][1] == 0:
            return ""
        return ordered[0][0]

    @property
    def recommended_platform_compliance(self) -> float:
        """Процент покрытия требований на рекомендуемой платформе.

        Считается ТОЛЬКО по тем требованиям, для которых эта платформа
        присутствует в platform_assessments. Это и есть «реальный потенциал
        Cloud.ru на выбранной платформе» — то, что пресейл может обещать
        заказчику. Метрика стабильнее `compliance_percentage` к расширению
        парсера: новые пункты, у которых нет оценки выбранной платформы,
        не попадают в знаменатель.
        """
        platform = self.recommended_platform
        if not platform:
            return 0.0
        score = 0
        max_score = 0
        for verdict in self.verdicts:
            if not self._is_in_scope(verdict):
                continue
            for item in verdict.platform_assessments:
                if (item.platform_name or "").strip() != platform:
                    continue
                max_score += 2
                if item.verdict == "match":
                    score += 2
                elif item.verdict == "partial":
                    score += 1
                # mismatch / needs_clarification = 0
                break  # одна оценка платформы на verdict достаточно
        if max_score == 0:
            return 0.0
        return round(score / max_score * 100, 1)

    @property
    def platform_summary(self) -> list[dict]:
        stats: dict[str, dict] = {}
        for verdict in self.verdicts:
            if not self._is_in_scope(verdict):
                continue
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

    @property
    def suspicious_items(self) -> list[dict]:
        items = []
        for verdict in self.verdicts:
            reasons = list(verdict.evidence_contract_notes or [])
            if verdict.verdict == "needs_clarification":
                reasons.append("Требует ручного уточнения")
            if verdict.confidence < 0.55:
                reasons.append(f"Низкая уверенность: {verdict.confidence:.0%}")
            if verdict.requires_external_service:
                reasons.append("Нужна проработка внешних услуг / подрядчиков")
            if verdict.evidence_status in {"missing", "weak", "downgraded"}:
                reasons.append(f"Статус доказательств: {verdict.evidence_status}")

            trace = verdict.trace or {}
            selected_sources = trace.get("selected_sources", []) if isinstance(trace, dict) else []
            if not selected_sources:
                reasons.append("Нет выбранных RAG-источников")
            elif not any(float(source.get("score", 0) or 0) >= 2.0 for source in selected_sources if isinstance(source, dict)):
                reasons.append("RAG-источники имеют слабую релевантность")

            if not reasons:
                continue
            items.append(
                {
                    "requirement_id": verdict.requirement_id,
                    "section": verdict.section,
                    "requirement_text": verdict.requirement_text,
                    "verdict": verdict.verdict,
                    "confidence": verdict.confidence,
                    "reasons": list(dict.fromkeys(reasons)),
                    "recommendation": verdict.recommendation,
                }
            )
        return items

    def to_dict(self) -> dict:
        return {
            "document_name": self.document_name,
            "summary": self.summary,
            "summary_platform": self.summary_platform,
            "summary_portfolio": self.summary_portfolio,
            "extraction_summary": self.extraction_summary,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "total": self.total,
            "total_with_procedural": self.total_with_procedural,
            "procedural_count": self.procedural_count,
            "match_count": self.match_count,
            "partial_count": self.partial_count,
            "mismatch_count": self.mismatch_count,
            "clarification_count": self.clarification_count,
            "score": self.score,
            "max_score": self.max_score,
            "compliance_percentage": self.compliance_percentage,
            "recommended_platform": self.recommended_platform,
            "recommended_platform_compliance": self.recommended_platform_compliance,
            "platform_summary": self.platform_summary,
            "suspicious_items": self.suspicious_items,
        }
