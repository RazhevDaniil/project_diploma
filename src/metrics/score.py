"""Compose components into a single Quality score per ТЗ and across runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from src.metrics.components import (
    DetectionStats,
    citation_score,
    compliance_closeness,
    detection_f2,
    quadratic_kappa,
)
from src.metrics.judge import JudgeItemResult, judge_pairs
from src.metrics.match import MatchResult, match_by_requirement_id

# Default component weights. Tunable per project policy.
DEFAULT_WEIGHTS = {
    "detection_f2": 0.40,
    "verdict_kappa": 0.20,
    "reasoning_judge": 0.25,
    "citation_acc": 0.10,
    "compliance_closeness": 0.05,
}


@dataclass
class QualityResult:
    document_name: str
    quality_score: float  # 0..100
    components: dict
    totals: dict
    deviations: list[dict]
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def to_dict(self) -> dict:
        return asdict(self)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _summarize_deviations(
    matches: MatchResult,
    judge_results: list[JudgeItemResult],
    top_n: int = 25,
    judge_was_run: bool = True,
) -> list[dict]:
    judge_by_id = {r.requirement_id: r for r in judge_results}
    deviations: list[dict] = []
    for pair in matches.pairs:
        ref_v = pair.reference.get("verdict")
        cand_v = pair.candidate.get("verdict")
        judge = judge_by_id.get(pair.requirement_id)
        verdict_changed = ref_v != cand_v
        low_judge = (
            judge_was_run
            and judge is not None
            and judge.comment != "judge_skipped_offline"
            and judge.score < 0.6
        )
        if not verdict_changed and not low_judge:
            continue
        deviations.append(
            {
                "requirement_id": pair.requirement_id,
                "section": pair.section,
                "requirement_text": pair.requirement_text[:240],
                "ref_verdict": ref_v,
                "cand_verdict": cand_v,
                "judge_score": round(judge.score, 3) if judge else None,
                "comment": (judge.comment if judge else "")[:240],
            }
        )
    # Surface "missed risks" first (ref=problem, cand=match).
    problem_set = {"partial", "mismatch", "needs_clarification"}
    deviations.sort(
        key=lambda d: (
            0 if (d["ref_verdict"] in problem_set and d["cand_verdict"] == "match") else 1,
            d.get("judge_score") if d.get("judge_score") is not None else 1.0,
        )
    )
    return deviations[:top_n]


def score_run(
    reference_report: dict,
    candidate_report: dict,
    weights: Optional[dict] = None,
    skip_judge: bool = False,
    llm_call=None,
) -> QualityResult:
    """Compute Quality for one ТЗ pair."""
    weights = dict(DEFAULT_WEIGHTS) if weights is None else {**DEFAULT_WEIGHTS, **weights}
    matches = match_by_requirement_id(reference_report, candidate_report)

    detection: DetectionStats = detection_f2(matches.pairs)
    kappa = quadratic_kappa(matches.pairs)
    citation = citation_score(matches.pairs)
    compl = compliance_closeness(reference_report, candidate_report)
    judge_results = judge_pairs(matches.pairs, llm_call=llm_call, skip_when_offline=skip_judge)
    reasoning = (
        sum(r.score for r in judge_results) / len(judge_results) if judge_results else 0.0
    )

    components = {
        "detection_f2": _clip01(detection.f2),
        "verdict_kappa": _clip01(kappa),
        "reasoning_judge": _clip01(reasoning),
        "citation_acc": _clip01(citation),
        "compliance_closeness": _clip01(compl),
    }
    quality_unit = sum(components[name] * weights.get(name, 0.0) for name in components)
    quality_score = round(100.0 * quality_unit, 2)

    totals = {
        "matched_items": len(matches.pairs),
        "only_in_reference": len(matches.only_in_reference),
        "only_in_candidate": len(matches.only_in_candidate),
        "tp": detection.tp,
        "fp": detection.fp,
        "fn": detection.fn,
        "tn": detection.tn,
        "precision": round(detection.precision, 3),
        "recall": round(detection.recall, 3),
        "compliance_ref": float(reference_report.get("compliance_percentage", 0.0) or 0.0),
        "compliance_cand": float(candidate_report.get("compliance_percentage", 0.0) or 0.0),
        "judge_skipped": skip_judge,
    }
    deviations = _summarize_deviations(matches, judge_results, judge_was_run=not skip_judge)

    return QualityResult(
        document_name=reference_report.get("document_name") or candidate_report.get("document_name", ""),
        quality_score=quality_score,
        components={k: round(v, 3) for k, v in components.items()},
        totals=totals,
        deviations=deviations,
        weights=weights,
    )


def aggregate_runs(results: Iterable[QualityResult]) -> dict:
    """Macro-average across N ТЗ."""
    items = list(results)
    if not items:
        return {"quality_score": 0.0, "n_runs": 0, "components": {}, "per_run": []}
    component_keys = list(DEFAULT_WEIGHTS.keys())
    avg_components = {
        key: round(sum(r.components.get(key, 0.0) for r in items) / len(items), 3)
        for key in component_keys
    }
    avg_quality = round(sum(r.quality_score for r in items) / len(items), 2)
    return {
        "quality_score": avg_quality,
        "n_runs": len(items),
        "components": avg_components,
        "per_run": [
            {
                "document_name": r.document_name,
                "quality_score": r.quality_score,
                "components": r.components,
                "matched_items": r.totals.get("matched_items"),
            }
            for r in items
        ],
    }


def render_markdown_report(result: QualityResult) -> str:
    """Human-readable Markdown digest for one ТЗ."""
    lines: list[str] = []
    lines.append(f"# Качество анализа: {result.document_name}")
    lines.append("")
    lines.append(f"**Итог: {result.quality_score:.1f} / 100**")
    lines.append("")
    lines.append("## Компоненты")
    lines.append("")
    lines.append("| Компонент | Значение | Вес |")
    lines.append("|---|---:|---:|")
    for key, value in result.components.items():
        weight = result.weights.get(key, 0.0)
        lines.append(f"| {key} | {value:.3f} | {weight:.2f} |")
    lines.append("")
    lines.append("## Сводка")
    lines.append("")
    t = result.totals
    lines.append(f"- Сматченные требования: **{t.get('matched_items')}**")
    lines.append(f"- TP / FP / FN / TN: {t.get('tp')} / {t.get('fp')} / {t.get('fn')} / {t.get('tn')}")
    lines.append(f"- Precision: {t.get('precision')}, Recall: {t.get('recall')}")
    lines.append(f"- Compliance ref vs cand: {t.get('compliance_ref')}% vs {t.get('compliance_cand')}%")
    if t.get("only_in_reference") or t.get("only_in_candidate"):
        lines.append(
            f"- Несовпадение по списку требований: только в reference {t.get('only_in_reference')}, "
            f"только в candidate {t.get('only_in_candidate')}"
        )
    lines.append("")
    if result.deviations:
        lines.append("## Топ расхождений")
        lines.append("")
        lines.append("| ID | Раздел | Ref → Cand | Judge | Комментарий |")
        lines.append("|---:|---|---|---:|---|")
        for dev in result.deviations:
            judge = dev.get("judge_score")
            judge_str = f"{judge:.2f}" if isinstance(judge, (int, float)) else "—"
            comment = (dev.get("comment") or "").replace("|", "\\|")
            lines.append(
                f"| {dev['requirement_id']} | {dev.get('section', '')} | "
                f"{dev.get('ref_verdict')} → {dev.get('cand_verdict')} | {judge_str} | "
                f"{comment} |"
            )
        lines.append("")
    return "\n".join(lines)


def load_report(path: str | Path) -> dict:
    """Load a service `runs/<id>.json` and return the embedded report.

    Accepts either a full RunStore record (with top-level `report` key) or
    a bare AnalysisReport.to_dict() payload.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "report" in payload and isinstance(payload["report"], dict):
        return payload["report"]
    return payload
