"""Match verdicts between reference and candidate reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class MatchedPair:
    requirement_id: int
    section: str
    requirement_text: str
    reference: dict  # verdict dict from reference report
    candidate: dict  # verdict dict from candidate report


@dataclass
class MatchResult:
    pairs: list[MatchedPair]
    only_in_reference: list[dict]  # verdicts present only on reference side
    only_in_candidate: list[dict]


def _index(verdicts: Iterable[dict]) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for verdict in verdicts:
        rid = verdict.get("requirement_id")
        if rid is None:
            continue
        result[int(rid)] = verdict
    return result


def match_by_requirement_id(reference_report: dict, candidate_report: dict) -> MatchResult:
    """Match by requirement_id.

    Reference and candidate are expected to be evaluated on the SAME set of
    extracted requirements. Items missing from one side are surfaced explicitly
    so the caller decides how to penalize them.
    """
    ref_index = _index(reference_report.get("verdicts", []))
    cand_index = _index(candidate_report.get("verdicts", []))

    common_ids = sorted(set(ref_index.keys()) & set(cand_index.keys()))
    pairs: list[MatchedPair] = []
    for rid in common_ids:
        ref = ref_index[rid]
        cand = cand_index[rid]
        pairs.append(
            MatchedPair(
                requirement_id=rid,
                section=ref.get("section") or cand.get("section", ""),
                requirement_text=ref.get("requirement_text") or cand.get("requirement_text", ""),
                reference=ref,
                candidate=cand,
            )
        )

    only_ref = [v for rid, v in ref_index.items() if rid not in cand_index]
    only_cand = [v for rid, v in cand_index.items() if rid not in ref_index]
    return MatchResult(pairs=pairs, only_in_reference=only_ref, only_in_candidate=only_cand)
