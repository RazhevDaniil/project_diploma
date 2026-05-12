"""Numeric quality components: F2, quadratic kappa, citation accuracy, compliance MAE."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from src.metrics.match import MatchedPair

# Verdict ordinal scale used for quadratic kappa.
# Order chosen so adjacent labels reflect "how positive": match → partial → ?clarify → mismatch.
_VERDICT_ORDER = {
    "match": 3,
    "partial": 2,
    "needs_clarification": 1,
    "mismatch": 0,
}

_PROBLEM_VERDICTS = {"partial", "mismatch", "needs_clarification"}


@dataclass
class DetectionStats:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f2: float


def detection_f2(pairs: list[MatchedPair]) -> DetectionStats:
    """Binary detection of 'requirement is problematic'.

    Positive = verdict in {partial, mismatch, needs_clarification}.
    """
    tp = fp = fn = tn = 0
    for pair in pairs:
        ref_pos = pair.reference.get("verdict") in _PROBLEM_VERDICTS
        cand_pos = pair.candidate.get("verdict") in _PROBLEM_VERDICTS
        if ref_pos and cand_pos:
            tp += 1
        elif cand_pos and not ref_pos:
            fp += 1
        elif ref_pos and not cand_pos:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if (4 * precision + recall) == 0:
        f2 = 0.0
    else:
        f2 = 5 * precision * recall / (4 * precision + recall)
    # Edge case: no positives in reference at all → degrade to accuracy on negatives.
    if (tp + fn) == 0 and (tp + fp) == 0:
        f2 = 1.0 if tn > 0 else 0.0
        precision = 1.0 if tn > 0 else 0.0
        recall = 1.0 if tn > 0 else 0.0
    return DetectionStats(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f2=f2)


def quadratic_kappa(pairs: list[MatchedPair]) -> float:
    """Cohen's quadratic-weighted kappa over the 4 verdict labels."""
    if not pairs:
        return 0.0
    refs = [_VERDICT_ORDER.get(p.reference.get("verdict", ""), -1) for p in pairs]
    cands = [_VERDICT_ORDER.get(p.candidate.get("verdict", ""), -1) for p in pairs]
    # Drop pairs where either label is unknown (shouldn't happen with validated data).
    filtered = [(r, c) for r, c in zip(refs, cands) if r >= 0 and c >= 0]
    if not filtered:
        return 0.0
    n = len(filtered)
    classes = sorted(set([r for r, _ in filtered] + [c for _, c in filtered]))
    k = len(classes)
    if k == 1:
        # Total agreement on a single class — perfect.
        return 1.0
    idx = {label: i for i, label in enumerate(classes)}
    # Confusion matrix.
    obs = [[0] * k for _ in range(k)]
    for r, c in filtered:
        obs[idx[r]][idx[c]] += 1
    # Marginals.
    row_marg = [sum(obs[i]) for i in range(k)]
    col_marg = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    # Quadratic weights (normalized so corners weigh 1).
    weights = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    # Expected matrix.
    expected = [[(row_marg[i] * col_marg[j]) / n for j in range(k)] for i in range(k)]
    num = sum(weights[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(weights[i][j] * expected[i][j] for i in range(k) for j in range(k))
    if den == 0:
        return 0.0
    kappa = 1.0 - num / den
    return max(-1.0, min(1.0, kappa))


# --- Citation accuracy ---


_TRAILING_SLASH = re.compile(r"/+$")


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip().lower())
    except ValueError:
        return url.strip().lower()
    netloc = parts.netloc
    path = _TRAILING_SLASH.sub("", parts.path)
    return f"{netloc}{path}"


def _section_root(url: str) -> str:
    """Coarse 'docs section' fingerprint: scheme+host + first 2 path segments."""
    if not url:
        return ""
    parts = urlsplit(url.strip().lower())
    segments = [s for s in parts.path.split("/") if s][:3]
    return f"{parts.netloc}/{'/'.join(segments)}"


def citation_score(pairs: list[MatchedPair]) -> float:
    """Average soft-Jaccard over source_urls for matched pairs."""
    if not pairs:
        return 0.0
    scores: list[float] = []
    for pair in pairs:
        ref_urls = {_normalize_url(u) for u in pair.reference.get("source_urls", []) if u}
        cand_urls = {_normalize_url(u) for u in pair.candidate.get("source_urls", []) if u}
        if not ref_urls and not cand_urls:
            # Reference didn't expect a citation; candidate didn't add one. Neutral → 1.
            scores.append(1.0)
            continue
        if not ref_urls and cand_urls:
            # Candidate over-cited; not strictly wrong, mild bonus: 0.6.
            scores.append(0.6)
            continue
        if ref_urls and not cand_urls:
            scores.append(0.0)
            continue
        union = ref_urls | cand_urls
        inter = ref_urls & cand_urls
        jaccard = len(inter) / len(union) if union else 0.0
        if jaccard == 0:
            ref_sections = {_section_root(u) for u in ref_urls}
            cand_sections = {_section_root(u) for u in cand_urls}
            if ref_sections & cand_sections:
                scores.append(0.5)
                continue
        scores.append(jaccard)
    return sum(scores) / len(scores)


def compliance_closeness(reference_report: dict, candidate_report: dict) -> float:
    """1 - |Δ| / 100 over compliance percentage."""
    ref = float(reference_report.get("compliance_percentage", 0.0) or 0.0)
    cand = float(candidate_report.get("compliance_percentage", 0.0) or 0.0)
    return max(0.0, 1.0 - abs(ref - cand) / 100.0)
