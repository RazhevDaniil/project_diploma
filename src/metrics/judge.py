"""LLM-as-judge for reasoning equivalence between reference and candidate verdicts."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

from src.metrics.match import MatchedPair

logger = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = """Ты — строгий судья качества обоснований при оценке соответствия ТЗ облачной платформе.
Тебе будут даны:
- текст требования из ТЗ;
- эталонное обоснование (REFERENCE);
- обоснование, выданное оцениваемым сервисом (CANDIDATE).

Оцени по трём критериям, каждый по шкале 0 / 0.5 / 1:

1) same_requirement — описывают ли REFERENCE и CANDIDATE одно и то же требование? (1 — точно одно; 0.5 — пересечение; 0 — про разное)
2) same_root_cause — совпадает ли причина соответствия / несоответствия / уточнения? (учитывай суть, не формулировку)
3) factual_consistency — содержит ли CANDIDATE факты, противоречащие REFERENCE? (1 — нет противоречий; 0.5 — мелкие неточности; 0 — есть прямое противоречие)

Верни строго JSON в формате:
{"same_requirement": 1, "same_root_cause": 0.5, "factual_consistency": 1, "comment": "одно предложение почему"}

Никакого текста вне JSON."""


JUDGE_USER_TEMPLATE = """Требование (ТЗ, п. {section}):
{requirement_text}

REFERENCE verdict: {ref_verdict}
REFERENCE reasoning:
{ref_reasoning}

CANDIDATE verdict: {cand_verdict}
CANDIDATE reasoning:
{cand_reasoning}

Оцени строго по инструкции."""


@dataclass
class JudgeItemResult:
    requirement_id: int
    same_requirement: float
    same_root_cause: float
    factual_consistency: float
    comment: str
    raw: str = ""

    @property
    def score(self) -> float:
        return (self.same_requirement + self.same_root_cause + self.factual_consistency) / 3.0


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_output(text: str) -> dict:
    if not text:
        return {}
    match = _JSON_BLOCK_RE.search(text)
    blob = match.group(0) if match else text
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        logger.warning("Failed to parse judge output: %s", text[:200])
        return {}


def _coerce(value, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v <= 0:
        return 0.0
    if v >= 1:
        return 1.0
    # Snap to 0 / 0.5 / 1 grid.
    return 0.5


# Type alias for the LLM call surface we depend on.
LLMCallable = Callable[[str, str, float, int], str]


def _default_llm_call(prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
    """Bridge to project's LLM client (Cloud.ru Foundation Models)."""
    from src.llm.client import call_llm

    return call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def judge_pair(
    pair: MatchedPair,
    llm_call: Optional[LLMCallable] = None,
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> JudgeItemResult:
    """Run LLM-as-judge for a single pair."""
    llm = llm_call or _default_llm_call
    prompt = JUDGE_USER_TEMPLATE.format(
        section=pair.section or "—",
        requirement_text=pair.requirement_text,
        ref_verdict=pair.reference.get("verdict", ""),
        ref_reasoning=pair.reference.get("reasoning", ""),
        cand_verdict=pair.candidate.get("verdict", ""),
        cand_reasoning=pair.candidate.get("reasoning", ""),
    )
    try:
        raw = llm(prompt, JUDGE_SYSTEM_PROMPT, temperature, max_tokens)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Judge LLM call failed for requirement %s: %s", pair.requirement_id, exc)
        return JudgeItemResult(
            requirement_id=pair.requirement_id,
            same_requirement=0.0,
            same_root_cause=0.0,
            factual_consistency=0.0,
            comment=f"judge_error: {exc}",
            raw="",
        )
    parsed = _parse_judge_output(raw)
    return JudgeItemResult(
        requirement_id=pair.requirement_id,
        same_requirement=_coerce(parsed.get("same_requirement"), 0.0),
        same_root_cause=_coerce(parsed.get("same_root_cause"), 0.0),
        factual_consistency=_coerce(parsed.get("factual_consistency"), 0.0),
        comment=str(parsed.get("comment", "")).strip()[:300],
        raw=raw,
    )


def judge_pairs(
    pairs: list[MatchedPair],
    llm_call: Optional[LLMCallable] = None,
    skip_when_offline: bool = False,
) -> list[JudgeItemResult]:
    """Run judge over a batch. Returns one result per input pair (in same order)."""
    if skip_when_offline:
        return [
            JudgeItemResult(
                requirement_id=p.requirement_id,
                same_requirement=0.0,
                same_root_cause=0.0,
                factual_consistency=0.0,
                comment="judge_skipped_offline",
            )
            for p in pairs
        ]
    return [judge_pair(p, llm_call=llm_call) for p in pairs]
