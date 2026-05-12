"""Generate reference report via Claude Opus on the same requirement set the service produced.

CLI: python -m src.metrics.generate_reference --run runs/<id>.json --out reference/<id>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.metrics.prompts import REFERENCE_SYSTEM_PROMPT, REFERENCE_USER_TEMPLATE


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    if not text:
        raise ValueError("empty LLM response")
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError(f"no JSON found in response: {text[:200]!r}")
    return json.loads(match.group(0))


def _load_run_payload(run_path: Path) -> dict:
    return json.loads(run_path.read_text(encoding="utf-8"))


def _build_user_message(payload: dict) -> str:
    document_name = payload.get("document_name") or payload.get("report", {}).get("document_name", "")
    requirements = payload.get("requirements")
    if not requirements:
        # Fall back to verdicts list — strip out evaluation noise, keep text/section/category.
        verdicts = payload.get("report", {}).get("verdicts", []) if isinstance(payload, dict) else []
        requirements = [
            {
                "id": v.get("requirement_id"),
                "section": v.get("section", ""),
                "text": v.get("requirement_text", ""),
                "category": v.get("category", ""),
            }
            for v in verdicts
        ]
    if not requirements:
        raise ValueError(f"no requirements found in {run_path_safe(payload)}")
    return REFERENCE_USER_TEMPLATE.format(
        document_name=document_name,
        requirements_json=json.dumps(requirements, ensure_ascii=False, indent=2),
    )


def run_path_safe(payload: dict) -> str:
    return payload.get("id") or payload.get("document_name") or "<run>"


def generate_reference(
    run_payload: dict,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> dict:
    """Call the project's LLM with the reference prompt; return parsed reference report."""
    from src.llm.client import call_llm

    user_msg = _build_user_message(run_payload)
    raw = call_llm(
        prompt=user_msg,
        system_prompt=REFERENCE_SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
        settings={"openai_model": model} if model else None,
    )
    parsed = _extract_json(raw)
    # Ensure required fields exist.
    parsed.setdefault("document_name", run_payload.get("document_name", ""))
    parsed.setdefault("verdicts", [])
    parsed.setdefault("compliance_percentage", _compute_compliance(parsed["verdicts"]))
    return parsed


def _compute_compliance(verdicts: list[dict]) -> float:
    if not verdicts:
        return 0.0
    score = 0
    for v in verdicts:
        if v.get("verdict") == "match":
            score += 2
        elif v.get("verdict") == "partial":
            score += 1
    return round(score / (len(verdicts) * 2) * 100, 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="src.metrics.generate_reference")
    parser.add_argument("--run", required=True, help="Path to runs/<id>.json from the service.")
    parser.add_argument("--out", required=True, help="Path to write reference JSON.")
    parser.add_argument("--model", default=None, help="Override model id (e.g. claude-opus-4-6).")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args(argv)

    run_path = Path(args.run)
    run_payload = _load_run_payload(run_path)
    reference = generate_reference(
        run_payload,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    Path(args.out).write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stdout.write(
        f"Reference saved: {args.out} (verdicts: {len(reference.get('verdicts', []))}, "
        f"compliance: {reference.get('compliance_percentage')}%)\n"
    )


if __name__ == "__main__":
    main()
