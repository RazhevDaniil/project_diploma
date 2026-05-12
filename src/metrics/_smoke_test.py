"""Offline smoke test for the metric. No LLM calls — judge is skipped.

Run:
    cd project_diploma
    python -m src.metrics._smoke_test
"""

from __future__ import annotations

import json

from src.metrics.score import score_run, render_markdown_report


def _make_verdict(rid: int, verdict: str, reasoning: str = "x", urls=None, section: str = "1") -> dict:
    return {
        "requirement_id": rid,
        "section": section,
        "requirement_text": f"Требование {rid}",
        "category": "technical",
        "verdict": verdict,
        "confidence": 0.8,
        "reasoning": reasoning,
        "evidence": "",
        "recommendation": "",
        "source_urls": urls or [],
        "platform_assessments": [],
        "evidence_status": "confirmed",
    }


def main() -> None:
    # Reference: 5 problems out of 10.
    reference = {
        "document_name": "fake_TZ.docx",
        "compliance_percentage": 75.0,
        "verdicts": [
            _make_verdict(1, "match", urls=["https://cloud.ru/docs/a/topics/x"]),
            _make_verdict(2, "match", urls=["https://cloud.ru/docs/a/topics/y"]),
            _make_verdict(3, "partial", reasoning="К1 атт-та нет на Evolution",
                          urls=["https://cloud.ru/documents/security/topics/certificates"]),
            _make_verdict(4, "mismatch", reasoning="Нет ГОСТ 27001-2021",
                          urls=["https://cloud.ru/documents/security/topics/certificates"]),
            _make_verdict(5, "needs_clarification", reasoning="CPU Ready публично нет"),
            _make_verdict(6, "match", urls=["https://cloud.ru/docs/vmware/sla"]),
            _make_verdict(7, "partial", urls=["https://cloud.ru/docs/vmware/sla"]),
            _make_verdict(8, "match"),
            _make_verdict(9, "mismatch", reasoning="Colocation отсутствует"),
            _make_verdict(10, "match", urls=["https://cloud.ru/docs/overview"]),
        ],
    }
    # Candidate: same on most, missed two problems, hallucinated one.
    candidate = {
        "document_name": "fake_TZ.docx",
        "compliance_percentage": 85.0,
        "verdicts": [
            _make_verdict(1, "match", urls=["https://cloud.ru/docs/a/topics/x"]),
            _make_verdict(2, "match", urls=["https://cloud.ru/docs/a/topics/y"]),
            _make_verdict(3, "match",  # MISSED problem (FN)
                          urls=["https://cloud.ru/documents/security/topics/certificates"]),
            _make_verdict(4, "partial",  # softer than ref (mismatch)
                          urls=["https://cloud.ru/documents/security/topics/certificates"]),
            _make_verdict(5, "needs_clarification"),
            _make_verdict(6, "partial",  # FALSE positive
                          urls=["https://cloud.ru/docs/vmware/sla"]),
            _make_verdict(7, "partial", urls=["https://cloud.ru/docs/advanced/sla"]),  # different doc
            _make_verdict(8, "match"),
            _make_verdict(9, "mismatch", reasoning="Colocation вне публичного портфеля"),
            _make_verdict(10, "match", urls=["https://cloud.ru/docs/overview"]),
        ],
    }

    result = score_run(reference, candidate, skip_judge=True)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print("---")
    print(render_markdown_report(result))


if __name__ == "__main__":
    main()
