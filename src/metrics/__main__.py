"""CLI: python -m src.metrics --reference ref.json --candidate cand.json [--out quality.json]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.metrics.score import (
    aggregate_runs,
    load_report,
    render_markdown_report,
    score_run,
)


def _score_pair(args: argparse.Namespace) -> None:
    reference = load_report(args.reference)
    candidate = load_report(args.candidate)
    result = score_run(reference, candidate, skip_judge=args.skip_judge)
    out_path = Path(args.out) if args.out else None
    payload = result.to_dict()
    if out_path:
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    if args.markdown:
        md_path = Path(args.markdown)
        md_path.write_text(render_markdown_report(result), encoding="utf-8")


def _aggregate(args: argparse.Namespace) -> None:
    reference_dir = Path(args.reference_dir)
    candidate_dir = Path(args.candidate_dir)
    refs = {p.stem: p for p in reference_dir.glob("*.json")}
    cands = {p.stem: p for p in candidate_dir.glob("*.json")}
    common = sorted(set(refs) & set(cands))
    if not common:
        raise SystemExit("No matching JSON files in reference and candidate directories.")
    results = []
    for stem in common:
        reference = load_report(refs[stem])
        candidate = load_report(cands[stem])
        results.append(score_run(reference, candidate, skip_judge=args.skip_judge))
    summary = aggregate_runs(results)
    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="src.metrics", description="Quality metric runner.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_score = sub.add_parser("score", help="Score one (reference, candidate) pair.")
    p_score.add_argument("--reference", required=True, help="Path to reference JSON.")
    p_score.add_argument("--candidate", required=True, help="Path to candidate JSON (runs/<id>.json or AnalysisReport).")
    p_score.add_argument("--out", help="Where to write the quality JSON. Stdout if omitted.")
    p_score.add_argument("--markdown", help="Optional path for human-readable Markdown report.")
    p_score.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge (offline mode, useful for CI).")
    p_score.set_defaults(func=_score_pair)

    p_agg = sub.add_parser("aggregate", help="Aggregate Quality across a directory of pairs.")
    p_agg.add_argument("--reference-dir", required=True)
    p_agg.add_argument("--candidate-dir", required=True)
    p_agg.add_argument("--out")
    p_agg.add_argument("--skip-judge", action="store_true")
    p_agg.set_defaults(func=_aggregate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
