"""Persistent analysis run storage."""

from __future__ import annotations

from datetime import datetime
import json
import threading
from uuid import uuid4

import config as cfg

_lock = threading.Lock()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _path(run_id: str):
    return cfg.RUNS_DIR / f"{run_id}.json"


def _write(run: dict) -> dict:
    cfg.RUNS_DIR.mkdir(exist_ok=True)
    run["updated_at"] = _now()
    target = _path(run["id"])
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return run


def create_run(document_name: str, files: list[dict] | None = None, settings: dict | None = None) -> dict:
    run = {
        "id": uuid4().hex,
        "document_name": document_name or "document",
        "status": "created",
        "stage": "created",
        "progress_done": 0,
        "progress_total": 0,
        "error": "",
        "created_at": _now(),
        "updated_at": _now(),
        "files": files or [],
        "parsed_files": [],
        "requirements": [],
        "report": None,
        "settings": settings or {},
    }
    with _lock:
        return _write(run)


def get_run(run_id: str) -> dict | None:
    path = _path(run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def require_run(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise KeyError(f"Run not found: {run_id}")
    return run


def update_run(run_id: str, **changes) -> dict:
    with _lock:
        run = require_run(run_id)
        run.update(changes)
        return _write(run)


def list_runs(limit: int = 50) -> list[dict]:
    cfg.RUNS_DIR.mkdir(exist_ok=True)
    runs = []
    for path in sorted(cfg.RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        runs.append(
            {
                "id": run.get("id"),
                "document_name": run.get("document_name", "document"),
                "status": run.get("status", "unknown"),
                "stage": run.get("stage", ""),
                "created_at": run.get("created_at", ""),
                "updated_at": run.get("updated_at", ""),
                "progress_done": run.get("progress_done", 0),
                "progress_total": run.get("progress_total", 0),
                "total_requirements": len(run.get("requirements", [])),
                "has_report": bool(run.get("report")),
                "error": run.get("error", ""),
            }
        )
        if len(runs) >= limit:
            break
    return runs
