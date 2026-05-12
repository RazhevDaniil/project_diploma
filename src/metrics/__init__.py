"""Quality metric for ТЗ-compliance service.

Compares service output against an Opus-generated reference and produces
a single Quality score in [0..100] plus component breakdown.

См. Метрика_качества_спека.md в корне проекта.
"""

from src.metrics.score import (
    QualityResult,
    score_run,
    aggregate_runs,
)

__all__ = ["QualityResult", "score_run", "aggregate_runs"]
