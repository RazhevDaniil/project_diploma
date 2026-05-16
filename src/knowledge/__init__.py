"""Knowledge subpackage: curated facts about Cloud.ru capabilities.

This module provides a hand-curated catalogue of confirmed Cloud.ru
capabilities with URLs to the official documentation. It serves as a
fallback context source for the LLM analyzer when Managed RAG does not
return relevant chunks — preventing the model from "guessing" answers
based on its training data alone.

The catalogue is editable by tech-sales (no code changes needed) — just
add an entry with keywords, platform, statement, and URL.
"""

from src.knowledge.curated_facts import (
    CuratedFact,
    find_relevant_facts,
    format_facts_for_prompt,
    find_enforce_verdict,
)

__all__ = [
    "CuratedFact",
    "find_relevant_facts",
    "format_facts_for_prompt",
    "find_enforce_verdict",
]
