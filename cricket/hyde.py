"""HyDE: search with a fake answer instead of the question (Gao et al., 2022, "Precise Zero-Shot Dense Retrieval
without Relevance Labels"). Carried over from Hybrid #1, where it was the best rung (r6a).

A question ("how was Kohli out in the Adelaide Test in December 2020?") and the passage that answers it ("Rahane
drove the ball to mid-off and called a run, but sent Kohli back...") are written very differently. HyDE asks the LLM
to WRITE a short passage in the corpus's own style that answers the question, then embeds that. The facts may be
wrong; only its shape and vocabulary matter. The one change from Hybrid #1 is the prompt: there it asked for a fault
manual entry, here for a Wikipedia match report or biography.

Only DENSE search uses it. BM25 / SPLADE keep the original question, so the names, years and grounds the user typed
are never replaced by whatever the LLM guessed.

Fake passages are cached on disk per question (data/cricket/cache/hyde_<model>.json) with the tokens and time they
took, so every HyDE rung reads the SAME passage (a fair comparison) and re-runs cost nothing; the ladder still
counts the original LLM time and tokens as part of each question.
"""

import json
import time
from functools import cache

from eval_harness import settings, traced_openai

from ingest import CACHE
from retrieve import _dense, timed

PROMPT = """You write Wikipedia articles about cricket: series and match reports, tournaments, players and grounds.
Write the passage that answers the question below, the way a Wikipedia article would print it: a match report
(day by day, with scores, wickets and partnerships), a scorecard summary, or a paragraph of a biography.
About 100 words. No introduction, no disclaimers.

Question: {question}"""
CACHE_FILE = CACHE / f"hyde_{settings.llm_model}.json"
USAGE = {"tokens_in": 0, "tokens_out": 0, "llm_ms": 0.0}  # this question's LLM use; the ladder resets it per question
# llm_ms counts only passages served from the cache: a live call is already inside the harness's own timing


@cache
def _client():
    return traced_openai()


@cache
def _cache() -> dict:
    return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}


def fake_passage(question: str) -> str:
    """The LLM's made-up passage for this question (cached), adding its tokens and time to USAGE."""
    store = _cache()
    cached = question in store
    if not cached:
        start = time.perf_counter()
        resp = _client().chat.completions.create(
            model=settings.llm_model, temperature=0, max_tokens=250,
            messages=[{"role": "user", "content": PROMPT.format(question=question)}])
        store[question] = {"text": resp.choices[0].message.content.strip(),
                           "tokens_in": resp.usage.prompt_tokens, "tokens_out": resp.usage.completion_tokens,
                           "ms": (time.perf_counter() - start) * 1000}
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(store, indent=1))
    hit = store[question]
    USAGE["tokens_in"] += hit["tokens_in"]
    USAGE["tokens_out"] += hit["tokens_out"]
    USAGE["llm_ms"] += hit["ms"] if cached else 0.0
    return hit["text"]


def hyde_vector(question: str) -> list[float]:
    """The dense query vector: the fake passage alone, embedded as a DOCUMENT (no query instruction)."""
    passage = fake_passage(question)
    with timed("hyde_embed"):
        return next(_dense().passage_embed([passage])).tolist()
