"""r3: an LLM note per chunk that says what happens IN it (Anthropic's Contextual Retrieval, 2024, with one twist).

The free label (r2b) already says WHERE a chunk sits: article, heading path, match. Every chunk of a match shares it,
so it can't tell Day One from Day Four. The note's only job is what tells this chunk apart from its siblings: which
innings, who did what, the score. Hybrid #1's notes failed by restating the label ("section detailing alarms..."),
which made neighbours look MORE alike, so the prompt shows the LLM the label and forbids repeating it.

Two modes decide what the LLM reads besides the chunk (the whole article was dropped: ~$3.6+ for all chunks):
  chunk     the free label only                         ~$0.6 for all chunks
  section   the free label plus the chunk's whole section ~$1.8 (can resolve "he" / "the visitors" across chunks)

Notes are cached per prompt version and mode in data/cricket/cache/notes_<version>_<mode>_<model>.json, so each is
paid for once.

Run from contextual_rag_in_depth/ (Qdrant running, OpenAI key sourced):
    uv run cricket/notes.py --pilot      # both modes on the pilot chunks, examples + similarity check
    uv run cricket/notes.py              # section-mode notes for every chunk (~$0.95 once)
"""

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache

import numpy as np
from eval_harness import load_golden, settings, traced_openai

from chunks import TEXT_DIR, Chunk, load_chunks, sections
from ingest import CACHE

PROMPT = """<label>
{label}
</label>
{section}<chunk>
{chunk}
</chunk>
The search index already labels this chunk with the label above: the article title, its heading path and the match's
date, teams and venue. Write a note (1-2 sentences, under 50 words) that tells THIS chunk apart from the other chunks
about the same match or the same person:
- Start with the innings or stage itself ("India's first innings ends...", "Australia's chase...", "The last over...",
  "Early career..."). Never start with "This chunk" or "This section".
- Cover the chunk from its FIRST sentence to its last. The opening events matter most: the chunk before only partly
  covers them.
- Name the innings only when the text makes it clear; if the chunk spans two innings, name both. Add the day if the
  heading doesn't say.
- The players and events in it by full name: runs, wickets, how batters were out, partnerships, the score where it
  starts and ends.
- Replace "he", "the visitors", "the hosts" with the names they stand for.
- Add the cricket word a reader would search for: century, five-wicket haul, duck, hat-trick.
- For a scorecard or table: whose batting or bowling, and which innings.
- Don't repeat the title, date, venue or teams. State the facts.
Answer only with the note."""
PROMPT_VERSION = "v2"  # v1 skipped a chunk's opening events and began most notes "This chunk covers..."
SECTION = "<section>\n{body}\n</section>\n"
MODES = ("chunk", "section")
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.15, 0.075, 0.60  # gpt-4o-mini, $ per million tokens


def label(c: Chunk) -> str:
    return "\n".join([" › ".join(c.path), *c.match])


def with_note(c: Chunk, note: str) -> str:
    """The r3 index text: r2b's free label, the note, then the chunk."""
    return "\n".join([label(c), note, c.text])


@cache
def section_body(chunk_id: str) -> str:
    doc, s, _ = chunk_id.split("::")
    return sections((TEXT_DIR / f"{doc}.txt").read_text())[int(s[1:])][1]


def cache_file(mode: str):
    return CACHE / f"notes_{PROMPT_VERSION}_{mode}_{settings.llm_model}.json"


def load_notes(mode: str) -> dict:
    path = cache_file(mode)
    return json.loads(path.read_text()) if path.exists() else {}


def add_notes(mode: str, chunks: list[Chunk], workers: int = 8) -> dict:
    """Write a note for every chunk that has none yet, `workers` calls at a time; saves every 50 notes and at the end,
    so a crash loses at most 50."""
    notes, client = load_notes(mode), traced_openai()
    usage = {"in": 0, "cached": 0, "out": 0, "calls": 0}
    # same id but different text means the chunking changed: the old note describes another chunk
    todo = [c for c in chunks if notes.get(c.chunk_id, {}).get("chunk_start") != c.text[:80]]

    def write(c: Chunk):
        section = SECTION.format(body=section_body(c.chunk_id)) if mode == "section" else ""
        resp = client.chat.completions.create(
            model=settings.llm_model, temperature=0, max_tokens=100,  # hard stop if it ignores "under 50 words"
            messages=[{"role": "user", "content": PROMPT.format(label=label(c), section=section, chunk=c.text)}])
        return c, resp

    save = lambda: cache_file(mode).write_text(json.dumps(notes, indent=1, ensure_ascii=False))
    start = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        for c, resp in pool.map(write, todo):
            u = resp.usage
            cached = u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else 0
            notes[c.chunk_id] = {"note": resp.choices[0].message.content.strip(), "chunk_start": c.text[:80]}
            for key, n in (("in", u.prompt_tokens), ("cached", cached), ("out", u.completion_tokens), ("calls", 1)):
                usage[key] += n
            if usage["calls"] % 50 == 0:
                save()
                if usage["calls"] % 500 == 0:
                    print(f"     {mode}: {usage['calls']}/{len(todo)} ({time.perf_counter() - start:.0f}s, "
                          f"${cost(usage):.3f})", flush=True)
    save()
    return usage


def cost(usage: dict) -> float:
    fresh = usage["in"] - usage["cached"]
    return (fresh * PRICE_IN + usage["cached"] * PRICE_CACHED + usage["out"] * PRICE_OUT) / 1e6


def pilot_chunks() -> list[tuple[str, Chunk, list[Chunk]]]:
    """Per tune question: the chunk holding its evidence, and up to 3 chunks r2b ranks above it (its rivals)."""
    from ingest import collection
    from ladder import CANDIDATES, TUNE, fragments, hybrid_stack, norm
    by_id = {c.chunk_id: c for c in load_chunks()}
    picks = []
    for g in load_golden(TUNE):
        if g.question_type == "no_answer":
            continue
        frags = fragments(g)
        answer = next(c for c in by_id.values() if c.doc_id == g.source_doc and any(f in norm(c.text) for f in frags))
        ranked = [h.chunk_id for h in hybrid_stack(g.question, CANDIDATES, collection("header_match"))]
        above = ranked[:ranked.index(answer.chunk_id)] if answer.chunk_id in ranked else ranked
        picks.append((g.id, answer, [by_id[i] for i in above[:3]]))
    return picks


def pilot():
    from retrieve import _dense
    picks = pilot_chunks()
    todo = list({c.chunk_id: c for _, a, rivals in picks for c in [a, *rivals]}.values())
    print(f"1. pilot: {len(picks)} questions -> {len(todo)} chunks (answer chunks + the rivals r2b ranks above them)")
    for mode in MODES:
        u = add_notes(mode, todo)
        print(f"   {mode}: {u['calls']} new notes, {u['in']:,} in ({u['cached']:,} cached) / {u['out']:,} out "
              f"tokens, ${cost(u):.4f}  -> all 4,650 chunks ~${cost(u) / max(u['calls'], 1) * 4650:.2f}")
    notes = {m: load_notes(m) for m in MODES}

    # Does the note pull the answer chunk AWAY from the rivals that beat it? Cosine of answer vs each rival, as the
    # dense search sees them, before (label + chunk) and after (label + note + chunk). Lower = easier to tell apart.
    embed = lambda ts: np.array(list(_dense().passage_embed(ts)))
    sims = {"r2b label": [], **{m: [] for m in MODES}}
    for _, a, rivals in picks:
        for r in rivals:
            pair = {"r2b label": [f"{label(a)}\n{a.text}", f"{label(r)}\n{r.text}"],
                    **{m: [with_note(a, notes[m][a.chunk_id]["note"]), with_note(r, notes[m][r.chunk_id]["note"])]
                       for m in MODES}}
            for k, ts in pair.items():
                v = embed(ts)
                sims[k].append(float(v[0] @ v[1]))
    print(f"\n2. answer vs rival similarity ({len(sims['r2b label'])} pairs, lower = easier to tell apart):")
    for k, s in sims.items():
        print(f"   {k:10s} mean {statistics.mean(s):.3f}")

    print("\n3. examples")
    for qid, a, rivals in picks:
        if qid in ("twin-04", "which-03", "twin-03", "pron-02", "num-01", "pron-01"):
            print(f"\n[{qid}] {a.chunk_id}   ({len(rivals)} rivals above it in r2b)")
            print(f"   CHUNK:   {a.text[:160]} …".replace("\n", " / "))
            for m in MODES:
                print(f"   {m:8s} {notes[m][a.chunk_id]['note']}")
            if rivals:
                r = rivals[0]
                print(f"   rival #1 {' › '.join(r.path[1:])}: {notes['chunk'][r.chunk_id]['note']}")
    words = {m: statistics.mean(len(notes[m][c.chunk_id]["note"].split()) for c in todo) for m in MODES}
    print(f"\n4. note length, mean words: " + ", ".join(f"{m} {w:.0f}" for m, w in words.items()))


if __name__ == "__main__":
    if "--pilot" in sys.argv:
        pilot()
    else:  # every chunk, section mode: the pilot's pick (fewer wrong facts than chunk mode for ~$0.35 more)
        u = add_notes("section", load_chunks())
        print(f"{u['calls']} new notes, {u['in']:,} in ({u['cached']:,} cached) / {u['out']:,} out tokens, "
              f"${cost(u):.3f}; {len(load_notes('section'))} in {cache_file('section').name}")
