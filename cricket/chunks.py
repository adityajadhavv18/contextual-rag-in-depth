"""Section-aware fixed-size chunks: the baseline every rung searches over. Only the index text changes later.

Each included article (data/cricket/text/<doc_id>.txt, headings as #-lines) is split at every heading, and each
section's body is cut into windows of CHUNK_TOKENS with OVERLAP_TOKENS shared between neighbours. A chunk never
runs from one section into the next, so it belongs to exactly one place in the article ("1st Test › Day Three").

The chunk TEXT is the section body only: no title, no heading. That's what the LLM reads and what golden evidence is
matched against, on every rung. Two labels are kept beside it, unused by r0/r1, so the free header rungs (r2) can
add them to the INDEX text without changing a single chunk:
  Chunk.path    the article title and every heading above the section
  Chunk.match   the match box(es) the chunk belongs to: "17–21 December 2020 (D/N) | India v Australia | Adelaide
                Oval, Adelaide". That's the box in effect where the chunk starts (the last one earlier in its
                section, or in a parent section: a "Day One" section's box sits under "1st Test"), plus any box that
                begins inside the chunk, at most two. A box that prints no year gets it from the article title.
                In a single-match article (one box in all: the finals) the box labels every chunk: finals print it
                under "Match › Details", but tell the story in the lead and under "Match › Summary › Day 1".
                IPL final articles print no box at all; their title names the match.

Why 300 / 50 (same as Hybrid #1): bge-small reads at most 512 tokens, and 300 leaves room for a ~40-token header and
a ~100-token note in later rungs. The overlap of 50 is longer than the longest golden evidence quote, so every
quote lands whole inside at least one window.
"""

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from tokenizers import Tokenizer

DATA = Path(__file__).resolve().parent.parent / "data" / "cricket"
TEXT_DIR, MANIFEST = DATA / "text", DATA / "manifest.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS = 300  # measured in bge tokens
OVERLAP_TOKENS = 50


@dataclass
class Chunk:
    chunk_id: str  # "indian_cricket_team_in_australia_in_2020-21::s014::01" (section 14, window 1)
    doc_id: str  # becomes context_ids in PipelineOutput
    path: tuple[str, ...]  # article title, then each heading above the section: ("...2020–21", "1st Test", "Day Three")
    text: str
    n_tokens: int
    match: tuple[str, ...] = ()  # the match box facts this chunk belongs to (r2b label)


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
BOX = re.compile(r"^Match: (.*?) \| (.*)$\n(?:.*\n){0,10}?^Venue: (.*)$", re.M)  # Match: line + its Venue: line


def _with_year(date: str, title: str) -> str:
    """'10–14 July' + '2013 Ashes series' -> '10–14 July 2013'. A "2013–14" season's Jan-June matches are 2014."""
    if re.search(r"\b\d{4}\b", date):
        return date
    season = re.search(r"\b(\d{4})(–\d{2})?\b", title)
    month = next((i for i, m in enumerate(MONTHS, start=1) if m in date), 0)
    if not season or not month:
        return date
    year = int(season.group(1)) + (bool(season.group(2)) and month <= 6)
    head, _, tail = date.partition(";")  # "18 April; 16:00" -> keep the time after the year
    return f"{head.strip()} {year}" + (f";{tail}" if tail else "")


def boxes(body: str, title: str) -> list[tuple[int, str]]:
    """(char position, label) for every match box in a section body."""
    return [(m.start(), f"{_with_year(m.group(1), title)} | {m.group(2)} | {m.group(3)}") for m in BOX.finditer(body)]


@cache
def tokenizer() -> Tokenizer:
    tok = Tokenizer.from_pretrained(EMBED_MODEL)
    tok.no_truncation()  # every token of a section, not just the first 512
    return tok


def sections(text: str) -> list[tuple[tuple[str, ...], str]]:
    """(heading path, body) per section, in order; sections with no body are skipped."""
    out, path, body = [], {}, []

    def close():
        if "".join(body).strip():
            out.append((tuple(path[k] for k in sorted(path)), "\n".join(body)))
        body.clear()

    for line in text.splitlines():
        if line.startswith("#"):
            close()
            level = len(line) - len(line.lstrip("#"))
            path = {k: v for k, v in path.items() if k < level} | {level: line.lstrip("# ").strip()}
        else:
            body.append(line)
    close()
    return out


def chunk_section(doc_id: str, n: int, path: tuple[str, ...], body: str,
                  section_boxes: list[tuple[int, str]] = (), inherited: str | None = None) -> list[Chunk]:
    """inherited: the match box in effect when this section starts (from a parent section), if any."""
    offsets = tokenizer().encode(body, add_special_tokens=False).offsets  # (start_char, end_char) per token
    chunks, step = [], CHUNK_TOKENS - OVERLAP_TOKENS
    for i, start in enumerate(range(0, len(offsets), step)):
        end = min(start + CHUNK_TOKENS, len(offsets))
        lo, hi = offsets[start][0], offsets[end - 1][1]
        before = [label for pos, label in section_boxes if pos <= lo]
        inside = [label for pos, label in section_boxes if lo < pos < hi]
        match = tuple(dict.fromkeys(([before[-1]] if before else [inherited] if inherited else []) + inside))[:2]
        # slice the ORIGINAL text by character offsets: bge's tokenizer lowercases, so never decode its tokens
        chunks.append(Chunk(f"{doc_id}::s{n:03d}::{i:02d}", doc_id, path, body[lo:hi], end - start, match))
        if end == len(offsets):
            break
    return chunks


def included_docs() -> list[str]:
    return [e["doc_id"] for e in json.loads(MANIFEST.read_text()) if e["include"]]


@cache
def load_chunks() -> list[Chunk]:
    chunks = []
    for doc in included_docs():
        secs = sections((TEXT_DIR / f"{doc}.txt").read_text())
        all_boxes = [label for p, b in secs for _, label in boxes(b, p[0])]
        last = None  # (path of the section it was in, label) of the latest match box in this article
        for n, (path, body) in enumerate(secs):
            if len(all_boxes) == 1 and "final" in path[0].lower():  # a final's article is about that one match
                chunks.extend(chunk_section(doc, n, path, body, inherited=all_boxes[0]))
                continue
            found = boxes(body, path[0])
            # a box carries into later sections only while they sit under the section it was printed in
            inherited = last[1] if last and path[:len(last[0])] == last[0] else None
            chunks.extend(chunk_section(doc, n, path, body, found, inherited))
            if found:
                last = (path, found[-1][1])
            elif inherited is None:
                last = None
    return chunks


if __name__ == "__main__":
    import statistics
    cs = load_chunks()
    sizes = [c.n_tokens for c in cs]
    print(f"{len(cs)} chunks from {len({c.doc_id for c in cs})} articles; tokens median {statistics.median(sizes):.0f}, "
          f"under 50: {sum(s < 50 for s in sizes)}, full 300: {sum(s == CHUNK_TOKENS for s in sizes)}")
    for c in cs[1000:1002]:
        print(f"\n{c.chunk_id}  path={' › '.join(c.path)}\n{c.text[:300]}")
