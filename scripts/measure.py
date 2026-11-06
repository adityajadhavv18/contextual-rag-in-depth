"""Size the corpus in chunks and cut it to the chunk budget: sets `include` in manifest.json.

    uv run scripts/measure.py            # print sizes by tier and the proposed cut, change nothing
    uv run scripts/measure.py --apply    # write the cut into the manifest

Chunks are estimated the way the baseline chunker will cut them: 300-token windows with 50 overlap (bge tokens),
inside one section at a time (a chunk never runs from one section into the next). So a section of n tokens makes
max(1, ceil((n - 50) / 250)) chunks.

The cut keeps every tier (the corpus is meant to be wide) and trims only player biographies, which are long and
the most optional: a biography earns its place by how often its player is named in the match and tournament
articles, so the least-named players go first. (Cutting the biggest first would drop Smith and Anderson.)
"""

import json
import math
import re
import sys
from collections import defaultdict
from functools import cache

from tokenizers import Tokenizer

from build_manifest import MANIFEST
from clean_wiki import ARTICLES, TEXT

BUDGET = 5000
CHUNK, OVERLAP = 300, 50


@cache
def tokenizer() -> Tokenizer:
    tok = Tokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    tok.no_truncation()
    return tok


def sections(text: str) -> list[str]:
    out, cur = [], []
    for line in text.splitlines():
        if line.startswith("#"):
            out.append("\n".join(cur))
            cur = []
        cur.append(line)
    return [s for s in out + ["\n".join(cur)] if s.strip()]


def size(doc_id: str) -> tuple[int, int]:
    """(tokens, estimated chunks) of one article."""
    lens = [len(t.ids) for t in tokenizer().encode_batch(sections((TEXT / f"{doc_id}.txt").read_text()),
                                                        add_special_tokens=False)]
    return sum(lens), sum(max(1, math.ceil((n - OVERLAP) / (CHUNK - OVERLAP))) for n in lens)


def mentions(entries: list[dict]) -> dict[str, int]:
    """Player doc_id -> times the player's name appears in the non-biography articles."""
    others = " ".join((TEXT / f"{e['doc_id']}.txt").read_text() for e in entries if e["tier"] != "players")
    return {e["doc_id"]: others.count(re.sub(r" \(.*\)$", "", e["title"]))
            for e in entries if e["tier"] == "players"}


def main() -> None:
    entries = json.loads(MANIFEST.read_text())
    sizes = {e["doc_id"]: size(e["doc_id"]) for e in entries}
    total = sum(c for _, c in sizes.values())
    include = {e["doc_id"]: True for e in entries}
    named = mentions(entries)
    for d in sorted(named, key=named.get):  # least-named player first, until the corpus fits
        if total <= BUDGET:
            break
        include[d] = False
        total -= sizes[d][1]

    by_tier = defaultdict(lambda: [0, 0, 0, 0])  # articles, tokens, chunks, articles kept
    for e in entries:
        t = by_tier[e["tier"]]
        tok, ch = sizes[e["doc_id"]]
        t[0] += 1
        t[1] += tok
        t[2] += ch * include[e["doc_id"]]
        t[3] += include[e["doc_id"]]
    print(f"{'tier':18} {'articles':>9} {'tokens':>10} {'chunks kept':>12}")
    for tier, (n, tok, ch, kept) in by_tier.items():
        print(f"{tier:18} {f'{kept}/{n}':>9} {tok:>10,} {ch:>12,}")
    print(f"{'total':18} {sum(include.values()):>5}/{len(entries)} {sum(t for t, _ in sizes.values()):>10,} "
          f"{total:>12,}  (budget {BUDGET:,})")
    print("left out:", ", ".join(f"{e['title']} ({named[e['doc_id']]} mentions)"
                                 for e in entries if not include[e["doc_id"]]) or "none")

    if "--apply" in sys.argv:
        for e in entries:
            e["include"] = include[e["doc_id"]]
        MANIFEST.write_text(json.dumps(entries, indent=1, ensure_ascii=False) + "\n")
        meta = json.loads(ARTICLES.read_text())
        for m in meta:
            m["tokens"], m["est_chunks"] = sizes[m["doc_id"]]
            m["include"] = include[m["doc_id"]]
        ARTICLES.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
        print("manifest.json and articles.json updated")


if __name__ == "__main__":
    main()
