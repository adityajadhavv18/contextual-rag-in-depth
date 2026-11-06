"""Search the cleaned corpus while writing golden questions.

    uv run scripts/find.py "lbw for 99"                 # every hit: doc, section path, the line around it
    uv run scripts/find.py --section <doc_id> "1st Test" # print one section (by heading text)

Hits are counted the way the harness matches evidence (whitespace collapsed, lowercase), over included articles
only, so "1 hit" means the phrase is safe to use as an evidence quote.
"""

import json
import re
import sys

from build_manifest import MANIFEST
from clean_wiki import TEXT


def included() -> dict[str, str]:
    return {e["doc_id"]: (TEXT / f"{e['doc_id']}.txt").read_text()
            for e in json.loads(MANIFEST.read_text()) if e["include"]}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def find(phrase: str, width: int = 160) -> None:
    p = norm(phrase)
    hits = 0
    for doc, text in included().items():
        path = {}
        for line in text.splitlines():
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                path = {k: v for k, v in path.items() if k < level} | {level: line.lstrip("# ")}
                continue
            n = norm(line)
            for m in re.finditer(re.escape(p), n):
                hits += 1
                lo = max(0, m.start() - width)
                print(f"[{doc}] {' › '.join(list(path.values())[1:])}\n    …{n[lo:m.end() + width]}…")
    print(f"{hits} hit(s)")


def section(doc: str, heading: str) -> None:
    lines, out, depth = included()[doc].splitlines(), False, 0
    for line in lines:
        if line.startswith("#"):
            d = len(line) - len(line.lstrip("#"))
            if out and d <= depth:
                break
            if line.lstrip("# ").strip() == heading:
                out, depth = True, d
        if out:
            print(line)


if __name__ == "__main__":
    if sys.argv[1] == "--section":
        section(sys.argv[2], sys.argv[3])
    else:
        find(" ".join(sys.argv[1:]))
