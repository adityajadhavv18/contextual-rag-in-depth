"""Fetch every manifest article at its pinned revision into data/cricket/raw/<doc_id>.html (git-ignored).

    uv run scripts/download_wiki.py

Uses the parse API on the exact revid, so the HTML is the same edition every time. Already-downloaded pages
are skipped, so adding articles to the manifest only fetches the new ones. Every candidate is fetched, included
or not: measure.py needs their sizes to choose what fits the chunk budget.
"""

import json
from pathlib import Path

from build_manifest import MANIFEST
from wiki import api

RAW = MANIFEST.parent / "raw"


def main() -> None:
    RAW.mkdir(exist_ok=True)
    entries = json.loads(MANIFEST.read_text())
    todo = [e for e in entries if not (RAW / f"{e['doc_id']}.html").exists()]
    for i, e in enumerate(todo, start=1):
        data = api(action="parse", oldid=e["revid"], prop="text", disableeditsection=1, disabletoc=1)
        (RAW / f"{e['doc_id']}.html").write_text(data["parse"]["text"])
        print(f"  {i}/{len(todo)} {e['title']}")
    print(f"{len(entries)} articles in {RAW.relative_to(MANIFEST.parent.parent.parent)}/ ({len(todo)} new)")


if __name__ == "__main__":
    main()
