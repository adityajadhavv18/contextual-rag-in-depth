"""Cricsheet ball-by-ball data (open, cricsheet.org) -> data/cricket/cricsheet/ (git-ignored) + a match index.

    uv run scripts/download_cricsheet.py           # download the zips if missing, rebuild the index
    uv run scripts/download_cricsheet.py --fresh   # download again (Cricsheet adds new matches)

Not part of the corpus: it is the second source that golden answers are checked against. The index has one entry
per match: date, format, teams, venue, event and match number, result, player of the match, and per innings the
total plus every batter's runs (balls) and every bowler's wickets/runs, computed from the deliveries the usual way:
  - a batter faces every ball except wides; a bowler is charged everything except byes and leg byes
  - a bowler gets the wicket unless it is a run out, retired, or obstructing the field
  - overs count legal balls only (no wides, no no-balls)
The JSON is read straight from the zips, never unpacked to disk.
"""

import io
import json
import sys
import zipfile
from collections import defaultdict

import requests

from build_manifest import MANIFEST
from wiki import USER_AGENT

DIR = MANIFEST.parent / "cricsheet"
INDEX = DIR / "index.json"
ZIPS = ["tests", "odis", "t20s", "ipl"]
NOT_BOWLERS = {"run out", "retired hurt", "retired out", "retired not out", "obstructing the field"}


def download() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    for name in ZIPS:
        path = DIR / f"{name}_json.zip"
        if path.exists() and "--fresh" not in sys.argv:
            continue
        r = requests.get(f"https://cricsheet.org/downloads/{name}_json.zip", timeout=300,
                         headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        path.write_bytes(r.content)
        print(f"  {path.name}: {len(r.content) / 1e6:.1f} MB")


def innings_summary(inn: dict) -> dict:
    runs = wkts = balls = 0
    bat = defaultdict(lambda: [0, 0])  # runs, balls
    bowl = defaultdict(lambda: [0, 0, 0])  # wickets, runs, legal balls
    out = set()
    for over in inn.get("overs", []):
        for d in over["deliveries"]:
            ex = d.get("extras", {})
            legal = "wides" not in ex and "noballs" not in ex
            runs += d["runs"]["total"]
            balls += legal
            bat[d["batter"]][0] += d["runs"]["batter"]
            bat[d["batter"]][1] += "wides" not in ex
            bowl[d["bowler"]][1] += d["runs"]["total"] - ex.get("byes", 0) - ex.get("legbyes", 0)
            bowl[d["bowler"]][2] += legal
            for w in d.get("wickets", []):
                wkts += w["kind"] not in ("retired hurt", "retired not out")
                out.add(w["player_out"])
                if w["kind"] not in NOT_BOWLERS:
                    bowl[d["bowler"]][0] += 1
    return {"team": inn["team"], "runs": runs, "wickets": wkts, "overs": f"{balls // 6}.{balls % 6}",
            "declared": bool(inn.get("declared")), "super_over": bool(inn.get("super_over")),
            "batting": {p: f"{r}{'' if p in out else '*'} ({b})" for p, (r, b) in bat.items()},
            "bowling": {p: f"{w}/{r} ({lb // 6}.{lb % 6} ov)" for p, (w, r, lb) in bowl.items()}}


def match_entry(match_id: str, m: dict) -> dict:
    info = m["info"]
    o = info.get("outcome", {})
    by = o.get("by", {})
    result = (f"{o['winner']} won by " + ", ".join(f"{v} {k}" for k, v in by.items()) if "winner" in o
              else o.get("result", "unknown"))
    return {"id": match_id, "format": info.get("match_type"), "dates": info.get("dates"),
            "teams": info.get("teams"), "venue": info.get("venue"), "city": info.get("city"),
            "event": info.get("event", {}).get("name"), "match_number": info.get("event", {}).get("match_number"),
            "stage": info.get("event", {}).get("stage"), "result": result,
            "toss": info.get("toss"), "player_of_match": info.get("player_of_match"),
            "innings": [innings_summary(i) for i in m.get("innings", [])]}


def main() -> None:
    download()
    index = []
    for name in ZIPS:
        with zipfile.ZipFile(DIR / f"{name}_json.zip") as z:
            for f in z.namelist():
                if f.endswith(".json"):
                    index.append(match_entry(f"{name}/{f[:-5]}", json.load(io.BytesIO(z.read(f)))))
    INDEX.write_text(json.dumps(index, ensure_ascii=False))
    by_fmt = defaultdict(int)
    for e in index:
        by_fmt[e["id"].split("/")[0]] += 1
    first = min(e["dates"][0] for e in index)
    print(f"{len(index):,} matches indexed {dict(by_fmt)}, from {first} -> {INDEX.relative_to(MANIFEST.parent.parent.parent)}")


if __name__ == "__main__":
    main()
