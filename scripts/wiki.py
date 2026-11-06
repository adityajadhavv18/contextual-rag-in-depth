"""The one place that talks to the Wikipedia API: a polite client (named user agent, ~1 request a second, retries)."""

import time

import requests

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "contextual-rag-in-depth/0.1 (https://github.com/adityajadhavv18/contextual-rag-in-depth; research)"
PAUSE_S = 1.0

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last = 0.0


def api(**params) -> dict:
    global _last
    params |= {"format": "json", "formatversion": 2}
    for attempt in range(5):
        time.sleep(max(0.0, _last + PAUSE_S - time.time()))
        _last = time.time()
        r = _session.get(API, params=params, timeout=60)
        if r.status_code in (429, 503):  # throttled: back off and retry
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"Wikipedia API error: {data['error']}")
        return data
    raise RuntimeError(f"Wikipedia API kept throttling: {params}")
