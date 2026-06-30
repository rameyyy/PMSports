"""
Shared HTTP layer for UFC scrapers.
Tapology URLs are routed through FlareSolverr (Cloudflare bypass).
All other URLs (UFCStats, etc.) use cloudscraper directly.
"""
import time
import requests
import cloudscraper
from requests.exceptions import ConnectionError as ReqConnectionError

FLARESOLVERR_URL = "http://localhost:8191/v1"
_scraper = cloudscraper.create_scraper()


class _FlareSolverrResponse:
    """Thin wrapper so FlareSolverr responses look like requests.Response."""
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text


def _get_via_flaresolverr(url: str, max_timeout_ms: int = 60000) -> _FlareSolverrResponse:
    resp = requests.post(FLARESOLVERR_URL, json={
        "cmd": "request.get",
        "url": url,
        "maxTimeout": max_timeout_ms,
    }, timeout=max_timeout_ms / 1000 + 10)
    data = resp.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"FlareSolverr error: {data}")
    solution = data["solution"]
    return _FlareSolverrResponse(solution["status"], solution["response"])


_FLARESOLVERR_HOSTS = ("tapology.com", "ufcstats.com")


def _get(url: str, max_retries: int = 4, base_wait: int = 20, timeout: int = 30):
    """
    GET with exponential backoff. Cloudflare/JS-challenged sites (Tapology,
    UFCStats) route through FlareSolverr; everything else uses cloudscraper.
    UFCStats added a JS bot-challenge in mid-2026 that cloudscraper can't pass,
    which silently broke outcome-settling — hence routing it through FlareSolverr.
    """
    use_flaresolverr = any(h in url for h in _FLARESOLVERR_HOSTS)

    for attempt in range(max_retries):
        try:
            if use_flaresolverr:
                resp = _get_via_flaresolverr(url, max_timeout_ms=60000)
            else:
                resp = _scraper.get(url, timeout=timeout)

            if resp.status_code == 429:
                wait = base_wait * (2 ** attempt)
                print(f"  Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            return resp
        except (ReqConnectionError, Exception) as e:
            if attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt)
                print(f"  Connection error ({e.__class__.__name__}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    # Final attempt
    if use_flaresolverr:
        return _get_via_flaresolverr(url, max_timeout_ms=60000)
    return _scraper.get(url, timeout=timeout)
