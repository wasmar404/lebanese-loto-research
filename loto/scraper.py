"""Polite, resumable scraper for the LLDJ Loto history.

* Detects the latest completed draw from the no-parameter results page.
* Downloads every draw page 1..latest, caching raw HTML (gzip) so re-runs never
  re-download what is already on disk (resumable after interruption).
* Pulls the site's own draw index (the dropdown endpoint) as an independent list of
  draw numbers and dates for the completeness check.
* Single connection-pooled client, global rate limit, timeouts, exponential backoff.
* robots.txt is checked first; disallowed paths are never fetched.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .config import ROOT, load_config, setup_logging
from .parse import ParseError, parse_draw_page, parse_index_page, parse_latest_draw_number

log = setup_logging("loto.scraper")


class PoliteClient:
    def __init__(self, cfg: dict):
        s = cfg["scraper"]
        self.min_interval = float(s["min_interval_seconds"])
        self.max_retries = int(s["max_retries"])
        self.backoff = float(s["backoff_base_seconds"])
        self._last = 0.0
        self.n_requests = 0
        self.client = httpx.Client(
            headers={"User-Agent": s["user_agent"], "Accept-Language": "en"},
            timeout=httpx.Timeout(float(s["timeout_seconds"])),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=2),
            follow_redirects=True,
        )

    def _wait(self):
        dt = time.monotonic() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.monotonic()

    def request(self, method: str, url: str, *, retry_statuses=(429, 500, 502, 503, 504), **kw) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._wait()
            try:
                self.n_requests += 1
                r = self.client.request(method, url, **kw)
                if r.status_code in retry_statuses and attempt < self.max_retries:
                    delay = self.backoff * (2 ** attempt)
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        delay = max(delay, float(ra))
                    log.warning("HTTP %s for %s (attempt %d) - retrying in %.1fs", r.status_code, url, attempt + 1, delay)
                    time.sleep(delay)
                    continue
                return r
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                delay = self.backoff * (2 ** attempt)
                log.warning("%s for %s (attempt %d) - retrying in %.1fs", type(exc).__name__, url, attempt + 1, delay)
                time.sleep(delay)
        raise RuntimeError(f"giving up on {url}: {last_exc}")

    def close(self):
        self.client.close()


def check_robots(cfg: dict) -> dict:
    base = cfg["scraper"]["base_url"].rstrip("/")
    rp = urllib.robotparser.RobotFileParser()
    try:
        r = httpx.get(base + "/robots.txt", timeout=20, headers={"User-Agent": cfg["scraper"]["user_agent"]})
        status = r.status_code
        if status == 200:
            rp.parse(r.text.splitlines())
            allowed = rp.can_fetch(cfg["scraper"]["user_agent"], cfg["scraper"]["draw_url"].format(draw=1))
        else:
            allowed = True  # no robots.txt => no published crawl restrictions
    except httpx.HTTPError as exc:
        status, allowed = f"error: {exc}", True
    info = {"robots_status": status, "draw_pages_allowed": allowed}
    log.info("robots.txt check: %s", info)
    return info


def raw_path(cache_dir: Path, draw: int) -> Path:
    return cache_dir / f"{draw:05d}.html.gz"


def scrape(max_draw: int | None = None, refresh_latest: int = 0) -> dict:
    """Download all draw pages into the raw cache. Returns a run summary.

    refresh_latest: re-download the N most recent draws even if cached.
    """
    cfg = load_config()
    cache = ROOT / cfg["scraper"]["raw_cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    robots = check_robots(cfg)
    if not robots["draw_pages_allowed"]:
        raise SystemExit("robots.txt disallows draw pages - not scraping.")

    client = PoliteClient(cfg)
    try:
        latest_html = client.request("GET", cfg["scraper"]["latest_url"]).text
        latest = parse_latest_draw_number(latest_html)
        if max_draw:
            latest = min(latest, max_draw)
        log.info("latest completed draw on site: %d", latest)

        # independent draw index from the dropdown endpoint
        index = fetch_index(client, cfg, latest)

        downloaded, cached, failed = 0, 0, []
        t0 = time.time()
        for draw in range(1, latest + 1):
            path = raw_path(cache, draw)
            if path.exists() and draw <= latest - refresh_latest:
                cached += 1
                continue
            url = cfg["scraper"]["draw_url"].format(draw=draw)
            try:
                r = client.request("GET", url)
                if r.status_code != 200:
                    failed.append({"draw": draw, "error": f"HTTP {r.status_code}"})
                    continue
                parse_draw_page(r.text, expected_draw=draw)  # verify before caching
                tmp = path.with_suffix(".tmp")
                with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                    fh.write(r.text)
                tmp.replace(path)
                downloaded += 1
            except (ParseError, RuntimeError) as exc:
                failed.append({"draw": draw, "error": str(exc)})
                log.error("draw %d failed: %s", draw, exc)
            if downloaded and downloaded % 100 == 0:
                rate = downloaded / (time.time() - t0)
                log.info("downloaded %d (cached %d, failed %d) - %.2f pages/s, at draw %d", downloaded, cached, len(failed), rate, draw)

        summary = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "latest_draw": latest,
            "downloaded": downloaded,
            "already_cached": cached,
            "failed": failed,
            "requests": client.n_requests,
            "elapsed_seconds": round(time.time() - t0, 1),
            "robots": robots,
            "index_draws": len(index),
        }
        (cache.parent / "scrape_summary.json").write_text(json.dumps(summary, indent=2))
        log.info("scrape finished: %s", {k: v for k, v in summary.items() if k != "failed"} | {"failed": len(failed)})
        return summary
    finally:
        client.close()


def fetch_index(client: PoliteClient, cfg: dict, latest: int) -> list[tuple[int, str]]:
    """Fetch the site's own draw list (dropdown), 20 entries per call. Cached to JSON."""
    out_path = ROOT / cfg["paths"]["data_dir"] / "site_draw_index.json"
    existing: dict[int, str] = {}
    if out_path.exists():
        existing = {int(k): v for k, v in json.loads(out_path.read_text()).items()}
        if existing and max(existing) >= latest and len(existing) >= latest * 0.98:
            log.info("site draw index already cached (%d entries)", len(existing))
            return sorted(existing.items())
    page = int(cfg["scraper"]["index_page_size"])
    offset, items = 0, {}
    while True:
        r = client.request("POST", cfg["scraper"]["index_url"], data={"offset": offset},
                           headers={"X-Requested-With": "XMLHttpRequest"})
        if r.status_code != 200:
            log.warning("index offset %d -> HTTP %s; stopping", offset, r.status_code)
            break
        payload = r.json()
        if payload.get("errorcode") != 0:
            break
        rows = parse_index_page(payload.get("value", ""))
        if not rows:
            break
        for d, dt in rows:
            items[d] = dt.isoformat()
        offset += page
        if offset > latest + 5 * page:
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({str(k): v for k, v in sorted(items.items())}, indent=0))
    log.info("site draw index: %d entries", len(items))
    return sorted(items.items())


def load_cached_records(cache_dir: Path | None = None):
    """Parse every cached raw page. Returns (records, errors)."""
    cfg = load_config()
    cache = cache_dir or ROOT / cfg["scraper"]["raw_cache_dir"]
    records, errors = [], []
    for path in sorted(cache.glob("*.html.gz")):
        draw = int(path.name.split(".")[0])
        html = gzip.open(path, "rt", encoding="utf-8").read()
        try:
            rec = parse_draw_page(html, expected_draw=draw)
            records.append((rec, hashlib.sha256(html.encode()).hexdigest()))
        except ParseError as exc:
            errors.append({"draw": draw, "error": str(exc)})
    return records, errors
