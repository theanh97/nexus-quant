"""
NEXUS Web Research Tool - Fetches and parses web content for research.
Pure stdlib: urllib, html.parser, json, pathlib.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.hashing import sha256_text


_CACHE_TTL_SECS = 24 * 60 * 60  # 24h
_DEFAULT_UA = "NEXUS-WebResearch/1.0"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cache_dir() -> Path:
    return _project_root() / "artifacts" / "brain" / "web_cache"


def _cache_path(url: str) -> Path:
    key = sha256_text(url)
    return _cache_dir() / f"{key}.json"


def _is_cache_fresh(path: Path, ttl_secs: int) -> bool:
    try:
        age = time.time() - float(path.stat().st_mtime)
        return age >= 0 and age <= float(ttl_secs)
    except Exception:
        return False


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True)
    tmp.replace(path)


_ROBOTS_CACHE: Dict[str, Tuple[float, Optional[urllib.robotparser.RobotFileParser]]] = {}
_ROBOTS_TTL_SECS = 6 * 60 * 60  # 6h
_ROBOTS_FAIL_TTL_SECS = 30 * 60  # 30m


def _robots_allows(url: str, user_agent: str = _DEFAULT_UA, timeout: int = 10) -> bool:
    """
    Best-effort robots.txt check:
    - If robots.txt can't be fetched/parsed, we allow.
    - Cache allow/deny decisions per origin for a few hours.
    """
    try:
        p = urllib.parse.urlparse(url)
    except Exception:
        return True

    if p.scheme not in {"http", "https"} or not p.netloc:
        return True

    origin = f"{p.scheme}://{p.netloc}"
    now = time.time()
    cached = _ROBOTS_CACHE.get(origin)
    if cached is not None:
        ts, rp = cached
        if rp is None:
            if now - ts <= _ROBOTS_FAIL_TTL_SECS:
                return True
        else:
            if now - ts <= _ROBOTS_TTL_SECS:
                try:
                    return bool(rp.can_fetch(user_agent, url))
                except Exception:
                    return True

    robots_url = origin + "/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)

    try:
        # urllib.robotparser doesn't expose a timeout, so we fetch ourselves.
        req = urllib.request.Request(robots_url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(200_000)
        text = data.decode("utf-8", errors="replace")
        rp.parse(text.splitlines())
    except Exception:
        _ROBOTS_CACHE[origin] = (now, None)
        return True

    _ROBOTS_CACHE[origin] = (now, rp)
    try:
        return bool(rp.can_fetch(user_agent, url))
    except Exception:
        return True


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = (tag or "").lower()
        if t in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if t in {"p", "div", "br", "hr", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = (tag or "").lower()
        if t in {"br", "hr"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = (tag or "").lower()
        if t in {"script", "style", "noscript"}:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if t in {"p", "div", "li", "tr"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if data:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts).replace("\xa0", " ")
        lines = [ln.strip() for ln in raw.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)


def _strip_html(html: str) -> str:
    p = _HTMLTextExtractor()
    try:
        p.feed(html)
        p.close()
    except Exception:
        # If parsing fails, fall back to a whitespace-collapsed string.
        return " ".join((html or "").split())
    return p.text()


def _decode_bytes(data: bytes, content_type: str) -> str:
    enc = "utf-8"
    ctype = (content_type or "").lower()
    if "charset=" in ctype:
        try:
            enc = ctype.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
        except Exception:
            enc = "utf-8"
    try:
        return data.decode(enc, errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")


def fetch_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Fetch a URL and return a dict: {url, status, content, error}.

    - Best-effort robots.txt compliance (User-agent: NEXUS-WebResearch/1.0)
    - Redirects are handled by urllib (returned `url` is final URL)
    - HTML is converted to plain text (tags stripped)
    - Cached under artifacts/brain/web_cache/<sha256(url)>.json with 24h TTL
    """
    url = (url or "").strip()
    if not url:
        return {"url": "", "status": None, "content": "", "error": "Empty url"}

    cache_file = _cache_path(url)
    if cache_file.exists() and _is_cache_fresh(cache_file, _CACHE_TTL_SECS):
        cached = _load_json(cache_file)
        if cached is not None:
            # Ensure required keys exist even if cache file is malformed/old.
            return {
                "url": str(cached.get("url", url)),
                "status": cached.get("status", None),
                "content": str(cached.get("content", "")),
                "error": str(cached.get("error", "")),
            }

    if not _robots_allows(url, timeout=timeout):
        out = {"url": url, "status": 403, "content": "", "error": "Disallowed by robots.txt"}
        try:
            _save_json(cache_file, out)
        except Exception:
            pass
        return out

    req = urllib.request.Request(url, headers={"User-Agent": _DEFAULT_UA})
    status: Optional[int] = None
    content = ""
    error = ""
    final_url = url

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", resp.getcode()))
            final_url = str(resp.geturl() or url)
            ctype = str(resp.headers.get("Content-Type", ""))
            data = resp.read(5_000_000)  # 5MB guardrail

        text = _decode_bytes(data, ctype)
        if "html" in (ctype or "").lower():
            content = _strip_html(text)
        else:
            content = text
    except Exception as exc:
        error = str(exc)

    out = {"url": final_url, "status": status, "content": content, "error": error}
    try:
        _save_json(cache_file, out)
    except Exception:
        pass
    return out


@dataclass
class _ArxivEntry:
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    url: str = ""
    published: str = ""


class _ArxivAtomParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entries: List[_ArxivEntry] = []
        self._in_entry = False
        self._in_author = False
        self._field: Optional[str] = None
        self._buf: List[str] = []
        self._cur: Optional[_ArxivEntry] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = (tag or "").lower()
        if t == "entry":
            self._in_entry = True
            self._cur = _ArxivEntry()
            self._field = None
            self._buf = []
            return

        if not self._in_entry or self._cur is None:
            return

        if t == "author":
            self._in_author = True
            return

        if t in {"title", "summary", "published", "id"}:
            self._field = t
            self._buf = []
            return

        if t == "name" and self._in_author:
            self._field = "author_name"
            self._buf = []
            return

        if t == "link":
            d = {k.lower(): (v or "") for k, v in attrs}
            href = d.get("href", "").strip()
            rel = d.get("rel", "").strip().lower()
            typ = d.get("type", "").strip().lower()
            if href and (rel == "alternate" or (not rel and typ in {"text/html", ""})):
                if not self._cur.url:
                    self._cur.url = href

    def handle_endtag(self, tag: str) -> None:
        t = (tag or "").lower()

        if t == "entry":
            if self._cur is not None:
                if not self._cur.url:
                    self._cur.url = self._cur.url or ""
                self.entries.append(self._cur)
            self._cur = None
            self._in_entry = False
            self._in_author = False
            self._field = None
            self._buf = []
            return

        if not self._in_entry or self._cur is None:
            return

        if t == "author":
            self._in_author = False
            return

        if self._field is None:
            return

        # Assign on closing tag.
        if t == self._field:
            val = " ".join("".join(self._buf).split())
            if self._field == "title":
                self._cur.title = val
            elif self._field == "summary":
                self._cur.abstract = val
            elif self._field == "published":
                self._cur.published = val
            elif self._field == "id":
                if not self._cur.url:
                    self._cur.url = val
            self._field = None
            self._buf = []
            return

        if self._field == "author_name" and t == "name":
            name = " ".join("".join(self._buf).split())
            if name:
                self._cur.authors.append(name)
            self._field = None
            self._buf = []

    def handle_data(self, data: str) -> None:
        if not self._in_entry or self._field is None:
            return
        if data:
            self._buf.append(data)


def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search arXiv via Atom API and return:
        [{title, authors, abstract, url, published}, ...]

    Parsing is done with html.parser (stdlib) for portability.
    """
    q = (query or "").strip()
    if not q:
        return []

    try:
        mr = max(1, min(int(max_results), 50))
    except Exception:
        mr = 5

    search_q = q if ":" in q else f"all:{q}"
    params = {"search_query": search_q, "start": 0, "max_results": mr}
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _DEFAULT_UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read(5_000_000)
        xml = data.decode("utf-8", errors="replace")
    except Exception:
        return []

    p = _ArxivAtomParser()
    try:
        p.feed(xml)
        p.close()
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for e in p.entries[:mr]:
        out.append(
            {
                "title": e.title,
                "authors": e.authors,
                "abstract": e.abstract,
                "url": e.url,
                "published": e.published,
            }
        )
    return out


def fetch_binance_funding_rates(symbol: str = "BTCUSDT", limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch Binance USDM perpetual funding rates from the public REST API.

    Returns: [{symbol, fundingTime, fundingRate}, ...]
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        sym = "BTCUSDT"

    try:
        lim = max(1, min(int(limit), 1000))
    except Exception:
        lim = 100

    base = "https://fapi.binance.com/fapi/v1/fundingRate"
    url = base + "?" + urllib.parse.urlencode({"symbol": sym, "limit": lim})

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _DEFAULT_UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read(5_000_000)
        obj = json.loads(data.decode("utf-8", errors="replace"))
    except Exception:
        return []

    if not isinstance(obj, list):
        return []

    out: List[Dict[str, Any]] = []
    for row in obj:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "symbol": row.get("symbol", sym),
                "fundingTime": row.get("fundingTime"),
                "fundingRate": row.get("fundingRate"),
            }
        )
    return out


def fetch_binance_market_overview(limit: int = 20) -> Dict[str, Any]:
    """
    Fetch top Binance USDM perpetual tickers sorted by 24h quote volume.
    Returns: {"tickers": [{symbol, lastPrice, priceChangePercent, quoteVolume}, ...]}
    """
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _DEFAULT_UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read(5_000_000)
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return {"tickers": [], "error": "Fetch failed"}

    if not isinstance(data, list):
        return {"tickers": []}

    usdt_perps = [
        t for t in data
        if isinstance(t, dict) and str(t.get("symbol", "")).endswith("USDT")
    ]
    # Sort by 24h quote volume descending
    try:
        usdt_perps.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
    except Exception:
        pass

    tickers = []
    for t in usdt_perps[:limit]:
        tickers.append({
            "symbol": t.get("symbol", ""),
            "lastPrice": t.get("lastPrice", "0"),
            "priceChange": t.get("priceChange", "0"),
            "priceChangePercent": t.get("priceChangePercent", "0"),
            "volume": t.get("volume", "0"),
            "quoteVolume": t.get("quoteVolume", "0"),
        })

    return {"tickers": tickers}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_sentence(text: str, max_chars: int = 280) -> str:
    t = " ".join((text or "").split()).strip()
    if not t:
        return ""
    for sep in [". ", "? ", "! "]:
        i = t.find(sep)
        if 0 < i < max_chars:
            return t[: i + 1]
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def research_strategy(topic: str, artifacts_dir: Path) -> Dict[str, Any]:
    """
    Lightweight strategy research helper:
    - arXiv search for relevant papers
    - fetch (and cache) the arXiv abstract pages for quick context
    - store a short summary to long-term memory (SQLite) under artifacts_dir/memory/memory.db

    Returns: {papers: [], insights: [], stored_to_memory: bool}
    """
    t = (topic or "").strip()
    if not t:
        return {"papers": [], "insights": [], "stored_to_memory": False}

    artifacts_dir = Path(artifacts_dir)
    papers = search_arxiv(t, max_results=5)

    # "Web fetch" component: pull a couple of arXiv abstract pages (best-effort).
    fetched_pages: List[Dict[str, Any]] = []
    for p in papers[:3]:
        u = str(p.get("url", "")).strip()
        if not u:
            continue
        res = fetch_url(u, timeout=15)
        fetched_pages.append({"url": res.get("url", u), "status": res.get("status", None), "error": res.get("error", "")})

    insights: List[str] = []
    if papers:
        insights.append(f"Found {len(papers)} arXiv papers for topic: {t!r}.")
        for p in papers[:5]:
            title = str(p.get("title", "")).strip()
            published = str(p.get("published", "")).strip()
            abstract = str(p.get("abstract", "")).strip()
            snippet = _first_sentence(abstract)
            if title and published and snippet:
                insights.append(f"{title} ({published}): {snippet}")
            elif title and snippet:
                insights.append(f"{title}: {snippet}")
            elif title:
                insights.append(title)
    else:
        insights.append(f"No arXiv results for topic: {t!r}.")

    report_lines: List[str] = []
    report_lines.append(f"# Strategy research: {t}")
    report_lines.append("")
    report_lines.append("## Insights")
    for it in insights:
        report_lines.append(f"- {it}")
    report_lines.append("")
    report_lines.append("## Papers")
    for p in papers:
        title = str(p.get("title", "")).strip()
        published = str(p.get("published", "")).strip()
        url = str(p.get("url", "")).strip()
        authors = p.get("authors") or []
        if isinstance(authors, list):
            author_str = ", ".join(str(a) for a in authors if a)
        else:
            author_str = str(authors)
        report_lines.append(f"- {title} ({published})")
        if author_str:
            report_lines.append(f"  - Authors: {author_str}")
        if url:
            report_lines.append(f"  - URL: {url}")
    report_lines.append("")
    report_lines.append("## Web fetches")
    for f in fetched_pages:
        report_lines.append(f"- {f.get('url')} status={f.get('status')} err={f.get('error')}")
    report = "\n".join(report_lines).strip() + "\n"

    stored = False
    try:
        from ..memory.store import MemoryStore

        mem_db = artifacts_dir / "memory" / "memory.db"
        ms = MemoryStore(mem_db)
        try:
            ms.add(
                created_at=_utc_iso(),
                kind="research_strategy",
                tags=["research", "strategy"],
                content=report,
                meta={"topic": t, "paper_urls": [p.get("url") for p in papers if p.get("url")]},
                run_id=None,
            )
            stored = True
        finally:
            ms.close()
    except Exception:
        stored = False

    return {"papers": papers, "insights": insights, "stored_to_memory": stored}
