"""
NEXUS Multi-Source RSS/API Fetcher — stdlib only.

Fetches from 50+ sources (RSS, Reddit RSS, arXiv, CoinGecko API).
Scores each item for relevance to NEXUS quant trading topics.
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .source_registry import SOURCES, NEXUS_KEYWORDS, get_sources_by_kind

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
_REQUEST_TIMEOUT = 15          # seconds per HTTP request
_MAX_ITEMS_PER_SOURCE = 15     # max articles per source
_MIN_RELEVANCE = 0.05          # discard items below this threshold
_HEADERS = {
    "User-Agent": "NEXUSResearchBot/2.0 (+https://github.com/theanh97/nexus-quant)",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}
_REDDIT_HEADERS = {
    "User-Agent": "NEXUSQuantBot/2.0 (automated research; contact: nexus@quant.io)",
    "Accept": "application/json, application/rss+xml",
}


# ────────────────────────────────────────────────────────────────────────────
# Relevance scoring
# ────────────────────────────────────────────────────────────────────────────

def _score_relevance(text: str, source_tags: List[str], source_weight: float = 1.0) -> float:
    """Score 0-1 relevance of text to NEXUS quant topics."""
    text_lower = text.lower()
    hit_count = 0
    for kw in NEXUS_KEYWORDS:
        if kw in text_lower:
            hit_count += 1
    # Also boost for source-specific tags
    tag_hits = sum(1 for t in source_tags if t.replace("_", " ") in text_lower or t in text_lower)
    raw = (hit_count / max(1, len(NEXUS_KEYWORDS))) + 0.1 * tag_hits
    # Sigmoid-like squash so high-overlap articles don't dominate too much
    squashed = 1.0 - math.exp(-3.0 * raw)
    return min(1.0, squashed * source_weight)


# ────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ────────────────────────────────────────────────────────────────────────────

def _http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = _REQUEST_TIMEOUT) -> Optional[str]:
    """Fetch URL, return body text or None on error."""
    try:
        req = urllib.request.Request(url, headers=headers or _HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            encoding = resp.headers.get_content_charset("utf-8") or "utf-8"
            return raw.decode(encoding, errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, Exception):
        return None


# ────────────────────────────────────────────────────────────────────────────
# RSS parser
# ────────────────────────────────────────────────────────────────────────────

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "arxiv": "http://arxiv.org/schemas/atom",
    "media": "http://search.yahoo.com/mrss/",
}


def _get_text(el: Optional[ET.Element], tag: str, default: str = "") -> str:
    if el is None:
        return default
    child = el.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return default


def _parse_rss_items(xml_text: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse RSS/Atom XML into list of article dicts."""
    items: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    # Detect feed type
    tag = root.tag.lower()
    is_atom = "atom" in tag or root.tag.endswith("}feed") or root.tag == "{http://www.w3.org/2005/Atom}feed"

    if is_atom:
        # Atom feed
        entries = root.findall("{http://www.w3.org/2005/Atom}entry") or root.findall("entry")
        for entry in entries[:_MAX_ITEMS_PER_SOURCE]:
            title_el = entry.find("{http://www.w3.org/2005/Atom}title") or entry.find("title")
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            # summary/content
            summ_el = (entry.find("{http://www.w3.org/2005/Atom}summary")
                       or entry.find("{http://www.w3.org/2005/Atom}content")
                       or entry.find("summary") or entry.find("content"))
            summary = summ_el.text.strip() if summ_el is not None and summ_el.text else ""
            # link
            link_el = entry.find("{http://www.w3.org/2005/Atom}link") or entry.find("link")
            link = ""
            if link_el is not None:
                link = link_el.get("href", "") or (link_el.text or "")
            # published
            pub_el = (entry.find("{http://www.w3.org/2005/Atom}published")
                      or entry.find("{http://www.w3.org/2005/Atom}updated")
                      or entry.find("published") or entry.find("updated"))
            published = pub_el.text.strip() if pub_el is not None and pub_el.text else ""

            combined = f"{title} {summary}"
            relevance = _score_relevance(combined, source.get("tags", []), source.get("weight", 1.0))
            if relevance >= _MIN_RELEVANCE:
                items.append({
                    "source": source["name"],
                    "source_label": source["label"],
                    "category": source["category"],
                    "title": title,
                    "summary": summary[:500],
                    "url": link,
                    "published": published,
                    "relevance": round(relevance, 4),
                    "tags": source.get("tags", []),
                })
    else:
        # RSS 2.0 — channel > item
        channel = root.find("channel")
        entry_list = (channel.findall("item") if channel is not None else []) or root.findall("item")
        for entry in entry_list[:_MAX_ITEMS_PER_SOURCE]:
            title = _get_text(entry, "title")
            link = _get_text(entry, "link")
            desc = _get_text(entry, "description")
            # Also try content:encoded
            content_el = entry.find("{http://purl.org/rss/1.0/modules/content/}encoded")
            if content_el is not None and content_el.text:
                desc = content_el.text[:500]
            pub_date = _get_text(entry, "pubDate")
            dc_date = _get_text(entry, "{http://purl.org/dc/elements/1.1/}date")
            published = pub_date or dc_date

            combined = f"{title} {desc}"
            relevance = _score_relevance(combined, source.get("tags", []), source.get("weight", 1.0))
            if relevance >= _MIN_RELEVANCE:
                items.append({
                    "source": source["name"],
                    "source_label": source["label"],
                    "category": source["category"],
                    "title": title,
                    "summary": desc[:500],
                    "url": link,
                    "published": published,
                    "relevance": round(relevance, 4),
                    "tags": source.get("tags", []),
                })
    return items


# ────────────────────────────────────────────────────────────────────────────
# CoinGecko API
# ────────────────────────────────────────────────────────────────────────────

def _fetch_coingecko(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch trending coins + market data from CoinGecko public API."""
    items: List[Dict[str, Any]] = []
    urls = [
        ("trending", "https://api.coingecko.com/api/v3/search/trending"),
        ("global", "https://api.coingecko.com/api/v3/global"),
    ]
    for key, url in urls:
        body = _http_get(url, headers={"User-Agent": _HEADERS["User-Agent"]})
        if not body:
            continue
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            continue
        if key == "trending":
            coins = data.get("coins") or []
            for c in coins[:10]:
                item_data = c.get("item") or {}
                name = item_data.get("name", "")
                symbol = item_data.get("symbol", "")
                market_cap_rank = item_data.get("market_cap_rank", 0)
                title = f"Trending: {name} ({symbol}) — Market Cap Rank #{market_cap_rank}"
                text = f"trending crypto {name} {symbol}"
                relevance = _score_relevance(text, source.get("tags", []), source.get("weight", 1.0))
                items.append({
                    "source": source["name"],
                    "source_label": source["label"],
                    "category": source["category"],
                    "title": title,
                    "summary": f"CoinGecko trending coin. Market cap rank: {market_cap_rank}",
                    "url": f"https://www.coingecko.com/en/coins/{item_data.get('id', '')}",
                    "published": datetime.now(timezone.utc).isoformat(),
                    "relevance": round(max(relevance, 0.3), 4),
                    "tags": source.get("tags", []),
                    "raw": {"rank": market_cap_rank, "symbol": symbol},
                })
        elif key == "global":
            gdata = data.get("data") or {}
            total_mcap = gdata.get("total_market_cap", {}).get("usd", 0)
            btc_dom = gdata.get("market_cap_percentage", {}).get("btc", 0)
            title = f"Crypto Market: Total MCap ${total_mcap/1e9:.1f}B | BTC Dom {btc_dom:.1f}%"
            items.append({
                "source": source["name"],
                "source_label": "CoinGecko Global",
                "category": source["category"],
                "title": title,
                "summary": (
                    f"Market cap change 24h: {gdata.get('market_cap_change_percentage_24h_usd', 0):.2f}% | "
                    f"Active coins: {gdata.get('active_cryptocurrencies', 0)} | "
                    f"Markets: {gdata.get('markets', 0)}"
                ),
                "url": "https://www.coingecko.com/en/global_charts",
                "published": datetime.now(timezone.utc).isoformat(),
                "relevance": 0.6,
                "tags": source.get("tags", []),
                "raw": gdata,
            })
    return items


# ────────────────────────────────────────────────────────────────────────────
# Main fetcher
# ────────────────────────────────────────────────────────────────────────────

def fetch_all_sources(
    sources: Optional[List[Dict[str, Any]]] = None,
    min_relevance: float = _MIN_RELEVANCE,
    rate_limit_seconds: float = 0.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch from all (or given) sources. Returns (items, stats).
    items: list of article dicts sorted by relevance desc.
    stats: {"total": N, "fetched": M, "failed_sources": [...]}
    """
    if sources is None:
        sources = SOURCES

    all_items: List[Dict[str, Any]] = []
    failed: List[str] = []
    fetched_sources = 0

    for src in sources:
        kind = src.get("kind", "rss")
        try:
            if kind == "coingecko_api":
                items = _fetch_coingecko(src)
            else:
                # RSS / Atom / Reddit RSS
                body = _http_get(src["url"])
                if body is None:
                    failed.append(src["name"])
                    continue
                items = _parse_rss_items(body, src)

            items = [i for i in items if i.get("relevance", 0.0) >= min_relevance]
            all_items.extend(items)
            fetched_sources += 1
        except Exception:
            failed.append(src["name"])

        if rate_limit_seconds > 0:
            time.sleep(rate_limit_seconds)

    # Sort by relevance desc
    all_items.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)
    stats = {
        "total_sources": len(sources),
        "fetched_sources": fetched_sources,
        "failed_sources": failed,
        "total_items": len(all_items),
    }
    return all_items, stats


def fetch_quick(
    categories: Optional[List[str]] = None,
    max_sources: int = 20,
    min_relevance: float = 0.1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch a quick subset — fewer sources, faster."""
    pool = SOURCES
    if categories:
        pool = [s for s in SOURCES if s["category"] in categories]
    # Sort by weight desc, take top max_sources
    pool_sorted = sorted(pool, key=lambda s: s.get("weight", 0.5), reverse=True)
    subset = pool_sorted[:max_sources]
    return fetch_all_sources(subset, min_relevance=min_relevance, rate_limit_seconds=0.3)


# ────────────────────────────────────────────────────────────────────────────
# Hypothesis extraction (deterministic + optional LLM)
# ────────────────────────────────────────────────────────────────────────────

# Trading strategy keywords → hypothesis template
_STRATEGY_SIGNALS: List[Tuple[str, str]] = [
    ("momentum", "HYPOTHESIS: Apply momentum signal over {days}-day lookback to crypto universe"),
    ("mean reversion", "HYPOTHESIS: Mean-reversion strategy — buy recent losers, sell recent winners"),
    ("carry", "HYPOTHESIS: Funding carry — go long high-funding assets (or long-only mode)"),
    ("trend following", "HYPOTHESIS: Trend-following with vol-targeting on multi-asset crypto"),
    ("factor model", "HYPOTHESIS: Multi-factor cross-sectional alpha with rolling OLS regression"),
    ("machine learning", "HYPOTHESIS: ML-based return prediction with lagged features and walk-forward"),
    ("volatility targeting", "HYPOTHESIS: Scale position size inversely to realized 30-day volatility"),
    ("risk parity", "HYPOTHESIS: Risk-parity portfolio allocation across crypto assets"),
    ("regime", "HYPOTHESIS: Regime-switching strategy — different alpha model per market regime"),
    ("ensemble", "HYPOTHESIS: Ensemble of strategies, weighted by recent out-of-sample Sharpe"),
    ("dollar neutral", "HYPOTHESIS: Dollar-neutral long-short portfolio — long top K, short bottom K"),
    ("basis trade", "HYPOTHESIS: Perp-spot basis arbitrage — long spot, short perp when basis > threshold"),
    ("liquidation", "HYPOTHESIS: Liquidation cascade detector — go opposite direction after mass liquidations"),
    ("funding rate", "HYPOTHESIS: Funding rate alpha — aggregate funding signal across all perp assets"),
    ("open interest", "HYPOTHESIS: Open interest surge detector — signal from OI change rate"),
]


def extract_hypotheses(items: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Extract actionable trading hypotheses from fetched items.
    Returns list of {hypothesis, confidence, source_title, source_url, trigger_keyword}.
    """
    hypotheses: List[Dict[str, Any]] = []
    seen_hypotheses: set = set()

    top_items = sorted(items, key=lambda x: x.get("relevance", 0.0), reverse=True)[:top_n * 3]

    for item in top_items:
        text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        for keyword, template in _STRATEGY_SIGNALS:
            if keyword in text:
                hyp = template.replace("{days}", "30")
                if hyp not in seen_hypotheses:
                    seen_hypotheses.add(hyp)
                    hypotheses.append({
                        "hypothesis": hyp,
                        "confidence": round(item.get("relevance", 0.5), 3),
                        "trigger_keyword": keyword,
                        "source_title": item.get("title", ""),
                        "source_url": item.get("url", ""),
                        "source_name": item.get("source", ""),
                    })
        if len(hypotheses) >= top_n:
            break

    return hypotheses[:top_n]


# ────────────────────────────────────────────────────────────────────────────
# Save session
# ────────────────────────────────────────────────────────────────────────────

def save_research_session(
    items: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    stats: Dict[str, Any],
    artifacts_dir: Path,
) -> Path:
    """Save research session to artifacts/research/ and append to research_log.jsonl."""
    research_dir = artifacts_dir / "research"
    research_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_file = research_dir / f"session_{ts}.json"

    session = {
        "timestamp": ts,
        "stats": stats,
        "top_items": items[:50],
        "hypotheses": hypotheses,
    }
    session_file.write_text(json.dumps(session, indent=2, default=str), encoding="utf-8")

    # Append summary to rolling log
    log_file = research_dir / "research_log.jsonl"
    log_entry = {
        "ts": ts,
        "sources_fetched": stats.get("fetched_sources", 0),
        "total_items": stats.get("total_items", 0),
        "hypotheses_count": len(hypotheses),
        "top_hypothesis": hypotheses[0]["hypothesis"] if hypotheses else "",
        "session_file": str(session_file.name),
    }
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entry, default=str) + "\n")

    return session_file
