from __future__ import annotations

"""
NEXUS Research Paper Fetcher.

Fetches the latest quantitative finance, ML, and crypto research papers
from arXiv, SSRN-style APIs, and curated sources.

Runs daily as part of the NEXUS self-learning routine:
  1. Fetch latest papers from arXiv categories: q-fin.PM, q-fin.TR, cs.LG, stat.ML
  2. Filter for crypto/algorithmic trading relevance
  3. Score relevance with LLM
  4. Extract actionable insights
  5. Queue improvement hypotheses to Kanban

arXiv API: https://export.arxiv.org/api/query
No API key needed — fully public.
"""

import json
import math
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


_NEXUS_KEYWORDS = [
    # Core quant alpha
    "momentum", "carry", "mean reversion", "factor model", "cross-sectional",
    "time-series momentum", "funding rate", "perpetual futures", "crypto",
    "cryptocurrency", "bitcoin", "altcoin",
    # Risk & portfolio
    "risk parity", "volatility targeting", "kelly criterion", "drawdown",
    "portfolio optimization", "mean variance",
    # ML / Deep learning applied to finance
    "reinforcement learning trading", "neural network portfolio", "gradient boosting finance",
    "online learning trading", "regime detection", "hidden markov",
    # Market microstructure
    "order flow", "liquidity", "market impact", "high frequency",
    # Recent trends
    "large language model finance", "alpha signal", "systematic trading",
]

_ARXIV_CATEGORIES = [
    "q-fin.PM",  # Portfolio Management
    "q-fin.TR",  # Trading and Market Microstructure
    "q-fin.ST",  # Statistical Finance
    "q-fin.MF",  # Mathematical Finance
    "cs.LG",     # Machine Learning
    "stat.ML",   # Statistics ML
]

_NS = "http://www.w3.org/2005/Atom"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relevance_score(title: str, abstract: str) -> float:
    """Simple keyword-based relevance scoring (0-1)."""
    text = (title + " " + abstract).lower()
    hits = sum(1 for kw in _NEXUS_KEYWORDS if kw.lower() in text)
    return min(1.0, hits / 3.0)  # cap at 1.0 after 3 keyword hits


def fetch_arxiv_papers(
    categories: Optional[List[str]] = None,
    max_results: int = 20,
    days_back: int = 7,
    min_relevance: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Fetch recent papers from arXiv matching NEXUS research interests.

    Returns list of paper dicts: {id, title, abstract, authors, published, url, relevance}
    """
    if categories is None:
        categories = _ARXIV_CATEGORIES[:4]  # focus on q-fin categories

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    papers = []

    for cat in categories:
        try:
            query = urllib.parse.urlencode({
                "search_query": f"cat:{cat}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            })
            url = f"https://export.arxiv.org/api/query?{query}"
            req = urllib.request.Request(url, headers={"User-Agent": "NEXUS-Research/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                xml_text = resp.read().decode("utf-8")
            root = ET.fromstring(xml_text)
            for entry in root.findall(f"{{{_NS}}}entry"):
                paper_id = (entry.findtext(f"{{{_NS}}}id") or "").strip()
                title = (entry.findtext(f"{{{_NS}}}title") or "").strip().replace("\n", " ")
                abstract = (entry.findtext(f"{{{_NS}}}summary") or "").strip().replace("\n", " ")
                published = (entry.findtext(f"{{{_NS}}}published") or "").strip()
                authors = [
                    a.findtext(f"{{{_NS}}}name") or ""
                    for a in entry.findall(f"{{{_NS}}}author")
                ]
                relevance = _relevance_score(title, abstract)
                if relevance < min_relevance:
                    continue
                # Check date
                try:
                    pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    if pub_dt < cutoff:
                        continue
                except Exception:
                    pass
                papers.append({
                    "id": paper_id,
                    "title": title,
                    "abstract": abstract[:800],
                    "authors": authors[:4],
                    "published": published,
                    "url": paper_id,
                    "category": cat,
                    "relevance": round(relevance, 3),
                    "fetched_at": _now_iso(),
                })
        except Exception as e:
            # Silent fail — don't crash the whole loop on network issues
            print(f"[arxiv] Failed to fetch {cat}: {e}")

    # Deduplicate by ID
    seen = set()
    unique = []
    for p in sorted(papers, key=lambda x: x["relevance"], reverse=True):
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)

    return unique[:50]  # cap at 50 papers per cycle


def fetch_crypto_news_headlines() -> List[Dict[str, Any]]:
    """
    Fetch crypto news headlines from public sources (no API key).
    Uses CoinGecko news endpoint and other public feeds.
    """
    headlines = []
    sources = [
        # CoinGecko public news (no auth)
        ("https://api.coingecko.com/api/v3/news", "coingecko"),
    ]
    for url, source in sources:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NEXUS/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = data if isinstance(data, list) else data.get("data", data.get("news", []))
            for item in items[:20]:
                title = item.get("title") or item.get("headline") or ""
                headlines.append({
                    "title": title,
                    "url": item.get("url") or item.get("link") or "",
                    "source": source,
                    "ts": item.get("updated_at") or item.get("published_at") or _now_iso(),
                    "relevance": _relevance_score(title, ""),
                })
        except Exception as e:
            print(f"[news] Failed {source}: {e}")

    return sorted(headlines, key=lambda x: x["relevance"], reverse=True)[:15]


def extract_hypotheses_from_papers(
    papers: List[Dict[str, Any]],
    llm_client: Any = None,
) -> List[Dict[str, Any]]:
    """
    Extract actionable trading hypotheses from papers using LLM (GLM-5).
    Falls back to keyword-based extraction if no LLM available.
    """
    if not papers:
        return []

    hypotheses = []

    # Keyword-based hypothesis extraction (always runs as fallback)
    hypothesis_patterns = [
        (["momentum", "cross-sectional", "ranking"], "cross_sectional_momentum",
         "Implement cross-sectional momentum with ranking-based portfolio"),
        (["funding rate", "carry", "perpetual"], "funding_carry",
         "Optimize funding rate carry with current market regime filter"),
        (["volatility", "target", "risk parity"], "volatility_targeting",
         "Apply volatility targeting to reduce drawdown while maintaining returns"),
        (["mean reversion", "reversal", "contrarian"], "mean_reversion",
         "Test mean reversion at 4h-48h horizon for crypto"),
        (["machine learning", "gradient boosting", "random forest"], "ml_alpha",
         "Apply tree-based ML model for factor combination"),
        (["regime", "hidden markov", "switching"], "regime_detection",
         "Improve regime detection using hidden Markov model"),
        (["open interest", "liquidation", "market impact"], "microstructure",
         "Use open interest and liquidation data as alpha signal"),
        (["sentiment", "social", "twitter"], "sentiment_alpha",
         "Add social sentiment signal for entry timing"),
    ]

    paper_text = " ".join(p["title"] + " " + p["abstract"] for p in papers[:10]).lower()
    for keywords, strategy_type, hypothesis_text in hypothesis_patterns:
        if sum(1 for kw in keywords if kw in paper_text) >= 2:
            hypotheses.append({
                "hypothesis": hypothesis_text,
                "strategy_type": strategy_type,
                "confidence": 0.6,
                "source": "keyword_extraction",
                "papers": [p["title"][:60] for p in papers[:3]],
                "priority": "medium",
                "ts": _now_iso(),
            })

    # LLM-based extraction (if available)
    if llm_client and papers:
        try:
            paper_summaries = "\n".join([
                f"- [{p['category']}] {p['title']}: {p['abstract'][:200]}"
                for p in papers[:8]
            ])
            prompt = f"""You are NEXUS, an autonomous quantitative research AI.
Based on these recent research papers, generate 3 specific, actionable hypotheses
for improving a crypto perpetual futures trading strategy.

Papers:
{paper_summaries}

Current NEXUS strategies: funding_carry_perp_v1, momentum_xs_v1, ml_factor_xs_v1

For each hypothesis, respond in JSON format:
[
  {{
    "hypothesis": "specific improvement idea",
    "strategy_type": "which strategy type",
    "expected_sharpe_improvement": 0.1-0.5,
    "implementation_notes": "brief implementation guide",
    "priority": "high/medium/low"
  }}
]

Focus on improvements that can be implemented with OHLCV + funding rate data."""

            response = llm_client.messages.create(
                model="glm-5",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""
            m = re.search(r'\[.*\]', text, re.DOTALL)
            if m:
                llm_hyps = json.loads(m.group(0))
                for h in llm_hyps:
                    h["source"] = "llm_extraction"
                    h["ts"] = _now_iso()
                    h["confidence"] = min(0.9, h.get("expected_sharpe_improvement", 0.1) * 2)
                    hypotheses.insert(0, h)  # LLM hypotheses first
        except Exception as e:
            print(f"[arxiv] LLM extraction failed: {e}")

    return hypotheses[:10]


def save_research_session(
    papers: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    headlines: List[Dict[str, Any]],
    artifacts_dir: Path,
) -> Path:
    """Save research session to artifacts/research/"""
    research_dir = artifacts_dir / "research"
    research_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session = {
        "ts": _now_iso(),
        "papers_fetched": len(papers),
        "hypotheses_generated": len(hypotheses),
        "headlines_fetched": len(headlines),
        "papers": papers,
        "hypotheses": hypotheses,
        "headlines": headlines,
    }

    # Save session file
    session_path = research_dir / f"session_{ts}.json"
    session_path.write_text(json.dumps(session, indent=2), encoding="utf-8")

    # Append to running log
    log_path = research_dir / "research_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": _now_iso(),
            "papers": len(papers),
            "hypotheses": len(hypotheses),
            "top_papers": [p["title"][:80] for p in papers[:5]],
            "top_hypotheses": [h["hypothesis"][:80] for h in hypotheses[:3]],
        }) + "\n")

    return session_path
