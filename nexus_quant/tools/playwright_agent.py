from __future__ import annotations

"""
NEXUS Playwright Browser Agent.

Autonomous browser control using Playwright (or fallback to urllib/http.client).
Provides:
- Web page fetching + screenshot
- Form interaction + data extraction
- API endpoint polling
- Research gathering from financial news sites

Install: pip install playwright && playwright install chromium
Fallback: stdlib urllib.request (no JS rendering)
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlaywrightAgent:
    """
    Browser automation agent with GLM-5/Claude integration.
    Falls back to urllib if Playwright is not installed.
    """

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self._log_path = self.artifacts_dir / "browser" / "runs.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._has_playwright = self._check_playwright()

    def _check_playwright(self) -> bool:
        try:
            import playwright  # noqa: F401
            return True
        except ImportError:
            return False

    def _fetch_with_urllib(self, url: str, timeout: int = 15) -> str:
        """Fallback: fetch URL via stdlib urllib."""
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; NEXUS-Bot/1.0)",
                "Accept": "text/html,application/json,*/*",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return raw.decode("utf-8", errors="replace")[:50000]
        except Exception as e:
            return f"[error fetching {url}: {e}]"

    def _fetch_with_playwright(self, url: str, task: str, timeout_ms: int = 20000) -> Dict[str, Any]:
        """Fetch with Playwright - returns content + screenshot."""
        try:
            from playwright.sync_api import sync_playwright
            result: Dict[str, Any] = {"url": url, "content": "", "screenshot": None, "title": ""}
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 NEXUS-Research/1.0"})
                page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                page.wait_for_timeout(1500)
                result["title"] = page.title() or ""
                result["content"] = page.content()[:30000]
                result["text"] = page.inner_text("body")[:10000] if page.locator("body").count() else ""
                # Screenshot to file
                ss_dir = self.artifacts_dir / "browser" / "screenshots"
                ss_dir.mkdir(parents=True, exist_ok=True)
                ts = _now_iso()[:19].replace(":", "-")
                ss_path = ss_dir / f"ss_{ts}.png"
                page.screenshot(path=str(ss_path), full_page=False)
                result["screenshot"] = str(ss_path)
                browser.close()
            return result
        except Exception as e:
            return {"url": url, "content": "", "error": str(e)}

    def _analyze_with_llm(self, content: str, url: str, task: str) -> str:
        """Use GLM-5 via ZAI to analyze page content."""
        try:
            import os
            import anthropic
            api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            base_url = os.environ.get("ZAI_ANTHROPIC_BASE_URL")
            model = os.environ.get("ZAI_DEFAULT_MODEL", "glm-4-flash")
            if not api_key:
                return ""
            kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 1}
            if base_url:
                kwargs["base_url"] = base_url
            client = anthropic.Anthropic(**kwargs)
            system = (
                "You are NEXUS FLUX — browser research agent for a quant trading system. "
                "Extract key financial data, signals, and insights from web content. "
                "Be concise and structured. Output a brief JSON with: summary, key_data, signals."
            )
            user = f"URL: {url}\nTask: {task}\n\nPage content (first 3000 chars):\n{content[:3000]}"
            resp = client.messages.create(
                model=model,
                max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            if resp.content:
                return str(resp.content[0].text)
        except Exception:
            pass
        return ""

    def run(self, url: str, task: str = "extract key data") -> Dict[str, Any]:
        """
        Run browser task: fetch URL, optionally analyze with LLM.
        Returns structured result with content + analysis.
        """
        t0 = time.time()
        result: Dict[str, Any] = {
            "ts": _now_iso(),
            "url": url,
            "task": task,
            "ok": False,
            "content_length": 0,
            "analysis": "",
            "error": "",
            "mode": "playwright" if self._has_playwright else "urllib",
        }

        try:
            if self._has_playwright:
                page_data = self._fetch_with_playwright(url, task)
                content = page_data.get("text") or page_data.get("content") or ""
                result["title"] = page_data.get("title", "")
                result["screenshot"] = page_data.get("screenshot")
                result["error"] = page_data.get("error", "")
            else:
                content = self._fetch_with_urllib(url)
                result["title"] = ""

            result["content_length"] = len(content)
            result["content_preview"] = content[:2000]

            # LLM analysis
            if content and len(content) > 100:
                analysis = self._analyze_with_llm(content, url, task)
                result["analysis"] = analysis

            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)

        result["duration_sec"] = round(time.time() - t0, 2)

        # Log to JSONL
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                log_entry = {k: v for k, v in result.items() if k not in ("content_preview",)}
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return result

    def fetch_financial_news(self, query: str = "crypto funding rate perpetuals") -> List[Dict[str, Any]]:
        """Fetch financial research from multiple sources."""
        sources = [
            f"https://www.binance.com/en/research",
            f"https://dune.com/browse/dashboards?q={query.replace(' ', '+')}&filter=trending",
        ]
        results = []
        for url in sources[:2]:
            try:
                r = self.run(url=url, task=f"Find articles about: {query}")
                if r.get("ok"):
                    results.append(r)
            except Exception:
                continue
        return results

    def recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent browser runs from log."""
        if not self._log_path.exists():
            return []
        lines = self._log_path.read_text("utf-8").splitlines()[-limit:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return list(reversed(out))
