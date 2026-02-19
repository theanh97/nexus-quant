from __future__ import annotations

"""
NEXUS Quant Web Dashboard.

Serves a single-page dashboard over FastAPI.
Requires: pip install fastapi uvicorn[standard]

Usage:
    python3 -m nexus_quant dashboard --artifacts artifacts --port 8080
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel
    class ChatRequest(BaseModel):
        message: str = ""
    class FeedbackRequest(BaseModel):
        message: str = ""
        tags: List[str] = []
except ImportError:
    ChatRequest = None  # type: ignore
    FeedbackRequest = None  # type: ignore


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl_tail(path: Path, n: int = 50) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-n:]
    out = []
    for ln in lines:
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _latest_run_result(artifacts_dir: Path) -> Optional[Dict[str, Any]]:
    runs_dir = artifacts_dir / "runs"
    if not runs_dir.exists():
        return None
    dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    if not dirs:
        return None
    return _read_json(dirs[-1] / "result.json")


def _latest_metrics(artifacts_dir: Path) -> Optional[Dict[str, Any]]:
    runs_dir = artifacts_dir / "runs"
    if not runs_dir.exists():
        return None
    dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    if not dirs:
        return None
    return _read_json(dirs[-1] / "metrics.json")


def serve(artifacts_dir: Path, port: int = 8080, host: str = "127.0.0.1") -> None:
    """Launch the FastAPI dashboard server."""
    try:
        import fastapi
        import uvicorn
    except ImportError:
        raise SystemExit("Dashboard requires FastAPI + uvicorn. Run: pip install fastapi uvicorn[standard]")

    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn

    app = FastAPI(title="NEXUS Quant Dashboard", version="1.0")
    static_dir = Path(__file__).parent / "static"

    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _safe_resolve_path(root: Path, p: Path) -> Optional[Path]:
        try:
            rp = p.resolve()
        except Exception:
            return None
        try:
            rp.relative_to(root.resolve())
        except Exception:
            return None
        return rp

    def _extract_tail_after(phrase: str, text: str) -> Optional[str]:
        import re
        m = re.search(rf"\\b{re.escape(phrase)}\\b(.*)$", text, flags=re.IGNORECASE)
        if not m:
            return None
        raw = (m.group(1) or "").strip()
        return raw or None

    def _candidate_path_strings(text: str) -> List[str]:
        import re
        raw: List[str] = []
        raw.extend(re.findall(r"`([^`]+)`", text or ""))
        raw.extend(re.findall(r"\"([^\"]+)\"", text or ""))
        raw.extend(re.findall(r"'([^']+)'", text or ""))
        raw.extend((text or "").split())

        cleaned: List[str] = []
        for s in raw:
            v = (s or "").strip()
            v = v.strip(" \t\r\n'\"`")
            v = v.strip("()[]{}<>")
            v = v.rstrip("?.!,;:")
            if v:
                cleaned.append(v)
        return cleaned

    def _find_existing_file(
        text: str,
        *,
        allowed_roots: List[Path],
        must_suffix: Optional[str] = None,
    ) -> Optional[Path]:
        suffix = (must_suffix or "").lower().strip()
        for cand in _candidate_path_strings(text):
            p = Path(cand)
            if suffix and p.suffix.lower() != suffix:
                continue
            if p.is_absolute():
                for root in allowed_roots:
                    safe = _safe_resolve_path(root, p)
                    if safe and safe.exists() and safe.is_file():
                        return safe
                continue

            for root in allowed_roots:
                safe = _safe_resolve_path(root, root / p)
                if safe and safe.exists() and safe.is_file():
                    return safe
        return None

    def _looks_like_file_token(token: str) -> bool:
        t = (token or "").strip()
        if not t:
            return False
        if "/" in t or "\\" in t:
            return True
        if t.startswith(".") and len(t) > 1:
            return True
        suffix = Path(t).suffix.lower()
        return suffix in {
            ".py",
            ".json",
            ".md",
            ".txt",
            ".html",
            ".csv",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".log",
        }

    def _read_text_preview(path: Path, *, limit_chars: int = 12000) -> str:
        try:
            data = path.read_bytes()
        except Exception as e:
            return f"[error reading {path}: {e}]"
        try:
            text = data.decode("utf-8")
        except Exception:
            text = data.decode("utf-8", errors="replace")
        if len(text) <= limit_chars:
            return text
        return text[:limit_chars] + "\n\n[truncated]"

    def _rag_search(q: str, *, limit: int) -> List[Dict[str, Any]]:
        q = (q or "").strip()
        if not q:
            return []
        limit_i = max(1, min(int(limit or 5), 20))
        try:
            from ..memory.rag import NexusRAG
            rag = NexusRAG(artifacts_dir)
            if not rag.index_path.exists():
                rag.rebuild()
            return rag.search(q, top_k=limit_i)
        except Exception:
            return []

    # ── API routes ─────────────────────────────────────────────────────────

    @app.get("/api/metrics")
    def api_metrics() -> JSONResponse:
        data = _latest_metrics(artifacts_dir)
        return JSONResponse(data or {})

    @app.get("/api/equity")
    def api_equity() -> JSONResponse:
        result = _latest_run_result(artifacts_dir)
        if not result:
            return JSONResponse({"equity_curve": [], "returns": []})
        eq = result.get("equity_curve") or []
        ret = result.get("returns") or []
        # Downsample if > 2000 points
        step = max(1, len(eq) // 2000)
        return JSONResponse({
            "equity_curve": eq[::step],
            "returns": ret[::step],
            "n_bars": len(eq),
        })

    @app.get("/api/ledger")
    def api_ledger() -> JSONResponse:
        events = _read_jsonl_tail(artifacts_dir / "ledger" / "ledger.jsonl", n=30)
        # Strip large fields for API response
        slim = []
        for e in reversed(events):
            p = e.get("payload") or {}
            slim.append({
                "ts": e.get("ts"),
                "kind": e.get("kind"),
                "run_name": e.get("run_name"),
                "run_id": (e.get("run_id") or "")[:8],
                "verdict": (p.get("verdict") or {}).get("pass"),
                "metrics": p.get("metrics") or {},
                "accepted": p.get("accepted"),
            })
        return JSONResponse(slim)

    @app.get("/api/wisdom")
    def api_wisdom() -> JSONResponse:
        data = _read_json(artifacts_dir / "wisdom" / "latest.json")
        return JSONResponse(data or {})

    @app.get("/api/walk_forward")
    def api_walk_forward() -> JSONResponse:
        m = _latest_metrics(artifacts_dir)
        if not m:
            return JSONResponse({"windows": []})
        wf = m.get("walk_forward") or {}
        windows = wf.get("windows") or []
        return JSONResponse({"windows": windows, "stability": wf.get("stability") or {}})

    @app.get("/api/risk")
    def api_risk() -> JSONResponse:
        result = _latest_run_result(artifacts_dir)
        if not result:
            return JSONResponse({})
        returns = result.get("returns") or []
        if not returns:
            return JSONResponse({})
        try:
            from ..risk.var import var_report
            from ..risk.regime import detect_regime
            var = var_report(returns, result.get("equity_curve") or [])
            regime = detect_regime(returns)
            # Remove non-serialisable worst_5_periods tuples
            var["worst_5_periods"] = [
                {"bar": int(i), "return": round(float(r), 6)}
                for i, r in (var.get("worst_5_periods") or [])
            ]
            return JSONResponse({"var": var, "regime": regime})
        except Exception as e:
            return JSONResponse({"error": str(e)})

    @app.get("/api/memory")
    def api_memory(q: str = "", limit: int = 20) -> JSONResponse:
        try:
            from ..memory.store import MemoryStore
            db = artifacts_dir / "memory" / "memory.db"
            if not db.exists():
                return JSONResponse([])
            ms = MemoryStore(db)
            try:
                if q:
                    items = ms.search(query=q, limit=limit)
                else:
                    items = ms.recent(limit=limit)
                return JSONResponse([
                    {"id": it.id, "ts": it.created_at, "kind": it.kind,
                     "tags": it.tags, "content": it.content[:300]}
                    for it in items
                ])
            finally:
                ms.close()
        except Exception as e:
            return JSONResponse({"error": str(e)})

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        """Quick health check + summary."""
        ledger_events = _read_jsonl_tail(artifacts_dir / "ledger" / "ledger.jsonl", n=200)
        runs = [e for e in ledger_events if e.get("kind") == "run"]
        learns = [e for e in ledger_events if e.get("kind") == "self_learn"]
        accepted = [e for e in learns if (e.get("payload") or {}).get("accepted")]
        last_run = runs[-1] if runs else {}
        last_metrics = ((last_run.get("payload") or {}).get("metrics") or {})
        return JSONResponse({
            "total_runs": len(runs),
            "total_learns": len(learns),
            "total_accepted": len(accepted),
            "accept_rate": round(len(accepted) / max(1, len(learns)), 4),
            "last_run_name": last_run.get("run_name"),
            "last_sharpe": last_metrics.get("sharpe"),
            "last_calmar": last_metrics.get("calmar"),
            "last_mdd": last_metrics.get("max_drawdown"),
            "last_verdict": ((last_run.get("payload") or {}).get("verdict") or {}).get("pass"),
        })

    # ── SSE live-update stream ─────────────────────────────────────────────

    @app.get("/api/stream")
    def api_stream():
        """Server-Sent Events endpoint: pushes a live snapshot every 5 seconds."""
        from fastapi.responses import StreamingResponse

        def _snapshot() -> dict:
            result = _latest_run_result(artifacts_dir)
            metrics = _latest_metrics(artifacts_dir)
            eq = (result or {}).get("equity_curve") or []
            ret = (result or {}).get("returns") or []
            # Latest equity point
            latest_eq = eq[-1] if eq else None
            equity_tail = eq[-20:] if eq else []
            # Latest metrics
            sharpe = (metrics or {}).get("sharpe")
            mdd = (metrics or {}).get("max_drawdown")
            # Recent ledger events (last 20)
            ledger_events = _read_jsonl_tail(
                artifacts_dir / "ledger" / "ledger.jsonl", n=20
            )
            slim_events = []
            for e in reversed(ledger_events):
                p = e.get("payload") or {}
                slim_events.append({
                    "ts": e.get("ts"),
                    "kind": e.get("kind"),
                    "run_name": e.get("run_name"),
                    "verdict": (p.get("verdict") or {}).get("pass"),
                })
            return {
                "ts": time.time(),
                "latest_equity": latest_eq,
                "equity_tail": equity_tail,
                "sharpe": sharpe,
                "mdd": mdd,
                "ledger_tail": slim_events,
            }

        def _generator():
            while True:
                try:
                    data = json.dumps(_snapshot())
                    yield f"data: {data}\n\n"
                except Exception as exc:
                    err_msg = json.dumps({"error": str(exc)})
                    yield f"data: {err_msg}\n\n"
                time.sleep(5)

        return StreamingResponse(
            _generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── All run results for strategy comparison ────────────────────────────

    @app.get("/api/runs")
    def api_runs() -> JSONResponse:
        """List the 20 most recent run results, sorted by modification time."""
        runs_dir = artifacts_dir / "runs"
        if not runs_dir.exists():
            return JSONResponse([])
        dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )[:20]
        results = []
        for d in dirs:
            m = _read_json(d / "metrics.json")
            if m is None:
                m = {}
            r = _read_json(d / "result.json")
            if r is None:
                r = {}
            verdict_obj = m.get("verdict") or r.get("verdict") or {}
            verdict_pass = verdict_obj.get("pass") if isinstance(verdict_obj, dict) else None
            results.append({
                "run_name": d.name,
                "strategy": m.get("strategy") or r.get("strategy"),
                "sharpe": m.get("sharpe"),
                "calmar": m.get("calmar"),
                "mdd": m.get("max_drawdown"),
                "cagr": m.get("cagr"),
                "win_rate": m.get("win_rate"),
                "verdict": verdict_pass,
                "ts": d.stat().st_mtime,
            })
        return JSONResponse(results)

    # ── Anomaly events from ledger ─────────────────────────────────────────

    @app.get("/api/anomalies")
    def api_anomalies() -> JSONResponse:
        """Return detected anomaly events from the last 100 ledger entries."""
        ALERT_FLAGS = {
            "slippage_alert", "drawdown_alert", "drift_alert",
            "funding_alert", "regime_alert",
        }
        events = _read_jsonl_tail(artifacts_dir / "ledger" / "ledger.jsonl", n=100)
        anomalies = []
        for e in events:
            kind = e.get("kind", "")
            payload = e.get("payload") or {}
            is_anomaly = kind == "anomaly"
            has_alert = any(payload.get(flag) for flag in ALERT_FLAGS)
            if is_anomaly or has_alert:
                anomalies.append({
                    "ts": e.get("ts"),
                    "kind": kind,
                    "run_name": e.get("run_name"),
                    "payload": payload,
                })
        return JSONResponse({"anomalies": anomalies, "count": len(anomalies)})

    # ── Brain / identity / notification endpoints ──────────────────────────

    @app.get("/api/brain/diary")
    def api_brain_diary() -> JSONResponse:
        """Latest diary entry."""
        p = artifacts_dir / "brain" / "diary" / "latest.json"
        if p.exists():
            try:
                return JSONResponse(json.loads(p.read_text("utf-8")))
            except Exception:
                pass
        return JSONResponse({})

    @app.get("/api/brain/goals")
    def api_brain_goals() -> JSONResponse:
        """All goals with progress."""
        p = artifacts_dir / "brain" / "goals.json"
        if p.exists():
            try:
                goals = json.loads(p.read_text("utf-8"))
                return JSONResponse(goals)
            except Exception:
                pass
        return JSONResponse([])

    @app.get("/api/brain/identity")
    def api_brain_identity() -> JSONResponse:
        """NEXUS identity state."""
        p = artifacts_dir / "brain" / "identity.json"
        if p.exists():
            try:
                return JSONResponse(json.loads(p.read_text("utf-8")))
            except Exception:
                pass
        return JSONResponse({"name": "NEXUS", "mood": "initializing"})

    @app.get("/api/brain/notifications")
    def api_brain_notifications() -> JSONResponse:
        """Recent NEXUS notifications."""
        p = artifacts_dir / "brain" / "notifications.jsonl"
        if not p.exists():
            return JSONResponse([])
        lines = p.read_text("utf-8").splitlines()[-20:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return JSONResponse(list(reversed(out)))

    @app.get("/api/brain/search")
    def api_brain_search(q: str = "", limit: int = 5) -> JSONResponse:
        """RAG search over memory + diary."""
        hits = _rag_search(q, limit=limit)
        out = []
        for h in hits:
            content = str(h.get("content") or "")
            out.append(
                {
                    "doc_id": str(h.get("doc_id") or ""),
                    "score": float(h.get("score") or 0.0),
                    "content_preview": content[:300],
                    "source": str(h.get("source") or ""),
                    "date": str(h.get("date") or h.get("created_at") or ""),
                }
            )
        return JSONResponse(out)

    @app.post("/api/chat")
    async def api_chat(body: ChatRequest) -> JSONResponse:
        """Chat with NEXUS. POST body: {"message": "..."}"""
        import datetime as _dt
        try:
            user_msg = str(body.message or "")[:1000]
            if not user_msg:
                return JSONResponse({"response": "Please ask me something."})

            rag_hits = _rag_search(user_msg, limit=3)
            rag_ctx_lines: List[str] = []
            for i, h in enumerate(rag_hits[:3], start=1):
                date = str(h.get("date") or h.get("created_at") or "")
                src = str(h.get("source") or "")
                doc_id = str(h.get("doc_id") or "")
                score = float(h.get("score") or 0.0)
                snippet = str(h.get("content") or "")[:500].replace("\n", " ").strip()
                rag_ctx_lines.append(f"[{i}] {doc_id} ({src} {date}) score={score:.4f}: {snippet}")
            rag_ctx = "\n".join(rag_ctx_lines).strip()

            low = user_msg.lower()
            if "what are my goals" in low:
                try:
                    from ..brain.goals import GoalTracker
                    gt = GoalTracker(artifacts_dir)
                    goals = [g.to_dict() for g in gt.all_goals()]
                    active = [g for g in goals if str(g.get("status") or "").lower() == "active"]
                    if not goals:
                        resp = "No goals found."
                    elif not active:
                        resp = "No active goals.\n\n" + "\n".join(f"- {g.get('title','')}".rstrip() for g in goals[:10])
                    else:
                        resp = "Active goals:\n" + "\n".join(f"- {g.get('title','')}".rstrip() for g in active[:10])
                    return JSONResponse(
                        {
                            "response": resp,
                            "goals": goals,
                            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                        }
                    )
                except Exception as e:
                    return JSONResponse(
                        {
                            "response": f"Error loading goals: {e}",
                            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                        }
                    )

            show_tail = _extract_tail_after("show me", user_msg)
            if show_tail:
                repo_root = _repo_root()
                file_like = [c for c in _candidate_path_strings(show_tail) if _looks_like_file_token(c)]
                chosen = _find_existing_file(" ".join(file_like), allowed_roots=[repo_root, artifacts_dir])
                if file_like and chosen:
                    content = _read_text_preview(chosen)
                    return JSONResponse(
                        {
                            "response": f"File: {chosen}\n\n{content}",
                            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                        }
                    )
                if file_like and not chosen:
                    return JSONResponse(
                        {
                            "response": f"File not found or not allowed: {file_like[0]}",
                            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                        }
                    )

            # --- Live market data injection ---
            live_market: dict = {}
            market_keywords = ["funding rate", "btc", "eth", "sol", "market", "price", "binance", "funding", "basis", "perp"]
            if any(kw in low for kw in market_keywords):
                try:
                    from ..tools.web_research import fetch_binance_funding_rates, fetch_binance_market_overview
                    symbol = "ETHUSDT" if "eth" in low else "SOLUSDT" if "sol" in low else "BTCUSDT"
                    rates = fetch_binance_funding_rates(symbol, limit=5)
                    if rates:
                        live_market["funding_rates"] = rates[:3]
                        live_market["symbol"] = symbol
                    overview = fetch_binance_market_overview()
                    if overview.get("tickers"):
                        live_market["top_tickers"] = overview["tickers"][:5]
                except Exception:
                    pass

            # --- Executor: system state queries ---
            executor_result: dict = {}
            if any(kw in low for kw in ["status", "system state", "how are you doing", "mood", "nexus state", "current state"]):
                try:
                    from ..tools.executor import NexusExecutor
                    executor_result = NexusExecutor().get_system_state(str(artifacts_dir))
                except Exception:
                    pass

            # Build context
            ctx: dict = {}
            if live_market:
                ctx["live_market"] = live_market
            if executor_result:
                ctx["system_state"] = executor_result
            m = _latest_metrics(artifacts_dir)
            if m:
                ctx["metrics"] = m.get("summary") or {}

            # Load wisdom
            wp = artifacts_dir / "wisdom" / "latest.json"
            if wp.exists():
                try:
                    ctx["wisdom"] = json.loads(wp.read_text("utf-8"))
                except Exception:
                    pass

            # Load goals
            gp = artifacts_dir / "brain" / "goals.json"
            if gp.exists():
                try:
                    ctx["goals"] = [
                        g["title"]
                        for g in json.loads(gp.read_text("utf-8"))
                        if g.get("status") == "active"
                    ]
                except Exception:
                    pass

            # Load latest diary
            dp = artifacts_dir / "brain" / "diary" / "latest.json"
            if dp.exists():
                try:
                    ctx["last_diary"] = json.loads(dp.read_text("utf-8"))
                except Exception:
                    pass

            # Use reasoning engine
            try:
                from ..brain.reasoning import answer_question
                from ..brain.identity import NEXUS_PERSONA
                augmented = user_msg
                if rag_ctx:
                    augmented = f"{user_msg}\n\nRelevant context (RAG):\n{rag_ctx}"
                response = answer_question(augmented, ctx, persona=NEXUS_PERSONA)
            except Exception as e:
                response = f"I encountered an issue processing your question: {e}"

            tool_result: Optional[Dict[str, Any]] = None
            try:
                from ..run import run_one, improve_one
                if "run backtest" in low:
                    cfg_path = _find_existing_file(user_msg, allowed_roots=[_repo_root()], must_suffix=".json")
                    if cfg_path is None:
                        default_cfg = _repo_root() / "configs" / "run_synthetic_funding.json"
                        cfg_path = default_cfg if default_cfg.exists() else None
                    if cfg_path is None or not cfg_path.exists():
                        tool_result = {"tool": "run_backtest", "ok": False, "error": "No config file found"}
                    else:
                        run_id = run_one(cfg_path, artifacts_dir)
                        tool_result = {
                            "tool": "run_backtest",
                            "ok": True,
                            "run_id": run_id,
                            "config": str(cfg_path),
                        }
                elif "run experiment" in low:
                    cfg_path = _find_existing_file(user_msg, allowed_roots=[_repo_root()], must_suffix=".json")
                    if cfg_path is None:
                        default_cfg = _repo_root() / "configs" / "run_synthetic_funding.json"
                        cfg_path = default_cfg if default_cfg.exists() else None
                    if cfg_path is None or not cfg_path.exists():
                        tool_result = {"tool": "run_experiment", "ok": False, "error": "No config file found"}
                    else:
                        run_id = improve_one(cfg_path, artifacts_dir, trials=10)
                        tool_result = {
                            "tool": "run_experiment",
                            "ok": True,
                            "run_id": run_id,
                            "config": str(cfg_path),
                            "trials": 10,
                        }
            except Exception as e:
                if ("run backtest" in low) or ("run experiment" in low):
                    tool_result = {"tool": "runner", "ok": False, "error": str(e)}

            # Log to notifications
            try:
                from ..comms.notifier import NexusNotifier
                nn = NexusNotifier(artifacts_dir)
                nn.notify(
                    f"Chat: Q={user_msg[:50]}... A={response[:50]}...",
                    level="info",
                )
            except Exception:
                pass

            payload: Dict[str, Any] = {
                "response": response,
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
            if tool_result is not None:
                payload["tool_result"] = tool_result
            if live_market:
                payload["live_market"] = live_market
            if executor_result:
                payload["system_state"] = executor_result
            return JSONResponse(payload)
        except Exception as e:
            return JSONResponse({"response": f"Error: {e}"})

    @app.get("/api/alerts")
    def api_alerts(limit: int = 30) -> JSONResponse:
        """Return recent NEXUS system alerts."""
        try:
            from ..monitoring.alert_engine import AlertEngine
            engine = AlertEngine(artifacts_dir)
            alerts = engine.recent_alerts(limit=limit)
            return JSONResponse({"alerts": alerts, "total": len(alerts)})
        except Exception as e:
            return JSONResponse({"alerts": [], "error": str(e)})

    @app.post("/api/alerts/check")
    async def api_alerts_check() -> JSONResponse:
        """Run a full alert check with live market data."""
        try:
            from ..monitoring.alert_engine import AlertEngine
            from ..tools.web_research import fetch_binance_funding_rates
            market_data: Dict[str, Any] = {}
            for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                rates = fetch_binance_funding_rates(sym, limit=1)
                if rates:
                    market_data[sym] = {"latest_rate": float(rates[0].get("fundingRate", 0))}
            engine = AlertEngine(artifacts_dir)
            result = engine.run_full_check(market_data=market_data)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/market")
    def api_market() -> JSONResponse:
        """Live Binance market data: funding rates + top tickers."""
        try:
            from ..tools.web_research import fetch_binance_funding_rates, fetch_binance_market_overview
            import datetime as _dt
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
            funding = {}
            for sym in symbols:
                rates = fetch_binance_funding_rates(sym, limit=1)
                if rates:
                    r = rates[0]
                    funding[sym] = {
                        "rate": float(r.get("fundingRate", 0)),
                        "rate_pct": round(float(r.get("fundingRate", 0)) * 100, 6),
                        "time": r.get("fundingTime", ""),
                    }
            overview = fetch_binance_market_overview()
            return JSONResponse({
                "funding": funding,
                "top_tickers": (overview.get("tickers") or [])[:10],
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/research_cycle/latest")
    def api_research_cycle_latest() -> JSONResponse:
        """Return the latest research cycle report."""
        p = artifacts_dir / "research" / "latest_cycle.json"
        if not p.exists():
            return JSONResponse({"error": "No research cycle run yet"}, status_code=404)
        try:
            return JSONResponse(json.loads(p.read_text("utf-8")))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/research_cycle/run")
    async def api_research_cycle_run() -> JSONResponse:
        """Trigger a new autonomous research cycle (background)."""
        import threading, datetime as _dt
        def _bg_run():
            try:
                from ..orchestration.research_cycle import ResearchCycle
                # pick first available config
                cfg = None
                for name in ["run_synthetic_funding.json", "run_synthetic_ml_factor.json"]:
                    p = _repo_root() / "configs" / name
                    if p.exists():
                        cfg = p
                        break
                cycle = ResearchCycle(artifacts_dir=artifacts_dir, config_path=cfg)
                cycle.run()
            except Exception:
                pass
        t = threading.Thread(target=_bg_run, daemon=True)
        t.start()
        return JSONResponse({
            "ok": True,
            "message": "Research cycle started in background",
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        })

    @app.get("/api/agents/run")
    async def api_agents_run(agent: str = "atlas") -> JSONResponse:
        """Run a specific agent on-demand and return proposals."""
        try:
            # Build minimal context
            from ..agents.base import AgentContext
            metrics = {}
            runs_dir = artifacts_dir / "runs"
            if runs_dir.exists():
                dirs = sorted(runs_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
                for d in dirs[:3]:
                    mp = d / "metrics.json"
                    if mp.exists():
                        m = json.loads(mp.read_text("utf-8"))
                        metrics = m.get("summary") or {}
                        break
            wisdom = {}
            wp = artifacts_dir / "wisdom" / "latest.json"
            if wp.exists():
                wisdom = json.loads(wp.read_text("utf-8"))
            ctx = AgentContext(wisdom=wisdom, recent_metrics=metrics, regime={})

            agent_map = {
                "atlas": "AtlasAgent",
                "cipher": "CipherAgent",
                "echo": "EchoAgent",
                "flux": "FluxAgent",
            }
            agent_name = agent.lower().strip()
            if agent_name not in agent_map:
                return JSONResponse({"error": f"Unknown agent: {agent}. Options: atlas, cipher, echo, flux"}, status_code=400)

            mod_name = f"..agents.{agent_name}"
            cls_name = agent_map[agent_name]
            import importlib
            mod = importlib.import_module(mod_name, package=__name__)
            agent_cls = getattr(mod, cls_name)
            result = agent_cls().run(ctx)
            return JSONResponse({
                "agent": agent_name,
                "model_used": result.model_used,
                "fallback_used": result.fallback_used,
                "parsed": result.parsed,
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/feedback")
    async def api_feedback(body: FeedbackRequest) -> JSONResponse:
        import datetime as _dt
        try:
            msg = str(body.message or "").strip()
            tags = body.tags or []
            tags_list = [str(t) for t in tags if t is not None and str(t).strip()]
            if not msg:
                return JSONResponse({"ok": False, "error": "message is required"})

            from ..memory.store import MemoryStore
            db = artifacts_dir / "memory" / "memory.db"
            ms = MemoryStore(db)
            try:
                item_id = ms.add(
                    created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                    kind="feedback",
                    tags=tags_list,
                    content=msg,
                    meta={"source": "web", "endpoint": "/api/feedback"},
                )
            finally:
                ms.close()

            try:
                (artifacts_dir / "brain" / "rag_index.json").unlink()
            except Exception:
                pass

            return JSONResponse({"ok": True, "id": str(item_id)})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)})

    @app.get("/api/report")
    def api_report() -> HTMLResponse:
        out_path = artifacts_dir / "reports" / "latest.html"
        try:
            from ..report_cli import generate_report
            generate_report(artifacts_dir=artifacts_dir, out_path=out_path)
            html_doc = out_path.read_text("utf-8")
        except Exception as e:
            html_doc = f"<html><body><h1>Report error</h1><pre>{json.dumps({'error': str(e)})}</pre></body></html>"
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(html_doc, encoding="utf-8")
            except Exception:
                pass
        return HTMLResponse(content=html_doc)

    # ── Team Report Ingestion ──────────────────────────────────────────────

    @app.post("/api/ingest_report")
    async def api_ingest_report(request) -> JSONResponse:
        """
        Ingest a team report file (text/JSON/CSV/Markdown/PDF text).
        Analyzes with GLM-5, stores insights in NEXUS memory + RAG index.
        Multipart: field 'file' + optional 'author' + 'topic'.
        """
        import datetime as _dt
        try:
            from fastapi import UploadFile, Form
            form = await request.form()
            file_obj: UploadFile = form.get("file")
            author = str(form.get("author") or "team").strip()[:100]
            topic = str(form.get("topic") or "research").strip()[:200]

            if not file_obj:
                return JSONResponse({"ok": False, "error": "No file provided"})

            raw_bytes = await file_obj.read(512_000)  # max 500KB
            filename = str(file_obj.filename or "upload.txt")
            try:
                content = raw_bytes.decode("utf-8", errors="replace")
            except Exception:
                content = raw_bytes.decode("latin-1", errors="replace")

            # Truncate for LLM context
            excerpt = content[:8000]

            # Analyze with GLM-5
            analysis = ""
            try:
                from ..brain.reasoning import _call_llm
                system_prompt = (
                    "You are NEXUS, an elite quantitative research AI. "
                    "A team member submitted a research report. "
                    "Extract: (1) key findings, (2) trading signals or strategy ideas, "
                    "(3) risk factors, (4) recommended NEXUS experiments to run. "
                    "Be concise and structured. Reply in JSON with keys: "
                    "findings, signals, risks, experiments."
                )
                user_prompt = f"Author: {author}\nTopic: {topic}\nFilename: {filename}\n\nContent:\n{excerpt}"
                analysis = _call_llm(system_prompt, user_prompt, max_tokens=1200)
            except Exception as e:
                analysis = f"Analysis unavailable: {e}"

            # Store in memory
            from ..memory.store import MemoryStore
            db = artifacts_dir / "memory" / "memory.db"
            ms = MemoryStore(db)
            now = _dt.datetime.now(_dt.timezone.utc).isoformat()
            try:
                mem_id = ms.add(
                    created_at=now,
                    kind="team_report",
                    tags=["team_report", author, topic],
                    content=f"[Report from {author}] {topic}\n\n{excerpt[:2000]}",
                    meta={"filename": filename, "author": author, "topic": topic, "analysis": analysis},
                )
            finally:
                ms.close()

            # Invalidate RAG index so it rebuilds on next search
            try:
                (artifacts_dir / "brain" / "rag_index.json").unlink(missing_ok=True)
            except Exception:
                pass

            # Write to research inbox
            inbox = artifacts_dir / "research" / "inbox"
            inbox.mkdir(parents=True, exist_ok=True)
            safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)
            report_path = inbox / f"{now[:10]}_{safe_name}.json"
            report_path.write_text(json.dumps({
                "filename": filename, "author": author, "topic": topic,
                "received_at": now, "analysis": analysis, "excerpt": excerpt[:1000],
            }, ensure_ascii=False, indent=2), encoding="utf-8")

            return JSONResponse({
                "ok": True,
                "filename": filename,
                "author": author,
                "topic": topic,
                "mem_id": str(mem_id) if "mem_id" in dir() else None,
                "analysis": analysis,
                "saved_to": str(report_path),
            })
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)})

    @app.get("/api/team_reports")
    def api_team_reports() -> JSONResponse:
        """List ingested team reports."""
        inbox = artifacts_dir / "research" / "inbox"
        if not inbox.exists():
            return JSONResponse([])
        reports = []
        for p in sorted(inbox.iterdir(), reverse=True)[:20]:
            if p.suffix == ".json":
                try:
                    data = json.loads(p.read_text("utf-8"))
                    reports.append({
                        "filename": data.get("filename", p.name),
                        "author": data.get("author", "—"),
                        "topic": data.get("topic", "—"),
                        "received_at": data.get("received_at", ""),
                        "has_analysis": bool(data.get("analysis")),
                    })
                except Exception:
                    continue
        return JSONResponse(reports)

    # ── Kanban Task Board ──────────────────────────────────────────────────

    try:
        from pydantic import BaseModel as _BM
        class TaskCreateRequest(_BM):
            title: str = ""
            description: str = ""
            priority: str = "medium"
            assignee: str = "NEXUS"
            tags: List[str] = []
            due_date: str = ""
            created_by: str = "user"
        class TaskUpdateRequest(_BM):
            status: str = ""
            priority: str = ""
            progress: int = -1
            assignee: str = ""
            result: str = ""
            title: str = ""
            description: str = ""
    except Exception:
        TaskCreateRequest = None  # type: ignore
        TaskUpdateRequest = None  # type: ignore

    def _get_task_manager():
        from ..tasks.manager import TaskManager
        tm = TaskManager(artifacts_dir)
        tm.seed_defaults()
        return tm

    @app.get("/api/tasks")
    def api_tasks_list() -> JSONResponse:
        """Kanban board: all tasks grouped by status."""
        try:
            tm = _get_task_manager()
            cols = tm.by_status()
            return JSONResponse({
                "columns": {k: [t.to_dict() for t in v] for k, v in cols.items()},
                "summary": tm.kanban_summary(),
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/tasks")
    async def api_tasks_create(body: TaskCreateRequest) -> JSONResponse:
        """Create a new task."""
        try:
            tm = _get_task_manager()
            t = tm.create(
                title=str(body.title or "Untitled"),
                description=str(body.description or ""),
                priority=str(body.priority or "medium"),
                assignee=str(body.assignee or "NEXUS"),
                tags=[str(x) for x in (body.tags or [])],
                due_date=str(body.due_date) if body.due_date else None,
                created_by=str(body.created_by or "user"),
            )
            return JSONResponse({"ok": True, "task": t.to_dict()})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)})

    @app.patch("/api/tasks/{task_id}")
    async def api_tasks_update(task_id: str, body: TaskUpdateRequest) -> JSONResponse:
        """Update a task (status, progress, assignee, etc.)."""
        try:
            tm = _get_task_manager()
            updates: Dict[str, Any] = {}
            if body.status: updates["status"] = body.status
            if body.priority: updates["priority"] = body.priority
            if body.progress >= 0: updates["progress"] = body.progress
            if body.assignee: updates["assignee"] = body.assignee
            if body.result: updates["result"] = body.result
            if body.title: updates["title"] = body.title
            if body.description: updates["description"] = body.description
            t = tm.update(task_id, **updates)
            if not t:
                return JSONResponse({"ok": False, "error": "Task not found"}, status_code=404)
            return JSONResponse({"ok": True, "task": t.to_dict()})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)})

    @app.delete("/api/tasks/{task_id}")
    async def api_tasks_delete(task_id: str) -> JSONResponse:
        """Delete a task."""
        try:
            tm = _get_task_manager()
            ok = tm.delete(task_id)
            return JSONResponse({"ok": ok})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)})

    # ── Strategy Generator endpoints ───────────────────────────────────────

    @app.get("/api/strategy_generator/proposals")
    def api_strategy_proposals(limit: int = 10) -> JSONResponse:
        """Get recent LLM-generated strategy proposals."""
        try:
            from ..self_learn.strategy_generator import StrategyGenerator
            sg = StrategyGenerator(artifacts_dir)
            return JSONResponse(sg.recent_proposals(limit=limit))
        except Exception as e:
            return JSONResponse({"error": str(e)})

    @app.post("/api/strategy_generator/run")
    async def api_strategy_generator_run() -> JSONResponse:
        """Generate new strategy proposals via LLM and queue to Kanban."""
        import asyncio
        try:
            from ..self_learn.strategy_generator import StrategyGenerator
            sg = StrategyGenerator(artifacts_dir)
            proposals = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: sg.generate_improvement_proposals(),
            )
            task_ids = sg.queue_proposals_to_kanban(proposals, source="dashboard")
            return JSONResponse({
                "proposals": proposals,
                "tasks_created": task_ids,
                "count": len(proposals),
            })
        except Exception as e:
            return JSONResponse({"error": str(e)})

    # ── Browser Agent endpoints ─────────────────────────────────────────────

    @app.post("/api/browser/run")
    async def api_browser_run(request: Request) -> JSONResponse:
        """Run a browser task using Playwright agent."""
        import asyncio
        try:
            body = await request.json()
        except Exception:
            body = {}
        url = body.get("url") or "https://fapi.binance.com/fapi/v1/ticker/24hr"
        task = body.get("task") or "fetch page content"
        try:
            from ..tools.playwright_agent import PlaywrightAgent
            agent = PlaywrightAgent(artifacts_dir)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: agent.run(url=url, task=task),
            )
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e), "url": url})

    @app.get("/api/browser/history")
    def api_browser_history(limit: int = 10) -> JSONResponse:
        """Get recent browser agent runs."""
        try:
            from ..tools.playwright_agent import PlaywrightAgent
            agent = PlaywrightAgent(artifacts_dir)
            return JSONResponse(agent.recent_runs(limit=limit))
        except Exception as e:
            return JSONResponse([])

    # ── Serve static dashboard ─────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def root() -> HTMLResponse:
        html_path = static_dir / "index.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>NEXUS Dashboard</h1><p>Static files not found.</p>")

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    print(f"[NEXUS Dashboard] http://{host}:{port}  artifacts={artifacts_dir}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
