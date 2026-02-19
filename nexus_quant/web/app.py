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
except ImportError:
    ChatRequest = None  # type: ignore


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

    @app.post("/api/chat")
    async def api_chat(body: ChatRequest) -> JSONResponse:
        """Chat with NEXUS. POST body: {"message": "..."}"""
        import datetime as _dt
        try:
            user_msg = str(body.message or "")[:1000]
            if not user_msg:
                return JSONResponse({"response": "Please ask me something."})

            # Build context
            ctx: dict = {}
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
                response = answer_question(user_msg, ctx, persona=NEXUS_PERSONA)
            except Exception as e:
                response = f"I encountered an issue processing your question: {e}"

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

            return JSONResponse({
                "response": response,
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            })
        except Exception as e:
            return JSONResponse({"response": f"Error: {e}"})

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
