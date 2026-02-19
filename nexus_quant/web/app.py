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
            return JSONResponse(payload)
        except Exception as e:
            return JSONResponse({"response": f"Error: {e}"})

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
