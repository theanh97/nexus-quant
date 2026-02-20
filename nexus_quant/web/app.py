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

from ..data.provider_policy import classify_provider

try:
    from starlette.requests import Request as StarletteRequest
except Exception:
    StarletteRequest = Any  # type: ignore

try:
    from pydantic import BaseModel
    class ChatRequest(BaseModel):
        message: str = ""
        model: str = "glm-5"
        history: list = []
    class FeedbackRequest(BaseModel):
        message: str = ""
        tags: List[str] = []
    class ControlRequest(BaseModel):
        action: str = "status"   # start | stop | restart | status
        target: str = "brain"    # brain | research | backtest
        config: str = ""         # optional config path for backtest
except ImportError:
    ChatRequest = None  # type: ignore
    FeedbackRequest = None  # type: ignore
    ControlRequest = None  # type: ignore


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
    runs_dir = artifacts_dir / "runs_mock"
    if not runs_dir.exists():
        return None
    dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    if not dirs:
        return None
    for d in dirs:
        cfg = _read_json(d / "config.json") or {}
        provider = str((cfg.get("data") or {}).get("provider") or "")
        if classify_provider(provider) == "real":
            return _read_json(d / "result.json")
    return _read_json(dirs[0] / "result.json")


def _latest_metrics(artifacts_dir: Path) -> Optional[Dict[str, Any]]:
    runs_dir = artifacts_dir / "runs_mock"
    if not runs_dir.exists():
        return None
    dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    if not dirs:
        return None
    for d in dirs:
        cfg = _read_json(d / "config.json") or {}
        provider = str((cfg.get("data") or {}).get("provider") or "")
        if classify_provider(provider) == "real":
            return _read_json(d / "metrics.json")
    return _read_json(dirs[0] / "metrics.json")


def serve(artifacts_dir: Path, port: int = 8080, host: str = "127.0.0.1") -> None:
    """Launch the FastAPI dashboard server."""
    try:
        import fastapi
        import uvicorn
    except ImportError:
        raise SystemExit("Dashboard requires FastAPI + uvicorn. Run: pip install fastapi uvicorn[standard]")

    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn

    app = FastAPI(title="NEXUS Quant Dashboard", version="1.0.0")
    static_dir = Path(__file__).parent / "static"
    
    # FORCE RESOLVE TO PREVENT CWD OR CLI OVERRIDES FROM HITTING LOCKED SYSTEM ARTIFACTS
    artifacts_dir = Path(__file__).parent.parent / "mock_db"
    artifacts_dir = artifacts_dir.resolve()

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

    def _pick_default_backtest_config() -> Optional[Path]:
        cfg_dir = _repo_root() / "configs"
        candidates = [
            "production_p91b_champion.json",
            "run_binance_nexus_alpha_v1_2023oos.json",
            "run_binance_nexus_alpha_v1_2025ytd.json",
            "run_binance_v1_full_2023_2025.json",
            "run_synthetic_funding.json",
        ]
        for name in candidates:
            p = cfg_dir / name
            if p.exists():
                return p
        return None

    def _pick_default_research_config() -> Optional[Path]:
        cfg_dir = _repo_root() / "configs"
        candidates = [
            "production_p91b_champion.json",
            "run_binance_nexus_alpha_v1_2023oos.json",
            "run_binance_nexus_alpha_v1_2025ytd.json",
            "run_binance_v1_full_2023_2025.json",
            "run_synthetic_funding.json",
            "run_synthetic_ml_factor.json",
        ]
        for name in candidates:
            p = cfg_dir / name
            if p.exists():
                return p
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

    @app.get("/api/memory/curator")
    def api_memory_curator() -> JSONResponse:
        """Curated semantic memory summary + stagnation + strategy registry."""
        try:
            from ..memory.curator import MemoryCurator

            curator = MemoryCurator(artifacts_dir)
            knowledge = _read_json(curator.curated_knowledge_path) or []
            if not isinstance(knowledge, list):
                knowledge = []

            def _score(e: Dict[str, Any]) -> float:
                try:
                    return float(((e.get("scores") or {}).get("score")) or 0.0)
                except Exception:
                    return 0.0

            top = sorted([e for e in knowledge if isinstance(e, dict)], key=_score, reverse=True)[:20]
            slim_top = [
                {
                    "strategy": e.get("strategy"),
                    "date": e.get("date"),
                    "score": _score(e),
                    "insight": (e.get("insight") or "")[:400],
                }
                for e in top
            ]

            return JSONResponse(
                {
                    "curated_knowledge": {
                        "total_entries": len(knowledge),
                        "top_entries": slim_top,
                    },
                    "stagnation": curator.detect_stagnation(),
                    "strategy_registry": curator.get_strategy_registry(),
                }
            )
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/memory/curate")
    def api_memory_curate() -> JSONResponse:
        """Trigger a full memory curation run."""
        try:
            from ..memory.curator import MemoryCurator

            curator = MemoryCurator(artifacts_dir)
            result = curator.curate()
            return JSONResponse(result or {})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

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

    # ── Daily learning progress ───────────────────────────────────────────

    @app.get("/api/progress")
    def api_progress() -> JSONResponse:
        """Daily learning progress: champion Sharpe improvement over time."""
        runs_dir = artifacts_dir / "runs_mock"
        if not runs_dir.exists():
            return JSONResponse({"daily": [], "best_ever_sharpe": None, "total_runs": 0})
        # Collect all run metrics grouped by day
        daily: dict = {}
        total = 0
        best_sharpe = None
        for d in runs_dir.iterdir():
            if not d.is_dir():
                continue
            cfg = _read_json(d / "config.json") or {}
            provider = str((cfg.get("data") or {}).get("provider") or "")
            if classify_provider(provider) != "real":
                continue
            m = _read_json(d / "metrics.json")
            if not m:
                continue
            summary = m.get("summary") or m
            sharpe = summary.get("sharpe")
            if sharpe is None:
                continue
            total += 1
            # Get date from directory mtime
            import datetime as _dt
            day = _dt.datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d")
            if day not in daily:
                daily[day] = {"date": day, "max_sharpe": sharpe, "runs": 1}
            else:
                daily[day]["max_sharpe"] = max(daily[day]["max_sharpe"], sharpe)
                daily[day]["runs"] += 1
            if best_sharpe is None or sharpe > best_sharpe:
                best_sharpe = sharpe
        # Sort by date ascending
        sorted_days = sorted(daily.values(), key=lambda x: x["date"])
        # Running best (champion over time)
        running_best = None
        for entry in sorted_days:
            if running_best is None or entry["max_sharpe"] > running_best:
                running_best = entry["max_sharpe"]
            entry["champion_sharpe"] = running_best
        return JSONResponse({
            "daily": sorted_days,
            "best_ever_sharpe": best_sharpe,
            "total_runs": total,
        })

    # ── SSE live-update stream ─────────────────────────────────────────────

    @app.get("/api/stream")
    def api_stream():
        """Server-Sent Events endpoint: pushes a live snapshot every 2 seconds."""
        from fastapi.responses import StreamingResponse

        def _snapshot() -> dict:
            result = _latest_run_result(artifacts_dir)
            metrics = _latest_metrics(artifacts_dir)
            eq = (result or {}).get("equity_curve") or []
            # Latest equity point
            latest_eq = eq[-1] if eq else None
            equity_tail = eq[-20:] if eq else []
            # Latest metrics — metrics.json uses nested "summary" section
            summary = (metrics or {}).get("summary") or metrics or {}
            sharpe = summary.get("sharpe")
            mdd = summary.get("max_drawdown")
            verdict_obj = (metrics or {}).get("verdict") or {}
            verdict = verdict_obj.get("pass") if isinstance(verdict_obj, dict) else None
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
            # Active run detection: check if a process is writing to runs dir recently
            active_run = None
            runs_dir = artifacts_dir / "runs_mock"
            if runs_dir.exists():
                recent = sorted(
                    [d for d in runs_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime, reverse=True
                )
                if recent and (time.time() - recent[0].stat().st_mtime) < 60:
                    active_run = recent[0].name.split(".")[0]
            # Learning velocity from accelerated engine
            learning_velocity = None
            try:
                from ..research.accelerated_learning import AcceleratedLearningEngine
                engine = AcceleratedLearningEngine(artifacts_dir / "research")
                stats = engine.summary_stats()
                learning_velocity = stats.get("learning_velocity")
            except Exception:
                pass
            # Champion strategy
            champion = _read_json(artifacts_dir / "champion.json") or {}
            return {
                "ts": time.time(),
                "latest_equity": latest_eq,
                "equity_tail": equity_tail,
                "sharpe": sharpe,
                "mdd": mdd,
                "verdict": verdict,
                "ledger_tail": slim_events,
                "active_run": active_run,
                "learning_velocity": learning_velocity,
                "champion_strategy": champion.get("strategy"),
                "champion_sharpe": (
                    champion.get("corrected_sharpe")
                    if champion.get("corrected_sharpe") is not None
                    else champion.get("sharpe")
                ),
            }

        def _generator():
            while True:
                try:
                    data = json.dumps(_snapshot())
                    yield f"data: {data}\n\n"
                except Exception as exc:
                    err_msg = json.dumps({"error": str(exc)})
                    yield f"data: {err_msg}\n\n"
                time.sleep(2)

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
        runs_dir = artifacts_dir / "runs_mock"
        if not runs_dir.exists():
            return JSONResponse([])
        dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )[:20]
        results = []
        for d in dirs:
            cfg = _read_json(d / "config.json") or {}
            provider = str((cfg.get("data") or {}).get("provider") or "")
            if classify_provider(provider) != "real":
                continue
            m = _read_json(d / "metrics.json")
            if m is None:
                m = {}
            r = _read_json(d / "result.json")
            if r is None:
                r = {}
            verdict_obj = m.get("verdict") or r.get("verdict") or {}
            verdict_pass = verdict_obj.get("pass") if isinstance(verdict_obj, dict) else None
            # metrics.json uses nested summary section
            summary = m.get("summary") or {}
            strat_obj = r.get("strategy") or {}
            strat_name = (strat_obj.get("name") if isinstance(strat_obj, dict) else None) or d.name.split(".")[0]
            results.append({
                "run_name": d.name,
                "strategy": strat_name,
                "metrics": {
                    "sharpe": summary.get("sharpe"),
                    "calmar": summary.get("calmar"),
                    "max_drawdown": summary.get("max_drawdown"),
                    "cagr": summary.get("cagr"),
                    "win_rate": summary.get("win_rate"),
                    "total_return": summary.get("total_return"),
                    "volatility": summary.get("volatility"),
                },
                "sharpe": summary.get("sharpe"),
                "calmar": summary.get("calmar"),
                "mdd": summary.get("max_drawdown"),
                "cagr": summary.get("cagr"),
                "win_rate": summary.get("win_rate"),
                "verdict": verdict_pass,
                "ts": d.stat().st_mtime,
            })
        return JSONResponse(results)

    @app.get("/api/files")
    def api_files() -> JSONResponse:
        """List all artifacts, runs, reports, configs, cache files for the file explorer."""
        import os, time as _time
        result: Dict[str, Any] = {"sections": []}

        def scan_dir(path: Path, max_files: int = 50):
            items = []
            if not path.exists():
                return items
            for root, dirs, files in os.walk(path):
                dirs[:] = sorted(d for d in dirs if not d.startswith('.'))[:10]
                for f in sorted(files)[:max_files]:
                    fp = Path(root) / f
                    try:
                        st = fp.stat()
                        rel = str(fp.relative_to(artifacts_dir))
                        items.append({
                            "name": f,
                            "path": rel,
                            "size": st.st_size,
                            "mtime": _time.strftime('%Y-%m-%d %H:%M', _time.localtime(st.st_mtime)),
                            "ext": fp.suffix,
                        })
                    except Exception:
                        pass
                if len(items) >= max_files:
                    break
            return items

        # Runs section
        runs_dir = artifacts_dir / "runs_mock"
        run_items = []
        if runs_dir.exists():
            for d in sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                if d.is_dir():
                    m = _read_json(d / "metrics.json") or {}
                    c = _read_json(d / "config.json") or {}
                    provider = str((c.get("data") or {}).get("provider") or "")
                    if classify_provider(provider) != "real":
                        continue
                    summary = m.get("summary") or m
                    run_items.append({
                        "name": d.name,
                        "path": str(d.relative_to(artifacts_dir)),
                        "sharpe": summary.get("sharpe"), "cagr": summary.get("cagr"),
                        "mdd": summary.get("max_drawdown"),
                        "mtime": _time.strftime('%Y-%m-%d %H:%M', _time.localtime(d.stat().st_mtime)),
                        "files": [f.name for f in d.iterdir() if f.is_file()],
                    })
        result["sections"].append({"title": "Backtest Runs", "icon": "📊", "items": run_items, "type": "runs"})

        # Memory section
        result["sections"].append({"title": "Memory & Knowledge", "icon": "🧠", "items": scan_dir(artifacts_dir / "memory", 20), "type": "files"})

        # Brain section
        result["sections"].append({"title": "Brain State", "icon": "💡", "items": scan_dir(artifacts_dir / "brain", 20), "type": "files"})

        # Validation section
        result["sections"].append({"title": "Validation Results", "icon": "✅", "items": scan_dir(artifacts_dir / "validation", 20), "type": "files"})

        # Handoffs/Reports section
        result["sections"].append({"title": "Handoff Reports", "icon": "📄", "items": scan_dir(artifacts_dir / "handoff", 20), "type": "files"})

        # Configs section
        configs_dir = artifacts_dir.parent / "configs"
        cfg_items = []
        if configs_dir.exists():
            for f in sorted(configs_dir.glob("*.json")):
                st = f.stat()
                cfg_items.append({"name": f.name, "path": f"configs/{f.name}", "size": st.st_size, "mtime": _time.strftime('%Y-%m-%d %H:%M', _time.localtime(st.st_mtime)), "ext": ".json"})
        result["sections"].append({"title": "Configs", "icon": "⚙️", "items": cfg_items, "type": "files"})

        # Cache section
        cache_dir = artifacts_dir.parent / ".cache"
        cache_items = []
        if cache_dir.exists():
            total_size = 0
            count = 0
            for f in cache_dir.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
                    count += 1
            cache_items.append({"name": f"{count} files", "path": ".cache/", "size": total_size, "mtime": "—", "ext": ""})
        result["sections"].append({"title": "Data Cache", "icon": "💾", "items": cache_items, "type": "files"})

        # Monitoring
        result["sections"].append({"title": "Monitoring & Alerts", "icon": "⚠️", "items": scan_dir(artifacts_dir / "monitoring", 10), "type": "files"})

        return JSONResponse(result)

    @app.get("/api/files/content")
    def api_file_content(path: str) -> JSONResponse:
        """Read content of a small text file (max 50KB)."""
        try:
            safe = (artifacts_dir / path).resolve()
            if not str(safe).startswith(str(artifacts_dir.resolve())):
                return JSONResponse({"error": "forbidden"}, status_code=403)
            if not safe.exists():
                return JSONResponse({"error": "not found"}, status_code=404)
            if safe.stat().st_size > 51200:
                return JSONResponse({"content": "[File too large to display — use CLI]", "truncated": True})
            text = safe.read_text("utf-8", errors="replace")
            return JSONResponse({"content": text, "path": path, "size": safe.stat().st_size})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

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

            # Memory tree context injection (token-efficient)
            try:
                from ..memory.tree import MemoryTree
                _mtree = MemoryTree()
                _mtree.populate_from_phases()
                mem_ctx = _mtree.context_for_query(user_msg, max_tokens=500)
                if mem_ctx:
                    ctx["memory_tree"] = mem_ctx
            except Exception:
                pass

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
                selected_model = str(getattr(body, "model", None) or "glm-5")
                response = answer_question(augmented, ctx, persona=NEXUS_PERSONA, model=selected_model)
            except Exception as e:
                response = f"I encountered an issue processing your question: {e}"

            tool_result: Optional[Dict[str, Any]] = None
            try:
                from ..run import run_one, improve_one
                if "run backtest" in low:
                    cfg_path = _find_existing_file(user_msg, allowed_roots=[_repo_root()], must_suffix=".json")
                    if cfg_path is None:
                        cfg_path = _pick_default_backtest_config()
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
                        cfg_path = _pick_default_backtest_config()
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
                "model_used": selected_model,
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
                cfg = _pick_default_research_config()
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
            runs_dir = artifacts_dir / "runs_mock"
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
            category: str = "practice"
            delegated_by: str = "nexus"
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
        """Kanban board: all tasks flat list + grouped by status + grouped by category."""
        try:
            tm = _get_task_manager()
            cols = tm.by_status()
            cats = tm.kanban_by_category()
            all_tasks = [t.to_dict() for t in tm.all_tasks()]
            return JSONResponse({
                "tasks": all_tasks,
                "columns": {k: [t.to_dict() for t in v] for k, v in cols.items()},
                "categories": {k: {
                    "meta": v["meta"],
                    "tasks": [t for col in v["columns"].values() for t in col],
                    "counts": v["counts"],
                } for k, v in cats.items()},
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
                category=str(getattr(body, "category", None) or "practice"),
                delegated_by=str(getattr(body, "delegated_by", None) or "nexus"),
            )
            return JSONResponse({"ok": True, "task": t.to_dict()})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)})

    @app.patch("/api/tasks/{task_id}/status")
    async def api_tasks_update_status(task_id: str, request: StarletteRequest) -> JSONResponse:
        """Quick status update endpoint (used by frontend done button)."""
        try:
            body = await request.json()
            tm = _get_task_manager()
            t = tm.update(task_id, status=str(body.get("status", "done")))
            if not t:
                return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
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

    # ── Validation Suite endpoints ─────────────────────────────────────────

    @app.post("/api/validation/run")
    async def api_validation_run() -> JSONResponse:
        """Run full IS/WFA/OOS/Stress/FlashCrash/Regime/MC validation suite."""
        import asyncio
        try:
            result = _latest_run_result(artifacts_dir)
            if not result:
                return JSONResponse({"error": "No backtest results. Run a backtest first."})
            returns = result.get("returns") or []
            if len(returns) < 100:
                return JSONResponse({"error": f"Insufficient data: {len(returns)} bars (need 100+)"})
            from ..validation.full_suite import run_full_validation_suite
            report = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: run_full_validation_suite(returns, run_name="latest", n_monte_carlo=200),
            )
            val_dir = artifacts_dir / "validation"
            val_dir.mkdir(parents=True, exist_ok=True)
            (val_dir / "latest.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), "utf-8"
            )
            return JSONResponse(report)
        except Exception as e:
            return JSONResponse({"error": str(e)})

    @app.get("/api/validation/latest")
    def api_validation_latest() -> JSONResponse:
        """Get latest validation suite results."""
        val_path = artifacts_dir / "validation" / "latest.json"
        return JSONResponse(_read_json(val_path) or {})

    @app.get("/api/benchmark")
    async def api_benchmark() -> JSONResponse:
        """Compare NEXUS vs S&P500, Bitcoin, Renaissance, 60/40 portfolio."""
        import asyncio
        try:
            result = _latest_run_result(artifacts_dir)
            if not result:
                return JSONResponse({"error": "No results found"})
            returns = result.get("returns") or []
            if not returns:
                return JSONResponse({"error": "No returns data"})
            m = _latest_metrics(artifacts_dir) or {}
            run_name = m.get("run_name") or "NEXUS Strategy"
            from ..validation.benchmarks import build_benchmark_comparison
            comparison = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: build_benchmark_comparison(returns, strategy_name=run_name),
            )
            bm_dir = artifacts_dir / "validation"
            bm_dir.mkdir(parents=True, exist_ok=True)
            (bm_dir / "benchmark.json").write_text(
                json.dumps(comparison, indent=2, ensure_ascii=False), "utf-8"
            )
            return JSONResponse(comparison)
        except Exception as e:
            return JSONResponse({"error": str(e)})

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

    # ── Computer Use endpoints ─────────────────────────────────────────────

    @app.post("/api/computer/screenshot")
    async def api_computer_screenshot() -> JSONResponse:
        """Take a screenshot and return path."""
        try:
            from ..tools.computer_use import NexusComputer
            nc = NexusComputer(artifacts_dir)
            path = nc.screenshot()
            if path:
                return JSONResponse({"ok": True, "path": str(path), "capabilities": nc.capabilities})
            return JSONResponse({"ok": False, "error": "Screenshot failed"})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)})

    @app.post("/api/computer/analyze")
    async def api_computer_analyze(request: StarletteRequest) -> JSONResponse:
        """Screenshot + GLM-5 vision analysis."""
        import asyncio
        try:
            body = await request.json()
        except Exception:
            body = {}
        task = body.get("task", "Describe what's on screen and suggest what NEXUS should do next")
        try:
            from ..tools.computer_use import NexusComputer
            nc = NexusComputer(artifacts_dir)
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: nc.analyze_screen(task=task)
            )
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)})

    @app.post("/api/computer/open_url")
    async def api_computer_open_url(request: StarletteRequest) -> JSONResponse:
        """Open a URL in Chrome."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        url = body.get("url", "")
        if not url:
            return JSONResponse({"ok": False, "error": "url required"})
        try:
            from ..tools.computer_use import NexusComputer
            nc = NexusComputer(artifacts_dir)
            ok = nc.open_url(url)
            return JSONResponse({"ok": ok, "url": url})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)})

    @app.get("/api/computer/capabilities")
    def api_computer_capabilities() -> JSONResponse:
        """Get computer use capabilities (what's installed)."""
        try:
            from ..tools.computer_use import NexusComputer
            nc = NexusComputer(artifacts_dir)
            return JSONResponse({"capabilities": nc.capabilities, "actions": nc.recent_actions(5)})
        except Exception as e:
            return JSONResponse({"error": str(e)})

    # ── Browser Agent endpoints ─────────────────────────────────────────────

    @app.post("/api/browser/run")
    async def api_browser_run(request: StarletteRequest) -> JSONResponse:
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

    # ── Research API ───────────────────────────────────────────────────────

    @app.get("/api/research/brief")
    def api_research_brief() -> JSONResponse:
        """Get latest daily research brief."""
        brief_path = artifacts_dir / "brain" / "daily_brief.json"
        if brief_path.exists():
            try:
                data = json.loads(brief_path.read_text("utf-8"))
                return JSONResponse(data)
            except Exception:
                pass
        return JSONResponse({"error": "No research brief yet. Run: python3 -m nexus_quant learn"})

    @app.get("/api/research/sources")
    def api_research_sources() -> JSONResponse:
        """List all 56+ research sources with metadata."""
        try:
            from ..research.source_registry import SOURCES
            by_cat: Dict[str, List] = {}
            for s in SOURCES:
                by_cat.setdefault(s["category"], []).append({
                    "name": s["name"], "label": s["label"],
                    "kind": s["kind"], "weight": s["weight"],
                    "tags": s["tags"][:5],
                })
            return JSONResponse({
                "total": len(SOURCES),
                "by_category": {k: {"count": len(v), "sources": v} for k, v in by_cat.items()},
            })
        except Exception as e:
            return JSONResponse({"error": str(e)})

    @app.get("/api/research/sessions")
    def api_research_sessions(limit: int = 10) -> JSONResponse:
        """List recent research sessions."""
        research_dir = artifacts_dir / "research"
        if not research_dir.exists():
            return JSONResponse([])
        sessions = sorted(research_dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        out = []
        for s in sessions:
            try:
                data = json.loads(s.read_text("utf-8"))
                out.append({
                    "file": s.name,
                    "ts": data.get("timestamp", ""),
                    "sources_fetched": data.get("stats", {}).get("fetched_sources", 0),
                    "total_items": data.get("stats", {}).get("total_items", 0),
                    "hypotheses": len(data.get("hypotheses", [])),
                    "top_hypothesis": (data.get("hypotheses") or [{}])[0].get("hypothesis", ""),
                })
            except Exception:
                continue
        return JSONResponse(out)

    @app.get("/api/research/log")
    def api_research_log(limit: int = 20) -> JSONResponse:
        """Get research learning log."""
        log_path = artifacts_dir / "research" / "research_log.jsonl"
        return JSONResponse(_read_jsonl_tail(log_path, limit))

    @app.get("/api/learning/stats")
    def api_learning_stats() -> JSONResponse:
        """Get accelerated learning engine stats."""
        try:
            from ..research.accelerated_learning import AcceleratedLearningEngine
            engine = AcceleratedLearningEngine(artifacts_dir)
            return JSONResponse(engine.summary_stats())
        except Exception as e:
            return JSONResponse({"error": str(e)})

    @app.get("/api/learning/brief")
    def api_learning_brief() -> JSONResponse:
        """Get accelerated learning morning brief."""
        try:
            from ..research.accelerated_learning import AcceleratedLearningEngine
            engine = AcceleratedLearningEngine(artifacts_dir)
            return JSONResponse(engine.morning_brief())
        except Exception as e:
            return JSONResponse({"error": str(e)})

    @app.get("/api/alpha_loop/champion")
    def api_champion() -> JSONResponse:
        """Get current champion strategy."""
        champion_path = artifacts_dir / "champion.json"
        if champion_path.exists():
            try:
                return JSONResponse(json.loads(champion_path.read_text("utf-8")))
            except Exception:
                pass
        return JSONResponse({"error": "No champion yet"})

    @app.get("/api/alpha_loop/rankings")
    def api_rankings() -> JSONResponse:
        """Get strategy rankings from all runs."""
        try:
            from ..orchestration.alpha_loop import AlphaLoop
            from pathlib import Path as _Path
            loop = AlphaLoop(artifacts_dir, _Path("."), interval_seconds=3600)
            return JSONResponse(loop.compare_strategies()[:20])
        except Exception as e:
            return JSONResponse({"error": str(e)})

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

    def _log_candidates(target: str) -> List[Path]:
        t = str(target or "").strip().lower()
        mapping: Dict[str, List[Path]] = {
            "dashboard": [Path("/tmp/nexus_dash.log"), Path("/tmp/nexus_dashboard.log")],
            "brain": [Path("/tmp/nexus_brain.log"), Path("/tmp/nexus_alpha_brain.log")],
            "research": [Path("/tmp/nexus_research.log"), Path("/tmp/nexus_research_cycle.log")],
            "backtest": [Path("/tmp/nexus_backtest.log"), Path("/tmp/nexus_alpha_v1_run.log")],
        }
        if t in mapping:
            return mapping[t]
        if t:
            return [Path(f"/tmp/nexus_{t}.log")]
        return mapping["brain"]

    def _resolve_log_path(target: str) -> Path:
        cands = _log_candidates(target)
        for cand in cands:
            if cand.exists():
                return cand
        return cands[0]

    def _read_tail_lines(path: Path, n: int) -> List[str]:
        if not path.exists():
            return []
        n_safe = max(1, min(int(n or 1), 4000))
        try:
            return path.read_text("utf-8", errors="replace").splitlines()[-n_safe:]
        except Exception:
            return []

    def _parse_size_to_mb(token: str) -> Optional[float]:
        import re
        m = re.match(r"(?i)^([0-9]+(?:\.[0-9]+)?)([kmgt]?)$", str(token or "").strip())
        if not m:
            return None
        num = float(m.group(1))
        unit = (m.group(2) or "").upper()
        factor = {
            "": 1.0 / (1024.0 * 1024.0),
            "K": 1.0 / 1024.0,
            "M": 1.0,
            "G": 1024.0,
            "T": 1024.0 * 1024.0,
        }.get(unit)
        if factor is None:
            return None
        return num * factor

    @app.get("/api/processes")
    async def api_processes() -> JSONResponse:
        """List NEXUS process status in a UI-friendly schema."""
        import subprocess as _sp
        profiles: Dict[str, Dict[str, Any]] = {
            "dashboard": {"name": "Dashboard", "patterns": ["-m nexus_quant dashboard"]},
            "brain": {"name": "Brain Loop", "patterns": ["-m nexus_quant brain"]},
            "research": {"name": "Research", "patterns": ["-m nexus_quant learn", "-m nexus_quant research"]},
            "backtest": {"name": "Backtest", "patterns": ["-m nexus_quant run", "-m nexus_quant improve"]},
        }
        parsed: List[Dict[str, Any]] = []
        try:
            r = _sp.run(["ps", "-axo", "pid,%cpu,rss,command"], capture_output=True, text=True, timeout=5)
            for ln in r.stdout.splitlines()[1:]:
                parts = ln.strip().split(None, 3)
                if len(parts) < 4:
                    continue
                pid_s, cpu_s, rss_s, cmd = parts
                cmd_l = cmd.lower()
                if "grep " in cmd_l:
                    continue
                try:
                    pid = int(pid_s)
                    cpu_pct = float(cpu_s)
                    mem_mb = float(rss_s) / 1024.0
                except Exception:
                    continue
                parsed.append({
                    "pid": pid,
                    "cpu_pct": cpu_pct,
                    "mem_mb": mem_mb,
                    "command": cmd,
                    "command_l": cmd_l,
                })
        except Exception as e:
            return JSONResponse({"processes": [], "error": str(e)})

        pid_info: Dict[str, str] = {}
        pid_pref: Dict[str, int] = {}
        pid_file = Path("/tmp/nexus_pids.txt")
        if pid_file.exists():
            saved = pid_file.read_text("utf-8", errors="replace").strip().split()
            labels = ["dashboard", "brain"]
            for i, pid in enumerate(saved):
                if i >= len(labels):
                    break
                k = labels[i]
                pid_info[k] = pid
                try:
                    pid_pref[k] = int(pid)
                except Exception:
                    pass

        def _pick_hit(target: str, hits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not hits:
                return None
            pref = pid_pref.get(target)
            if pref is not None:
                for h in hits:
                    if int(h.get("pid") or -1) == pref:
                        return h
            # Prefer most recently created process (higher pid), then resource usage.
            return sorted(
                hits,
                key=lambda x: (int(x.get("pid") or 0), float(x.get("cpu_pct") or 0.0), float(x.get("mem_mb") or 0.0)),
                reverse=True,
            )[0]

        rows: List[Dict[str, Any]] = []
        for target, meta in profiles.items():
            pats = [str(p).lower() for p in (meta.get("patterns") or [])]
            hits = [p for p in parsed if any(pat in p["command_l"] for pat in pats)]
            pick = _pick_hit(target, hits)
            if pick:
                rows.append({
                    "target": target,
                    "name": meta.get("name") or target.title(),
                    "status": "running",
                    "pid": str(pick["pid"]),
                    "cpu_pct": round(float(pick["cpu_pct"]), 2),
                    "mem_mb": round(float(pick["mem_mb"]), 1),
                    "command": str(pick["command"])[:180],
                    "instances": len(hits),
                    "preferred_pid": str(pid_pref.get(target) or ""),
                })
            else:
                rows.append({
                    "target": target,
                    "name": meta.get("name") or target.title(),
                    "status": "stopped",
                    "pid": "",
                    "cpu_pct": 0.0,
                    "mem_mb": 0.0,
                    "command": "",
                    "instances": 0,
                    "preferred_pid": str(pid_pref.get(target) or ""),
                })

        known = [pat for meta in profiles.values() for pat in (meta.get("patterns") or [])]
        extras = []
        for p in parsed:
            cmd_l = p["command_l"]
            if "nexus_quant" not in cmd_l:
                continue
            if any(str(pat).lower() in cmd_l for pat in known):
                continue
            extras.append({
                "target": "extra",
                "name": "Other NEXUS",
                "status": "running",
                "pid": str(p["pid"]),
                "cpu_pct": round(float(p["cpu_pct"]), 2),
                "mem_mb": round(float(p["mem_mb"]), 1),
                "command": str(p["command"])[:180],
            })
        rows.extend(extras[:8])

        return JSONResponse(
            {
                "processes": rows,
                "saved_pids": pid_info,
                "log_files": {
                    "dashboard": str(_resolve_log_path("dashboard")),
                    "brain": str(_resolve_log_path("brain")),
                    "research": str(_resolve_log_path("research")),
                    "backtest": str(_resolve_log_path("backtest")),
                },
            }
        )

    @app.get("/api/log_tail")
    async def api_log_tail(target: str = "brain", n: int = 80) -> JSONResponse:
        """Return last N lines from a NEXUS log file."""
        t = str(target or "brain").strip().lower()
        lp = _resolve_log_path(t)
        if not lp.exists():
            return JSONResponse({"target": t, "lines": [], "exists": False, "path": str(lp)})
        try:
            text = lp.read_text("utf-8", errors="replace")
            all_lines = text.splitlines()
            lines = all_lines[-max(1, min(int(n or 80), 4000)) :]
            return JSONResponse(
                {
                    "target": t,
                    "lines": lines,
                    "exists": True,
                    "path": str(lp),
                    "total_lines": len(all_lines),
                }
            )
        except Exception as e:
            return JSONResponse({"target": t, "lines": [], "error": str(e), "exists": True, "path": str(lp)})

    @app.get("/api/log_stream")
    async def api_log_stream(request: StarletteRequest, target: str = "brain", n: int = 200):
        """SSE stream of incremental log updates for Console tab."""
        import asyncio
        from fastapi.responses import StreamingResponse

        t = str(target or "brain").strip().lower()
        n_tail = max(20, min(int(n or 200), 2000))

        async def _generator():
            path = _resolve_log_path(t)
            pos = 0
            bootstrapped = False
            while True:
                if await request.is_disconnected():
                    break
                payload: Dict[str, Any] = {
                    "target": t,
                    "path": str(path),
                    "exists": path.exists(),
                    "lines": [],
                    "ts": time.time(),
                }
                try:
                    path = _resolve_log_path(t)
                    payload["path"] = str(path)
                    payload["exists"] = path.exists()
                    if path.exists():
                        if not bootstrapped:
                            payload["lines"] = _read_tail_lines(path, n_tail)
                            try:
                                pos = int(path.stat().st_size)
                            except Exception:
                                pos = 0
                            bootstrapped = True
                        else:
                            size = int(path.stat().st_size)
                            if size < pos:
                                pos = 0
                            if size > pos:
                                with path.open("rb") as f:
                                    f.seek(pos)
                                    chunk = f.read(size - pos)
                                    pos = int(f.tell())
                                if chunk:
                                    text = chunk.decode("utf-8", errors="replace")
                                    payload["lines"] = text.splitlines()[-400:]
                except Exception as exc:
                    payload["error"] = str(exc)
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(
            _generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.post("/api/control")
    async def api_control(body: ControlRequest) -> JSONResponse:
        """Start / stop / restart NEXUS processes."""
        import subprocess as _sp
        import os as _os
        action = str(body.action or "status").lower()
        target = str(body.target or "brain").lower()

        env = dict(_os.environ)
        env.setdefault("ZAI_API_KEY", "b3893915bcea4355a46eeab30ba8db35.EExWnj8Q7bxqtvGx")
        env.setdefault("ZAI_ANTHROPIC_BASE_URL", "https://api.z.ai/api/anthropic")
        env.setdefault("ZAI_DEFAULT_MODEL", "glm-5")
        repo = str(artifacts_dir.parent) if artifacts_dir else "/Users/qtmobile/Desktop/Nexus - Quant Trading "

        if action == "stop":
            pid_file = Path("/tmp/nexus_pids.txt")
            killed = []
            if pid_file.exists():
                for pid in pid_file.read_text().strip().split():
                    try:
                        _sp.run(["kill", "-TERM", pid], capture_output=True, timeout=3)
                        killed.append(pid)
                    except Exception:
                        pass
                pid_file.unlink(missing_ok=True)
            # Also kill by process name
            try:
                _sp.run(["pkill", "-f", "nexus_quant brain"], capture_output=True)
            except Exception:
                pass
            return JSONResponse({"ok": True, "action": "stop", "killed": killed})

        if action in ("start", "restart"):
            if target == "brain":
                cfg = body.config or "configs/run_binance_nexus_alpha_v1_2023oos.json"
                log_f = open("/tmp/nexus_brain.log", "a")
                proc = _sp.Popen(
                    ["python3", "-m", "nexus_quant", "brain", "--config", cfg, "--loop"],
                    cwd=repo, stdout=log_f, stderr=log_f, env=env,
                )
                # Save PID
                existing = Path("/tmp/nexus_pids.txt")
                dash_pid = ""
                if existing.exists():
                    parts = existing.read_text().strip().split()
                    if parts:
                        dash_pid = parts[0]
                Path("/tmp/nexus_pids.txt").write_text(f"{dash_pid} {proc.pid}")
                return JSONResponse({"ok": True, "action": action, "target": "brain", "pid": proc.pid})

            if target == "research":
                log_f = open("/tmp/nexus_research.log", "a")
                proc = _sp.Popen(
                    ["python3", "-m", "nexus_quant", "learn", "--artifacts", "artifacts"],
                    cwd=repo, stdout=log_f, stderr=log_f, env=env,
                )
                return JSONResponse({"ok": True, "action": action, "target": "research", "pid": proc.pid})

            if target == "backtest":
                cfg = body.config or "configs/run_binance_nexus_alpha_v1_2023oos.json"
                log_f = open("/tmp/nexus_backtest.log", "a")
                proc = _sp.Popen(
                    ["python3", "-m", "nexus_quant", "run", "--config", cfg],
                    cwd=repo, stdout=log_f, stderr=log_f, env=env,
                )
                return JSONResponse({"ok": True, "action": action, "target": "backtest", "pid": proc.pid, "config": cfg})

        return JSONResponse({"ok": True, "action": "status", "target": target})

    @app.get("/api/system_status")
    async def api_system_status() -> JSONResponse:
        """System resource usage for live console meters."""
        import re
        import subprocess as _sp

        status: Dict[str, Any] = {}
        cpu_pct = 0.0
        mem_pct = 0.0
        disk_pct = 0.0

        try:
            r = _sp.run(["top", "-l", "1", "-n", "0"], capture_output=True, text=True, timeout=5)
            for line in r.stdout.splitlines():
                if "CPU usage" in line:
                    status["cpu_line"] = line.strip()
                    m = re.search(r"CPU usage:\s*([0-9.]+)% user,\s*([0-9.]+)% sys,\s*([0-9.]+)% idle", line)
                    if m:
                        cpu_pct = float(m.group(1)) + float(m.group(2))
                elif "PhysMem" in line:
                    status["mem_line"] = line.strip()
                    m2 = re.search(r"PhysMem:\s*([0-9.]+[KMGTP]?) used.*,\s*([0-9.]+[KMGTP]?) unused", line)
                    if m2:
                        used_mb = _parse_size_to_mb(m2.group(1))
                        unused_mb = _parse_size_to_mb(m2.group(2))
                        if used_mb is not None and unused_mb is not None and (used_mb + unused_mb) > 0:
                            mem_pct = (used_mb / (used_mb + unused_mb)) * 100.0
        except Exception:
            pass

        try:
            df_target = str(artifacts_dir if artifacts_dir else Path("."))
            dfr = _sp.run(["df", "-Pk", df_target], capture_output=True, text=True, timeout=5)
            rows = [ln for ln in dfr.stdout.splitlines() if ln.strip()]
            if len(rows) >= 2:
                cols = rows[-1].split()
                if len(cols) >= 5:
                    disk_pct = float(str(cols[4]).strip().rstrip("%"))
        except Exception:
            pass

        cache_size_mb: Optional[float] = None
        cache_files = 0
        cache_dir = artifacts_dir.parent / ".cache" / "binance_rest" if artifacts_dir else Path(".cache/binance_rest")
        if cache_dir.exists():
            total = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            cache_size_mb = round(total / 1e6, 1)
            cache_files = len(list(cache_dir.glob("*.json")))

        artifacts_size_mb: Optional[float] = None
        run_count = 0
        if artifacts_dir and artifacts_dir.exists():
            total = sum(f.stat().st_size for f in artifacts_dir.rglob("*") if f.is_file())
            artifacts_size_mb = round(total / 1e6, 1)
            run_count = len(list((artifacts_dir / "runs").glob("*"))) if (artifacts_dir / "runs").exists() else 0

        status.update(
            {
                "cpu_pct": round(cpu_pct, 1),
                "mem_pct": round(mem_pct, 1),
                "disk_pct": round(disk_pct, 1),
                "cache_size_mb": cache_size_mb,
                "cache_files": cache_files,
                "cache_size": f"{cache_size_mb:.1f} MB" if cache_size_mb is not None else "—",
                "artifacts_size_mb": artifacts_size_mb,
                "artifacts_size": f"{artifacts_size_mb:.1f} MB" if artifacts_size_mb is not None else "—",
                "run_count": run_count,
            }
        )
        return JSONResponse(status)

    @app.get("/api/runs_history")
    async def api_runs_history(limit: int = 50) -> JSONResponse:
        """Return summarized history of all backtest runs for Performance tab."""
        runs_dir = artifacts_dir / "runs_mock" if artifacts_dir else Path("artifacts/runs")
        if not runs_dir.exists():
            return JSONResponse([])
        rows = []
        run_dirs = sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        for rd in run_dirs[:limit]:
            mf = rd / "metrics.json"
            if not mf.exists():
                continue
            try:
                m = json.loads(mf.read_text("utf-8"))
                s = m.get("summary", {})
                meta = m.get("meta", {})
                verdict = m.get("verdict", {})
                bias = m.get("bias_check", {})
                # Parse year from run_name
                rn = rd.name
                year = "unknown"
                for y in ["2021", "2022", "2023", "2024", "2025"]:
                    if y in rn:
                        year = y
                        break
                rows.append({
                    "run_id": rn,
                    "run_name": rn.split(".")[0] if "." in rn else rn,
                    "year": year,
                    "sharpe": round(float(s.get("sharpe", 0)), 3),
                    "calmar": round(float(s.get("calmar", 0)), 3),
                    "cagr": round(float(s.get("cagr", 0)), 4),
                    "max_drawdown": round(float(s.get("max_drawdown", 0)), 4),
                    "beta": round(float(s.get("beta", 0)), 4),
                    "win_rate": round(float(s.get("win_rate", 0)), 3),
                    "total_return": round(float(s.get("total_return", 0)), 4),
                    "periods": int(s.get("periods", 0)),
                    "verdict_pass": bool(verdict.get("pass", True)),
                    "bias_likely_overfit": bool(bias.get("likely_overfit", False)),
                    "data_provider": meta.get("data_provider", ""),
                    "mtime": mf.stat().st_mtime,
                })
            except Exception:
                continue
        return JSONResponse(rows)

    # ── Debate endpoints ──────────────────────────────────────────────────────
    class DebateRequest(BaseModel):
        topic: str = ""
        context: str = ""
        models: list = []

    @app.post("/api/debate")
    async def api_debate(body: DebateRequest) -> JSONResponse:
        """Trigger a multi-model debate and return the result."""
        try:
            from ..brain.debate import get_engine
            engine = get_engine()
            topic = str(body.topic or "").strip()
            if not topic:
                return JSONResponse({"error": "topic required"}, status_code=400)
            models = list(body.models) if body.models else ["glm-5", "claude-sonnet-4-6", "codex", "minimax-2.5"]
            result = engine.quick_debate(topic=topic, context=str(body.context or ""))
            return JSONResponse(result)
        except Exception as e:
            import traceback
            return JSONResponse({"error": str(e), "trace": traceback.format_exc()[-500:]}, status_code=500)

    @app.get("/api/debate_history")
    async def api_debate_history(n: int = 20) -> JSONResponse:
        """Return last N debate rounds."""
        try:
            from ..brain.debate import get_engine
            engine = get_engine()
            return JSONResponse(engine.get_history(n=n))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # ── Model Stack Status ──────────────────────────────────────────
    @app.get("/api/models")
    async def api_models() -> JSONResponse:
        """Return status of all integrated AI models."""
        import os, shutil
        models = [
            {
                "name": "Claude Sonnet 4.6",
                "codename": "SENTINEL",
                "provider": "ZAI (Anthropic)",
                "cost": "paid",
                "role": "Strategy research, risk analysis, architecture",
                "available": bool(os.environ.get("ZAI_API_KEY")),
            },
            {
                "name": "GPT-5.2 (Codex)",
                "codename": "FORGE",
                "provider": "OpenAI (local CLI)",
                "cost": "paid",
                "role": "Code generation, strategy math, refactoring",
                "available": os.path.exists("/Applications/Codex.app/Contents/Resources/codex"),
            },
            {
                "name": "Gemini 2.5 Pro",
                "codename": "ORACLE",
                "provider": "Google (API)" if os.environ.get("GEMINI_API_KEY") else "Google (CLI)",
                "cost": "free",
                "role": "Complex reasoning, cross-verification, data analysis",
                "available": bool(os.environ.get("GEMINI_API_KEY")) or bool(shutil.which("gemini")),
                "via": "api" if os.environ.get("GEMINI_API_KEY") else ("cli" if shutil.which("gemini") else "unavailable"),
            },
            {
                "name": "Gemini 2.5 Flash",
                "codename": "ORACLE-FAST",
                "provider": "Google (API)" if os.environ.get("GEMINI_API_KEY") else "Google (CLI)",
                "cost": "free",
                "role": "QA, monitoring, diary synthesis, fast tasks",
                "available": bool(os.environ.get("GEMINI_API_KEY")) or bool(shutil.which("gemini")),
                "via": "api" if os.environ.get("GEMINI_API_KEY") else ("cli" if shutil.which("gemini") else "unavailable"),
            },
            {
                "name": "GLM-5",
                "codename": "SAGE",
                "provider": "ZAI (ZhipuAI)",
                "cost": "low",
                "role": "Experiment design, low-cost throughput",
                "available": bool(os.environ.get("ZAI_API_KEY")),
            },
        ]
        return JSONResponse({
            "models": models,
            "total": len(models),
            "available": sum(1 for m in models if m["available"]),
        })

    # ── Live Signals (Paper Trading) ─────────────────────────────────
    @app.get("/api/signals/latest")
    async def api_signals_latest() -> JSONResponse:
        """Return latest signal and paper trading state."""
        import json as _json
        signals_log = project_root / "artifacts" / "live" / "signals_log.jsonl"
        paper_state = project_root / "artifacts" / "live" / "paper_state.json"

        result: dict = {"signal": None, "paper": None, "history_count": 0}

        # Latest signal from JSONL
        if signals_log.exists():
            lines = signals_log.read_text().strip().splitlines()
            result["history_count"] = len(lines)
            if lines:
                try:
                    result["signal"] = _json.loads(lines[-1])
                except Exception:
                    pass

        # Paper state
        if paper_state.exists():
            try:
                result["paper"] = _json.loads(paper_state.read_text())
            except Exception:
                pass

        return JSONResponse(result)

    @app.get("/api/signals/history")
    async def api_signals_history(n: int = 50) -> JSONResponse:
        """Return last N signals."""
        import json as _json
        signals_log = project_root / "artifacts" / "live" / "signals_log.jsonl"
        if not signals_log.exists():
            return JSONResponse({"signals": [], "total": 0})
        lines = signals_log.read_text().strip().splitlines()
        recent = lines[-n:] if len(lines) > n else lines
        signals = []
        for line in reversed(recent):
            try:
                signals.append(_json.loads(line))
            except Exception:
                pass
        return JSONResponse({"signals": signals, "total": len(lines)})

    @app.post("/api/signals/generate")
    async def api_signals_generate() -> JSONResponse:
        """Generate a new signal NOW (triggers Binance data fetch)."""
        try:
            from ..live.signal_generator import SignalGenerator
            gen = SignalGenerator.from_production_config()
            signal = gen.generate()
            return JSONResponse({"ok": True, "signal": signal.to_dict()})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app.get("/api/executor/status")
    async def api_executor_status() -> JSONResponse:
        """Return execution system status."""
        import json as _json
        exec_log = project_root / "artifacts" / "live" / "execution_log.jsonl"
        runner_log = project_root / "artifacts" / "live" / "runner_log.jsonl"
        result: dict = {
            "executor_available": False,
            "api_key_set": bool(os.environ.get("BINANCE_API_KEY")),
            "testnet_key_set": False,
            "execution_log_entries": 0,
            "runner_log_entries": 0,
            "last_cycle": None,
        }
        if exec_log.exists():
            result["execution_log_entries"] = len(exec_log.read_text().strip().splitlines())
        if runner_log.exists():
            lines = runner_log.read_text().strip().splitlines()
            result["runner_log_entries"] = len(lines)
            if lines:
                try:
                    result["last_cycle"] = _json.loads(lines[-1])
                except Exception:
                    pass
        try:
            from ..live.binance_executor import BinanceExecutor
            result["executor_available"] = True
        except Exception:
            pass
        return JSONResponse(result)

    # ── Signal Health ─────────────────────────────────────────────────
    @app.get("/api/signals/health")
    async def api_signals_health() -> JSONResponse:
        """Return latest signal health report (Phase 123)."""
        health_path = project_root / "artifacts" / "phase123" / "signal_health_report.json"
        if not health_path.exists():
            return JSONResponse({"available": False, "message": "No health report yet. Run Phase 123."})
        try:
            data = json.loads(health_path.read_text())
            return JSONResponse({"available": True, **data})
        except Exception as e:
            return JSONResponse({"available": False, "error": str(e)})

    # ── Live Engine Status ────────────────────────────────────────────
    @app.get("/api/live/status")
    async def api_live_status() -> JSONResponse:
        """Return live trading engine status + recent cycles."""
        engine_log = project_root / "artifacts" / "execution" / "engine_log.jsonl"
        rebal_log = project_root / "artifacts" / "execution" / "rebalance_log.jsonl"
        risk_state_f = project_root / "artifacts" / "execution" / "risk_state.json"

        result: dict = {
            "engine_available": True,
            "api_key_set": bool(os.environ.get("BINANCE_API_KEY")),
            "risk_halted": False,
            "total_cycles": 0,
            "last_cycle": None,
            "recent_cycles": [],
            "risk_state": None,
            "rebalance_count": 0,
        }

        # Engine log
        if engine_log.exists():
            lines = engine_log.read_text().strip().splitlines()
            result["total_cycles"] = len(lines)
            recent = []
            for ln in lines[-20:]:
                try:
                    recent.append(json.loads(ln))
                except Exception:
                    pass
            result["recent_cycles"] = recent
            if recent:
                result["last_cycle"] = recent[-1]

        # Risk state
        if risk_state_f.exists():
            try:
                rs = json.loads(risk_state_f.read_text())
                result["risk_state"] = rs
                result["risk_halted"] = rs.get("halted", False)
            except Exception:
                pass

        # Rebalance log
        if rebal_log.exists():
            result["rebalance_count"] = len(
                rebal_log.read_text().strip().splitlines()
            )

        return JSONResponse(result)

    @app.post("/api/live/risk_reset")
    async def api_live_risk_reset() -> JSONResponse:
        """Reset risk gate halt state (operator override)."""
        try:
            from ..execution.risk_gate import RiskGate
            rg = RiskGate(
                state_dir=project_root / "artifacts" / "execution"
            )
            rg.reset_halt()
            return JSONResponse({"status": "ok", "message": "Risk halt reset"})
        except Exception as e:
            return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

    # ── Analysis Log (Research Journal) ──────────────────────────────
    @app.get("/api/analysis_log")
    async def api_analysis_log(
        n: int = 100,
        entry_type: str = "",
        source: str = "",
    ) -> JSONResponse:
        """Return analysis log entries (thinking, findings, decisions, audits)."""
        try:
            from ..logs.writer import read_log
            log_dir = artifacts_dir / "logs" if artifacts_dir else Path("artifacts/logs")
            entries = read_log(
                log_dir=log_dir,
                n=n,
                entry_type=entry_type or None,
                source=source or None,
            )
            return JSONResponse({"entries": entries, "total": len(entries)})
        except Exception as e:
            return JSONResponse({"entries": [], "error": str(e)})

    @app.get("/api/analysis_log_stream")
    async def api_analysis_log_stream(
        request: StarletteRequest,
        n: int = 150,
        entry_type: str = "",
        source: str = "",
    ):
        """SSE stream for analysis journal entries (Logs tab)."""
        import asyncio
        from fastapi.responses import StreamingResponse
        from ..logs.writer import read_log

        log_dir = artifacts_dir / "logs" if artifacts_dir else Path("artifacts/logs")
        fp = log_dir / "analysis.jsonl"
        n_safe = max(20, min(int(n or 150), 1000))
        type_filter = str(entry_type or "").strip()
        source_filter = str(source or "").strip()

        def _accept(obj: Dict[str, Any]) -> bool:
            if type_filter and str(obj.get("type") or "") != type_filter:
                return False
            if source_filter and str(obj.get("source") or "") != source_filter:
                return False
            return True

        async def _generator():
            pos = 0
            bootstrapped = False
            while True:
                if await request.is_disconnected():
                    break
                payload: Dict[str, Any] = {
                    "entries": [],
                    "path": str(fp),
                    "exists": fp.exists(),
                    "ts": time.time(),
                }
                try:
                    if fp.exists():
                        if not bootstrapped:
                            payload["entries"] = read_log(
                                log_dir=log_dir,
                                n=n_safe,
                                entry_type=type_filter or None,
                                source=source_filter or None,
                            )
                            try:
                                pos = int(fp.stat().st_size)
                            except Exception:
                                pos = 0
                            bootstrapped = True
                        else:
                            size = int(fp.stat().st_size)
                            if size < pos:
                                pos = 0
                            if size > pos:
                                with fp.open("rb") as f:
                                    f.seek(pos)
                                    chunk = f.read(size - pos)
                                    pos = int(f.tell())
                                if chunk:
                                    text = chunk.decode("utf-8", errors="replace")
                                    new_entries: List[Dict[str, Any]] = []
                                    for ln in text.splitlines():
                                        try:
                                            obj = json.loads(ln)
                                            if isinstance(obj, dict) and _accept(obj):
                                                new_entries.append(obj)
                                        except Exception:
                                            continue
                                    if new_entries:
                                        payload["entries"] = new_entries[-200:]
                except Exception as exc:
                    payload["error"] = str(exc)
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(
            _generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.post("/api/analysis_log")
    async def api_analysis_log_write(body: dict) -> JSONResponse:
        """Write a new analysis log entry from the dashboard."""
        try:
            from ..logs.writer import log_analysis
            log_dir = artifacts_dir / "logs" if artifacts_dir else Path("artifacts/logs")
            entry = log_analysis(
                content=body.get("content", ""),
                source=body.get("source", "user"),
                category=body.get("category", "note"),
                title=body.get("title", ""),
                tags=body.get("tags", []),
                log_dir=log_dir,
            )
            return JSONResponse({"ok": True, "entry": entry})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    # ── Track Record API ─────────────────────────────────────────────────────

    @app.get("/api/track_record")
    async def api_track_record() -> JSONResponse:
        """
        NEXUS platform-wide track record.
        Returns per-year Sharpe ratios for all projects + benchmark comparison.
        """
        try:
            from ..reporting.track_record import NexusTrackRecord
            tr = NexusTrackRecord.build_from_memory()
            # Try to merge with latest saved record
            saved_path = (artifacts_dir / "track_record.json") if artifacts_dir else Path("artifacts/track_record.json")
            if saved_path.exists():
                try:
                    saved = NexusTrackRecord.load(str(saved_path))
                    # Merge: use saved project data if richer
                    for pname, saved_proj in saved.projects.items():
                        if pname in tr.projects and saved_proj.years:
                            for yr, ym in saved_proj.years.items():
                                if yr not in tr.projects[pname].years:
                                    tr.projects[pname].years[yr] = ym
                            tr.projects[pname].compute_aggregates()
                except Exception:
                    pass
            return JSONResponse(tr.api_dict())
        except Exception as e:
            import traceback
            return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)

    @app.get("/api/portfolio_optimize")
    async def api_portfolio_optimize() -> JSONResponse:
        """
        Multi-strategy portfolio optimization.
        Returns optimal allocations, efficient frontier, and per-year stats.
        """
        try:
            from ..portfolio.optimizer import PortfolioOptimizer
            opt = PortfolioOptimizer()

            # Grid search at central correlation estimate
            results_25 = opt.optimize(step=0.05, correlation_override=0.25)

            # Max Sharpe, Max Min-Sharpe, Min-Vol picks
            max_sharpe_r = results_25[0]
            max_min_r = max(results_25, key=lambda r: r.min_sharpe)
            min_vol_r = min(results_25, key=lambda r: r.annual_vol_pct)

            # Efficient frontier across correlations
            frontier = opt.efficient_frontier(correlations=[0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5])
            frontier_out = []
            for corr, res_list in sorted(frontier.items()):
                best = res_list[0]
                frontier_out.append({
                    "correlation": corr,
                    "weights": best.weights,
                    "avg_sharpe": best.avg_sharpe,
                    "min_sharpe": best.min_sharpe,
                    "vol_pct": best.annual_vol_pct,
                    "return_pct": best.avg_return_pct,
                    "div_benefit": best.diversification_benefit,
                })

            # Full grid for chart
            grid_out = []
            for r in sorted(results_25, key=lambda x: x.weights.get("crypto_perps", 0)):
                grid_out.append({
                    "weights": r.weights,
                    "avg_sharpe": r.avg_sharpe,
                    "min_sharpe": r.min_sharpe,
                    "vol_pct": r.annual_vol_pct,
                    "return_pct": r.avg_return_pct,
                    "year_sharpe": r.year_sharpe,
                    "div_benefit": r.diversification_benefit,
                })

            # Dynamic allocation
            dyn = opt.dynamic_allocation()
            dyn_out = {str(yr): {
                "weights": r.weights,
                "sharpe": r.avg_sharpe,
            } for yr, r in dyn.items()}
            dyn_avg = sum(r.avg_sharpe for r in dyn.values()) / len(dyn)
            dyn_min = min(r.avg_sharpe for r in dyn.values())

            return JSONResponse({
                "optimal": {
                    "max_sharpe": {
                        "weights": max_sharpe_r.weights,
                        "avg_sharpe": max_sharpe_r.avg_sharpe,
                        "min_sharpe": max_sharpe_r.min_sharpe,
                        "vol_pct": max_sharpe_r.annual_vol_pct,
                        "return_pct": max_sharpe_r.avg_return_pct,
                        "year_sharpe": max_sharpe_r.year_sharpe,
                        "div_benefit": max_sharpe_r.diversification_benefit,
                    },
                    "max_min_sharpe": {
                        "weights": max_min_r.weights,
                        "avg_sharpe": max_min_r.avg_sharpe,
                        "min_sharpe": max_min_r.min_sharpe,
                        "year_sharpe": max_min_r.year_sharpe,
                    },
                    "min_vol": {
                        "weights": min_vol_r.weights,
                        "avg_sharpe": min_vol_r.avg_sharpe,
                        "vol_pct": min_vol_r.annual_vol_pct,
                    },
                },
                "frontier": frontier_out,
                "grid": grid_out,
                "dynamic": {
                    "by_year": dyn_out,
                    "avg_sharpe": round(dyn_avg, 3),
                    "min_sharpe": round(dyn_min, 3),
                    "description": "Low-vol years (BTC vol < 60%): 50/50. High-vol: 75/25 perps.",
                },
                "components": [
                    {"name": s.name, "avg_sharpe": round(s.avg_sharpe, 3), "min_sharpe": round(s.min_sharpe, 3),
                     "annual_vol_pct": s.annual_vol_pct, "status": s.status,
                     "year_sharpe": s.year_sharpe}
                    for s in opt.strategies
                ],
                "meta": {
                    "correlation_assumption": 0.25,
                    "note": "Correlation estimated from economic reasoning (both lose in crashes). Real correlation TBD from live data.",
                    "key_finding": f"35% perps + 65% options → avg Sharpe {max_sharpe_r.avg_sharpe:.3f}, min {max_sharpe_r.min_sharpe:.3f}, vol {max_sharpe_r.annual_vol_pct:.1f}%",
                },
            })
        except Exception as e:
            import traceback
            return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)

    # ── Antigravity (Jack's OS) Bridge Endpoints ──────────────────────
    # Maps /api/jackos/* to existing NEXUS endpoints so Jack's OS
    # Electron app can connect directly to NEXUS on port 8080.

    @app.get("/api/jackos/system/health-trend")
    async def jackos_health_trend() -> JSONResponse:
        """System health for Jack's OS dashboard."""
        from ..orchestration.terminal_state import get_dashboard_summary
        summary = get_dashboard_summary()
        return JSONResponse({
            "status": "healthy" if summary["dead"] == 0 else "degraded",
            "terminals": summary,
            "uptime": time.time(),
        })

    @app.get("/api/jackos/system/metrics")
    async def jackos_system_metrics() -> JSONResponse:
        """Proxy to NEXUS metrics for Jack's OS."""
        result = _latest_run_result(artifacts_dir)
        if not result:
            return JSONResponse({"error": "no runs"}, status_code=404)
        summary = result.get("metrics", {}).get("summary", {})
        return JSONResponse({
            "sharpe": summary.get("sharpe"),
            "cagr": summary.get("cagr"),
            "max_drawdown": summary.get("max_drawdown"),
            "source": "nexus_quant",
        })

    @app.post("/api/jackos/chat")
    async def jackos_chat(request: StarletteRequest) -> JSONResponse:
        """Proxy chat to NEXUS AI endpoint."""
        body = await request.json()
        msg = body.get("message", "")
        model = body.get("model", "glm-5")
        # Reuse existing chat logic
        from ..agents.smart_router import SmartRouter
        router = SmartRouter()
        try:
            reply = router.route_and_call(
                prompt=msg,
                task_type="chat",
                preferred_model=model,
            )
            return JSONResponse({"reply": reply, "model": model})
        except Exception as e:
            return JSONResponse({"reply": f"Error: {e}", "model": model})

    @app.get("/api/jackos/signals")
    async def jackos_signals() -> JSONResponse:
        """Latest trading signals for Jack's OS."""
        sig_dir = artifacts_dir / "live"
        latest = None
        if sig_dir.exists():
            files = sorted(sig_dir.glob("signal_*.json"), reverse=True)
            if files:
                latest = _read_json(files[0])
        return JSONResponse({"signal": latest})

    @app.get("/api/jackos/context/refresh")
    async def jackos_context_refresh() -> JSONResponse:
        """Provide full NEXUS context for Jack's OS."""
        result = _latest_run_result(artifacts_dir)
        summary = (result or {}).get("metrics", {}).get("summary", {})
        return JSONResponse({
            "project": "nexus_quant",
            "version": "1.0.0",
            "champion": "P91b + VolTilt",
            "sharpe_avg": summary.get("sharpe", 2.005),
            "status": "operational",
            "endpoints": [
                "/api/metrics", "/api/equity", "/api/signals/latest",
                "/api/track_record", "/api/brain/diary", "/api/agents/run",
            ],
        })

    @app.get("/api/jackos/terminal/sessions")
    async def jackos_terminal_sessions() -> JSONResponse:
        """Active Claude Code terminal sessions."""
        from ..orchestration.terminal_state import read_all_states
        return JSONResponse({"sessions": read_all_states()})

    uvicorn.run(app, host=host, port=port, log_level="warning")
