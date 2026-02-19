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
from pathlib import Path
from typing import Any, Dict, List, Optional


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

    from fastapi import FastAPI
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
