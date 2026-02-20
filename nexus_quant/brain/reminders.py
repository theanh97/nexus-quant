from __future__ import annotations

try:
    import json
except Exception:  # pragma: no cover
    json = None  # type: ignore

try:
    from datetime import datetime, timezone
except Exception:  # pragma: no cover
    datetime = None  # type: ignore
    timezone = None  # type: ignore

try:
    from pathlib import Path
except Exception:  # pragma: no cover
    Path = None  # type: ignore

try:
    from typing import Any, Dict, List, Optional, Tuple
except Exception:  # pragma: no cover
    Any = object  # type: ignore
    Dict = dict  # type: ignore
    List = list  # type: ignore
    Optional = object  # type: ignore
    Tuple = tuple  # type: ignore

try:
    from ..data.provider_policy import classify_provider
except Exception:  # pragma: no cover
    classify_provider = None  # type: ignore


def write_reminder(artifacts_dir: "Path") -> None:
    """Write current system state as reminder for next Claude session."""

    if Path is None or json is None or datetime is None or timezone is None:  # pragma: no cover
        return

    root = Path(artifacts_dir)
    reminder_path = root / "state" / "current_reminder.md"
    reminder_path.parent.mkdir(parents=True, exist_ok=True)

    registry_path = root / "memory" / "semantic" / "strategy_registry.json"
    lessons_path = root / "memory" / "semantic" / "lessons_learned.jsonl"
    goals_path = root / "brain" / "goals.json"

    registry = _read_json_dict(registry_path)
    strategies = {}
    if isinstance(registry, dict):
        strategies = registry.get("strategies") or {}
    if not isinstance(strategies, dict):
        strategies = {}

    best_name, best_sharpe = _pick_best_strategy(strategies)
    best_line = "N/A"
    if best_name:
        if best_sharpe is None:
            best_line = str(best_name)
        else:
            best_line = f"{best_name} (Sharpe {best_sharpe:.3f})"
    else:
        best_line = "N/A (no real-data strategy yet)"

    lessons_tail = _read_jsonl_tail(lessons_path, n=200)
    lessons = lessons_tail[-5:]
    lessons_lines = _format_lessons(lessons)

    stagnant = sorted({str(x.get("strategy") or "") for x in lessons_tail if str(x.get("kind") or "") == "stagnation"})
    stagnant = [s for s in stagnant if s][:8]
    stagnant_s = f"[{', '.join(stagnant)}]" if stagnant else "[]"

    active_goals = _read_active_goals(goals_path)
    open_problem = _format_open_problem(active_goals)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    strategy_count = len(strategies)

    body = "\n".join(
        [
            f"# NEXUS System Reminder — {ts}",
            "## Current Focus",
            f"- Best strategy: {best_line}",
            f"- Open problem: {open_problem}",
            f"- Memory curator: {strategy_count} strategies, stagnation in {stagnant_s}",
            "",
            "## Rules for Brain Loop",
            "1. Always check BrainCritic.check_novelty() before proposing experiments",
            "2. Run MemoryCurator.curate() every 5 runs",
            "3. Explore NEW directions if stagnation detected: regime-adaptive strategy, new signals",
            "4. Target: Sharpe > 3.0 out-of-sample on 2022+2023+2025 combined",
            "",
            "## Today Lessons Learned",
            lessons_lines or "{lessons from lessons_learned.jsonl}",
            "",
        ]
    )
    reminder_path.write_text(body, encoding="utf-8")


def _read_json_dict(path: "Path") -> Optional[Dict[str, Any]]:
    if Path is None or json is None:  # pragma: no cover
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _read_json_list(path: "Path") -> Optional[List[Any]]:
    if Path is None or json is None:  # pragma: no cover
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else None
    except Exception:
        return None


def _read_jsonl_tail(path: "Path", *, n: int = 5) -> List[Dict[str, Any]]:
    if Path is None or json is None:  # pragma: no cover
        return []
    try:
        limit = int(n)
    except Exception:
        limit = 5
    limit = max(1, min(limit, 200))

    if not path.exists():
        return []
    try:
        lines = path.read_text("utf-8", errors="replace").splitlines()[-limit:]
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for ln in lines:
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _pick_best_strategy(strategies: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    best_real_name: Optional[str] = None
    best_real_sharpe: Optional[float] = None
    best_real_times_run: int = -1
    best_unknown_name: Optional[str] = None
    best_unknown_sharpe: Optional[float] = None
    best_unknown_times_run: int = -1

    for name, rec in (strategies or {}).items():
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(rec, dict):
            continue

        data_cls = _strategy_data_class(name, rec)
        if data_cls == "synthetic":
            continue

        sharpe = rec.get("best_sharpe")
        times_run = rec.get("times_run")
        try:
            s = float(sharpe) if sharpe is not None else None
        except Exception:
            s = None
        try:
            tr = int(times_run) if times_run is not None else 0
        except Exception:
            tr = 0

        if data_cls == "real":
            if s is not None:
                if best_real_sharpe is None or s > best_real_sharpe:
                    best_real_name, best_real_sharpe, best_real_times_run = name, s, tr
                continue
            if best_real_sharpe is None and tr > best_real_times_run:
                best_real_name, best_real_sharpe, best_real_times_run = name, None, tr
            continue

        if s is not None:
            if best_unknown_sharpe is None or s > best_unknown_sharpe:
                best_unknown_name, best_unknown_sharpe, best_unknown_times_run = name, s, tr
            continue

        if best_unknown_sharpe is None and tr > best_unknown_times_run:
            best_unknown_name, best_unknown_sharpe, best_unknown_times_run = name, None, tr

    if best_real_name:
        return best_real_name, best_real_sharpe
    return best_unknown_name, best_unknown_sharpe


def _strategy_data_class(name: str, rec: Dict[str, Any]) -> str:
    cls = rec.get("best_data_provider_class")
    if isinstance(cls, str) and cls.strip():
        return cls.strip().lower()

    provider = rec.get("best_data_provider")
    if isinstance(provider, str) and provider.strip() and classify_provider is not None:
        try:
            return str(classify_provider(provider)).lower()
        except Exception:
            pass

    low = name.lower()
    if "synthetic" in low:
        return "synthetic"
    if low.startswith("binance_"):
        return "real"
    return "unknown"


def _read_active_goals(goals_path: "Path") -> List[Dict[str, Any]]:
    raw = _read_json_list(goals_path) or []
    out: List[Dict[str, Any]] = []
    for g in raw:
        if not isinstance(g, dict):
            continue
        if str(g.get("status") or "") != "active":
            continue
        out.append(g)
    return out


def _format_open_problem(active_goals: List[Dict[str, Any]]) -> str:
    if not active_goals:
        return "No active goals set; explore new directions."
    g = active_goals[0]
    title = str(g.get("title") or "").strip() or "Untitled goal"
    metric = str(g.get("metric") or "").strip() or "metric"
    cur = g.get("current")
    tgt = g.get("target")
    try:
        cur_f = float(cur)
    except Exception:
        cur_f = None
    try:
        tgt_f = float(tgt)
    except Exception:
        tgt_f = None
    if cur_f is not None and tgt_f is not None:
        return f"{title} ({metric} {cur_f:.3f} → {tgt_f:.3f})"
    return title


def _format_goals(active_goals: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for g in active_goals[:6]:
        title = str(g.get("title") or "").strip() or "Untitled goal"
        gid = str(g.get("id") or "").strip()
        metric = str(g.get("metric") or "").strip() or "metric"
        cur = g.get("current")
        tgt = g.get("target")
        try:
            cur_f = float(cur)
        except Exception:
            cur_f = None
        try:
            tgt_f = float(tgt)
        except Exception:
            tgt_f = None
        if cur_f is not None and tgt_f is not None:
            prog = f"{metric} {cur_f:.3f} → {tgt_f:.3f}"
        else:
            prog = metric
        label = f"[{gid}] " if gid else ""
        lines.append(f"- {label}{title} ({prog})")
    return "\n".join(lines)


def _format_lessons(lessons: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for obj in lessons:
        if not isinstance(obj, dict):
            continue
        kind = str(obj.get("kind") or "").strip()
        msg = str(obj.get("message") or "").strip()
        strat = str(obj.get("strategy") or "").strip()
        prefix = f"{kind}: " if kind else ""
        if strat:
            prefix = f"{prefix}{strat}: "
        text = (prefix + msg).strip()
        if not text:
            continue
        lines.append(f"- {text}")
    return "\n".join(lines)
