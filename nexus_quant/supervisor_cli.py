from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .orchestration.tasks import TaskStore
from .utils.time import parse_iso_utc


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _acquire_pid_lock(pid_path: Path) -> bool:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    own_pid = os.getpid()
    try:
        fd = os.open(str(pid_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(own_pid))
        return True
    except FileExistsError:
        existing_pid = 0
        try:
            existing_pid = int(pid_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            existing_pid = 0
        if existing_pid and existing_pid != own_pid and _pid_alive(existing_pid):
            return False
        try:
            pid_path.unlink()
        except Exception:
            pass
        try:
            fd = os.open(str(pid_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(str(own_pid))
            return True
        except Exception:
            return False


def _rotate_log_if_needed(log_path: Path, max_log_bytes: int) -> None:
    if max_log_bytes <= 0:
        return
    try:
        if not log_path.exists():
            return
        if log_path.stat().st_size <= max_log_bytes:
            return
        rotated = log_path.with_suffix(log_path.suffix + ".1")
        if rotated.exists():
            rotated.unlink()
        log_path.replace(rotated)
    except Exception:
        return


def _heartbeat_age_seconds(artifacts_dir: Path) -> Optional[int]:
    hb = _read_json(artifacts_dir / "state" / "orion_heartbeat.json")
    ts = str(hb.get("ts") or "").strip()
    if not ts:
        return None
    try:
        return max(0, _now_ts() - parse_iso_utc(ts))
    except Exception:
        return None


def _running_task_age_seconds(artifacts_dir: Path) -> Optional[int]:
    tasks_db = artifacts_dir / "state" / "tasks.db"
    if not tasks_db.exists():
        return None
    store = TaskStore(tasks_db)
    try:
        now = _now_ts()
        ages: List[int] = []
        for t in store.recent(limit=200):
            if t.status != "running":
                continue
            try:
                ages.append(max(0, now - parse_iso_utc(str(t.updated_at))))
            except Exception:
                continue
        if not ages:
            return None
        return min(ages)
    finally:
        store.close()


def _estimate_daily_opus_spend_usd(artifacts_dir: Path, default_cost_per_rebuttal: float) -> float:
    review_dir = artifacts_dir / "state" / "policy_reviews"
    if not review_dir.exists():
        return 0.0
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    total = 0.0
    for p in review_dir.glob("rebuttal.*.json"):
        stem = p.stem
        parts = stem.split(".")
        if len(parts) < 3:
            continue
        ts_token = parts[-1]
        if not ts_token.startswith(today):
            continue
        cost = float(default_cost_per_rebuttal)
        obj = _read_json(p)
        try:
            if "budget_usd" in obj:
                cost = float(obj.get("budget_usd"))
        except Exception:
            pass
        cmd = obj.get("command")
        if isinstance(cmd, list):
            try:
                idx = cmd.index("--max-budget-usd")
                if idx + 1 < len(cmd):
                    cost = float(cmd[idx + 1])
            except Exception:
                pass
        total += max(0.0, cost)
    return float(round(total, 6))


def _build_autopilot_cmd(
    *,
    config_path: Path,
    artifacts_dir: Path,
    trials: int,
    steps: int,
    interval_seconds: int,
    bootstrap: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "nexus_quant",
        "autopilot",
        "--config",
        str(config_path),
        "--artifacts",
        str(artifacts_dir),
        "--trials",
        str(int(trials)),
        "--steps",
        str(int(steps)),
        "--loop",
        "--interval-seconds",
        str(int(interval_seconds)),
        "--max-cycles",
        "0",
    ]
    if bootstrap:
        cmd.append("--bootstrap")
    return cmd


def _start_autopilot(cmd: List[str], log_path: Path) -> Tuple[subprocess.Popen, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        close_fds=True,
    )
    return proc, log_handle


def _stop_autopilot(proc: subprocess.Popen, timeout_seconds: int = 20) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=max(1, int(timeout_seconds)))
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _write_status(path: Path, payload: Dict[str, Any]) -> None:
    _write_json(path, {"updated_at": _utc_iso(), **payload})


def supervisor_main(
    *,
    config_path: Path,
    artifacts_dir: Path,
    trials: int,
    steps: int,
    autopilot_interval_seconds: int,
    check_interval_seconds: int,
    stale_seconds: int,
    running_task_grace_seconds: int,
    max_restarts: int,
    restart_window_seconds: int,
    base_backoff_seconds: int,
    max_backoff_seconds: int,
    bootstrap: bool,
    daily_budget_usd: float,
    estimated_opus_cost_usd: float,
    budget_safety_multiplier: float,
    log_file: str,
    max_log_mb: int,
) -> int:
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    state_dir = artifacts_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    check_interval_seconds = max(5, min(int(check_interval_seconds), 3600))
    autopilot_interval_seconds = max(5, min(int(autopilot_interval_seconds), 3600))
    stale_seconds = max(60, int(stale_seconds))
    running_task_grace_seconds = max(stale_seconds, int(running_task_grace_seconds))
    max_restarts = max(1, min(int(max_restarts), 100))
    restart_window_seconds = max(60, int(restart_window_seconds))
    base_backoff_seconds = max(1, int(base_backoff_seconds))
    max_backoff_seconds = max(base_backoff_seconds, int(max_backoff_seconds))
    daily_budget_usd = max(0.0, float(daily_budget_usd))
    estimated_opus_cost_usd = max(0.0, float(estimated_opus_cost_usd))
    budget_safety_multiplier = max(1.0, min(float(budget_safety_multiplier), 10.0))
    max_log_mb = max(1, min(int(max_log_mb), 4096))
    max_log_bytes = int(max_log_mb) * 1024 * 1024

    status_path = state_dir / "orion_supervisor_status.json"
    control_path = state_dir / "orion_supervisor_control.json"
    pid_path = state_dir / "orion_supervisor.pid"
    log_path = Path(log_file).expanduser() if str(log_file).strip() else (artifacts_dir / "logs" / "orion_autopilot.log")

    if not _acquire_pid_lock(pid_path):
        existing_pid = 0
        try:
            existing_pid = int(pid_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            existing_pid = 0
        print(f"Supervisor already running (pid={existing_pid}).")
        return 0

    proc: Optional[subprocess.Popen] = None
    log_handle: Any = None
    restart_timestamps: List[int] = []
    backoff_seconds = int(base_backoff_seconds)
    last_reason = "startup"
    shutdown = {"requested": False, "signal": ""}

    def _request_shutdown(sig: int, _frame: Any) -> None:
        try:
            shutdown["signal"] = signal.Signals(sig).name
        except Exception:
            shutdown["signal"] = str(sig)
        shutdown["requested"] = True

    old_sigterm = signal.getsignal(signal.SIGTERM)
    old_sighup = signal.getsignal(signal.SIGHUP)
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGHUP, _request_shutdown)

    try:
        while True:
            if shutdown["requested"]:
                _write_status(
                    status_path,
                    {
                        "state": "stopped",
                        "reason": f"signal_{shutdown['signal'] or 'UNKNOWN'}",
                        "autopilot_pid": int(proc.pid) if proc is not None and proc.poll() is None else None,
                    },
                )
                return 0

            now = _now_ts()
            restart_timestamps = [ts for ts in restart_timestamps if (now - ts) <= restart_window_seconds]
            control = _read_json(control_path)
            control_mode = str(control.get("mode") or "").strip().upper()

            if control_mode in {"PAUSE", "PAUSED", "STOP"}:
                control_reason = str(control.get("reason") or "manual_control")
                if control_reason == "daily_budget_exceeded":
                    budget_day = str(control.get("budget_day_utc") or "")
                    today_day = datetime.now(timezone.utc).strftime("%Y%m%d")
                    if budget_day and budget_day != today_day:
                        try:
                            control_path.unlink()
                        except Exception:
                            pass
                        control = {}
                        control_mode = ""

            if control_mode in {"PAUSE", "PAUSED", "STOP"}:
                if proc is not None:
                    _stop_autopilot(proc)
                    proc = None
                if log_handle is not None:
                    try:
                        log_handle.close()
                    except Exception:
                        pass
                    log_handle = None
                _write_status(
                    status_path,
                    {
                        "state": "paused_by_control",
                        "reason": str(control.get("reason") or "manual_control"),
                        "control_mode": control_mode,
                        "autopilot_pid": None,
                        "restart_count_window": len(restart_timestamps),
                    },
                )
                time.sleep(check_interval_seconds)
                continue

            estimated_daily_spend = _estimate_daily_opus_spend_usd(artifacts_dir, estimated_opus_cost_usd)
            adjusted_daily_spend = float(estimated_daily_spend) * float(budget_safety_multiplier)
            if daily_budget_usd > 0.0 and adjusted_daily_spend >= daily_budget_usd:
                if proc is not None:
                    _stop_autopilot(proc)
                    proc = None
                if log_handle is not None:
                    try:
                        log_handle.close()
                    except Exception:
                        pass
                    log_handle = None
                budget_control = {
                    "mode": "PAUSED",
                    "reason": "daily_budget_exceeded",
                    "budget_day_utc": datetime.now(timezone.utc).strftime("%Y%m%d"),
                    "daily_budget_usd": float(daily_budget_usd),
                    "estimated_daily_spend_usd": float(estimated_daily_spend),
                    "adjusted_daily_spend_usd": float(adjusted_daily_spend),
                    "budget_safety_multiplier": float(budget_safety_multiplier),
                    "estimated_spend_method": "rebuttal_artifact_budget_ceiling",
                    "updated_at": _utc_iso(),
                }
                _write_json(control_path, budget_control)
                _write_status(
                    status_path,
                    {
                        "state": "paused_budget",
                        "reason": "daily_budget_exceeded",
                        "daily_budget_usd": float(daily_budget_usd),
                        "estimated_daily_spend_usd": float(estimated_daily_spend),
                        "adjusted_daily_spend_usd": float(adjusted_daily_spend),
                        "budget_safety_multiplier": float(budget_safety_multiplier),
                        "estimated_spend_method": "rebuttal_artifact_budget_ceiling",
                        "autopilot_pid": None,
                    },
                )
                print(
                    f"Supervisor paused by budget guard: estimated_daily_spend_usd={estimated_daily_spend} "
                    f"(adjusted={adjusted_daily_spend}, multiplier={budget_safety_multiplier}) "
                    f">= daily_budget_usd={daily_budget_usd}"
                )
                time.sleep(check_interval_seconds)
                continue

            child_dead = (proc is None) or (proc.poll() is not None)
            if child_dead:
                if len(restart_timestamps) >= max_restarts:
                    cap_control = {
                        "mode": "PAUSED",
                        "reason": "max_restarts_exceeded",
                        "restart_count_window": len(restart_timestamps),
                        "restart_window_seconds": int(restart_window_seconds),
                        "last_reason": str(last_reason),
                        "updated_at": _utc_iso(),
                    }
                    _write_json(control_path, cap_control)
                    _write_status(
                        status_path,
                        {
                            "state": "paused_restart_cap",
                            "reason": "max_restarts_exceeded",
                            "restart_count_window": len(restart_timestamps),
                            "restart_window_seconds": int(restart_window_seconds),
                            "autopilot_pid": None,
                        },
                    )
                    print(
                        f"Supervisor paused by restart cap: {len(restart_timestamps)} restarts "
                        f"within {restart_window_seconds}s."
                    )
                    time.sleep(check_interval_seconds)
                    continue

                if restart_timestamps:
                    time.sleep(backoff_seconds)

                if log_handle is not None:
                    try:
                        log_handle.close()
                    except Exception:
                        pass
                    log_handle = None

                _rotate_log_if_needed(log_path, max_log_bytes)

                cmd = _build_autopilot_cmd(
                    config_path=config_path,
                    artifacts_dir=artifacts_dir,
                    trials=int(trials),
                    steps=int(steps),
                    interval_seconds=int(autopilot_interval_seconds),
                    bootstrap=bool(bootstrap),
                )
                proc, log_handle = _start_autopilot(cmd, log_path)
                restart_timestamps.append(_now_ts())
                backoff_seconds = min(max_backoff_seconds, max(base_backoff_seconds, backoff_seconds * 2))
                _write_status(
                    status_path,
                    {
                        "state": "running",
                        "reason": "autopilot_started",
                        "autopilot_pid": int(proc.pid),
                        "autopilot_log_path": str(log_path),
                        "restart_count_window": len(restart_timestamps),
                        "restart_window_seconds": int(restart_window_seconds),
                        "command": cmd,
                        "estimated_daily_spend_usd": float(estimated_daily_spend),
                        "adjusted_daily_spend_usd": float(adjusted_daily_spend),
                        "budget_safety_multiplier": float(budget_safety_multiplier),
                        "estimated_spend_method": "rebuttal_artifact_budget_ceiling",
                        "daily_budget_usd": float(daily_budget_usd),
                    },
                )
                print(f"Supervisor started autopilot pid={proc.pid}.")
                time.sleep(check_interval_seconds)
                continue

            heartbeat_age = _heartbeat_age_seconds(artifacts_dir)
            running_task_age = _running_task_age_seconds(artifacts_dir)
            health_reason = "ok"
            healthy = True

            if heartbeat_age is None:
                process_age = max(0, now - restart_timestamps[-1]) if restart_timestamps else 0
                if process_age > stale_seconds:
                    if running_task_age is not None and running_task_age <= running_task_grace_seconds:
                        health_reason = "missing_heartbeat_but_task_running"
                    else:
                        healthy = False
                        health_reason = "missing_heartbeat"
            elif heartbeat_age > stale_seconds:
                if running_task_age is not None and running_task_age <= running_task_grace_seconds:
                    health_reason = "heartbeat_stale_but_task_running"
                else:
                    healthy = False
                    health_reason = f"heartbeat_stale_{heartbeat_age}s"

            if not healthy:
                _write_status(
                    status_path,
                    {
                        "state": "restarting",
                        "reason": str(health_reason),
                        "autopilot_pid": int(proc.pid),
                        "heartbeat_age_seconds": heartbeat_age,
                        "running_task_age_seconds": running_task_age,
                        "restart_count_window": len(restart_timestamps),
                        "restart_window_seconds": int(restart_window_seconds),
                        "estimated_daily_spend_usd": float(estimated_daily_spend),
                        "adjusted_daily_spend_usd": float(adjusted_daily_spend),
                        "budget_safety_multiplier": float(budget_safety_multiplier),
                        "estimated_spend_method": "rebuttal_artifact_budget_ceiling",
                        "daily_budget_usd": float(daily_budget_usd),
                    },
                )
                _stop_autopilot(proc)
                proc = None
                if log_handle is not None:
                    try:
                        log_handle.close()
                    except Exception:
                        pass
                    log_handle = None
                last_reason = str(health_reason)
                continue

            backoff_seconds = int(base_backoff_seconds)
            _write_status(
                status_path,
                {
                    "state": "running",
                    "reason": str(health_reason),
                    "autopilot_pid": int(proc.pid),
                    "autopilot_log_path": str(log_path),
                    "heartbeat_age_seconds": heartbeat_age,
                    "running_task_age_seconds": running_task_age,
                    "restart_count_window": len(restart_timestamps),
                    "restart_window_seconds": int(restart_window_seconds),
                    "estimated_daily_spend_usd": float(estimated_daily_spend),
                    "adjusted_daily_spend_usd": float(adjusted_daily_spend),
                    "budget_safety_multiplier": float(budget_safety_multiplier),
                    "estimated_spend_method": "rebuttal_artifact_budget_ceiling",
                    "daily_budget_usd": float(daily_budget_usd),
                },
            )
            time.sleep(check_interval_seconds)

    except KeyboardInterrupt:
        _write_status(
            status_path,
            {
                "state": "stopped",
                "reason": "keyboard_interrupt",
                "autopilot_pid": int(proc.pid) if proc is not None and proc.poll() is None else None,
            },
        )
        return 0
    finally:
        try:
            signal.signal(signal.SIGTERM, old_sigterm)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGHUP, old_sighup)
        except Exception:
            pass
        if proc is not None and proc.poll() is None:
            _stop_autopilot(proc)
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass
        try:
            current = int(pid_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            current = 0
        if current == os.getpid():
            try:
                pid_path.unlink()
            except Exception:
                pass
