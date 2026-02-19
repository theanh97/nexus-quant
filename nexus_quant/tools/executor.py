from __future__ import annotations

"""
NEXUS Executor - Runs tasks on behalf of NEXUS (computer use capabilities).
NEXUS can: run backtests, read files, list directories, execute research cycles.
"""

import contextlib
import json
import os
import re
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_TIMEOUT_SEC = 120
_MAX_READ_CHARS_HARD_LIMIT = 200_000


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class NexusExecutor:
    """Runs tasks on behalf of NEXUS with local safety restrictions."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        # Security: only allow reads/listing inside the project directory.
        self._allowed_roots = [self.project_root]

    # ── Safety helpers ──────────────────────────────────────────────────

    def _resolve_allowed_path(self, raw_path: str) -> tuple[Optional[Path], Optional[str]]:
        path_str = (raw_path or "").strip()
        if not path_str:
            return None, "empty_path"

        p = Path(path_str).expanduser()
        if not p.is_absolute():
            p = self.project_root / p

        try:
            rp = p.resolve(strict=False)
        except Exception as e:
            return None, f"invalid_path: {e}"

        for root in self._allowed_roots:
            try:
                rp.relative_to(root.resolve())
                return rp, None
            except Exception:
                continue

        return None, "access_denied"

    def _resolve_allowed_dir(self, raw_path: str) -> tuple[Optional[Path], Optional[str]]:
        rp, err = self._resolve_allowed_path(raw_path)
        if err:
            return None, err
        if rp is None:
            return None, "invalid_path"
        if rp.exists() and not rp.is_dir():
            return None, "not_a_directory"
        return rp, None

    # ── Tools ──────────────────────────────────────────────────────────

    def run_backtest(self, config_path: str, artifacts_dir: str) -> Dict[str, Any]:
        """
        Runs:
            python3 -m nexus_quant run --config <config_path> --out <artifacts_dir>
        Returns:
            {ok, stdout, stderr, duration_sec, run_id}
        Timeout:
            120s
        """
        start = time.monotonic()
        out: Dict[str, Any] = {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "duration_sec": 0.0,
            "run_id": None,
        }

        cfg_path, cfg_err = self._resolve_allowed_path(config_path)
        if cfg_err or cfg_path is None:
            out["stderr"] = f"config_path_error: {cfg_err or 'invalid_path'}"
            return out
        if not cfg_path.exists() or not cfg_path.is_file():
            out["stderr"] = f"config_not_found: {config_path}"
            return out

        artifacts_path, aerr = self._resolve_allowed_dir(artifacts_dir)
        if aerr or artifacts_path is None:
            out["stderr"] = f"artifacts_dir_error: {aerr or 'invalid_path'}"
            return out

        runs_dir = artifacts_path / "runs"
        before: set[str] = set()
        try:
            if runs_dir.exists():
                before = {d.name for d in runs_dir.iterdir() if d.is_dir()}
        except Exception:
            before = set()

        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.project_root)

        cmd = [
            "python3",
            "-m",
            "nexus_quant",
            "run",
            "--config",
            str(cfg_path),
            "--out",
            str(artifacts_path),
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=_DEFAULT_TIMEOUT_SEC,
                env=env,
            )
            out["stdout"] = proc.stdout or ""
            out["stderr"] = proc.stderr or ""
            out["ok"] = proc.returncode == 0
        except subprocess.TimeoutExpired as e:
            out["stdout"] = (e.stdout or "") if isinstance(e.stdout, str) else ""
            out["stderr"] = (e.stderr or "") if isinstance(e.stderr, str) else ""
            out["stderr"] = (out["stderr"] + "\n" if out["stderr"] else "") + f"timeout_after_{_DEFAULT_TIMEOUT_SEC}s"
            out["ok"] = False
        except Exception as e:
            out["stderr"] = f"exception: {e}"
            out["ok"] = False
        finally:
            out["duration_sec"] = float(time.monotonic() - start)

        # Best-effort: infer run_id from newly created runs directory.
        try:
            if runs_dir.exists():
                after = {d.name for d in runs_dir.iterdir() if d.is_dir()}
                created = sorted(
                    (runs_dir / n for n in (after - before)),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if created:
                    out["run_id"] = created[0].name
        except Exception:
            pass

        return out

    def read_file(self, path: str, max_chars: int = 3000) -> Dict[str, Any]:
        """Safely reads a file and returns {content, size, modified, error}."""
        out: Dict[str, Any] = {"content": "", "size": None, "modified": None, "error": None}

        try:
            max_chars_i = int(max_chars)
        except Exception:
            max_chars_i = 3000
        max_chars_i = max(0, min(max_chars_i, _MAX_READ_CHARS_HARD_LIMIT))

        rp, err = self._resolve_allowed_path(path)
        if err or rp is None:
            out["error"] = err or "invalid_path"
            return out
        if not rp.exists():
            out["error"] = "file_not_found"
            return out
        if not rp.is_file():
            out["error"] = "not_a_file"
            return out

        try:
            st = rp.stat()
            out["size"] = int(st.st_size)
            out["modified"] = _utc_iso(st.st_mtime)

            if max_chars_i <= 0:
                return out

            byte_limit = min(int(st.st_size), max_chars_i * 4 + 1024)
            with rp.open("rb") as f:
                data = f.read(max(0, int(byte_limit)))
            text = data.decode("utf-8", errors="replace")
            out["content"] = text[:max_chars_i]
            return out
        except Exception as e:
            out["error"] = str(e)
            return out

    def list_directory(self, path: str) -> Dict[str, Any]:
        """Lists directory contents with sizes/dates. Returns {files: [...], dirs: [...], error}."""
        out: Dict[str, Any] = {"files": [], "dirs": [], "error": None}

        rp, err = self._resolve_allowed_path(path)
        if err or rp is None:
            out["error"] = err or "invalid_path"
            return out
        if not rp.exists():
            out["error"] = "directory_not_found"
            return out
        if not rp.is_dir():
            out["error"] = "not_a_directory"
            return out

        try:
            files: List[Dict[str, Any]] = []
            dirs: List[Dict[str, Any]] = []
            for item in sorted(rp.iterdir(), key=lambda p: p.name.lower()):
                with contextlib.suppress(Exception):
                    st = item.stat()
                    entry = {
                        "name": item.name,
                        "path": str(item),
                        "size": int(st.st_size),
                        "modified": _utc_iso(st.st_mtime),
                    }
                    if item.is_dir():
                        dirs.append(entry)
                    else:
                        files.append(entry)
            out["files"] = files
            out["dirs"] = dirs
            return out
        except Exception as e:
            out["error"] = str(e)
            return out

    def get_system_state(self, artifacts_dir: str) -> Dict[str, Any]:
        """
        Returns current NEXUS system state:
          {
            latest_run: {name, sharpe, mdd, verdict},
            goals: [...],
            mood: ...,
            brain_diary: ...,
            active_experiments: int,
            memory_items: int
          }
        """
        state: Dict[str, Any] = {
            "latest_run": {"name": None, "sharpe": None, "mdd": None, "verdict": None},
            "goals": [],
            "mood": "unknown",
            "brain_diary": {},
            "active_experiments": 0,
            "memory_items": 0,
        }

        artifacts_path, err = self._resolve_allowed_dir(artifacts_dir)
        if err or artifacts_path is None:
            return state

        # Latest run: use ledger if present (more precise than mtime).
        ledger_path = artifacts_path / "ledger" / "ledger.jsonl"
        latest_run_event: Optional[Dict[str, Any]] = None
        if ledger_path.exists():
            try:
                lines = ledger_path.read_text(encoding="utf-8").splitlines()
                for ln in reversed(lines[-1000:]):  # guardrail
                    ln = (ln or "").strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and obj.get("kind") == "run":
                        latest_run_event = obj
                        break
            except Exception:
                latest_run_event = None

        if latest_run_event is not None:
            payload = latest_run_event.get("payload") or {}
            metrics = payload.get("metrics") or {}
            state["latest_run"] = {
                "name": latest_run_event.get("run_name") or latest_run_event.get("run_id"),
                "sharpe": metrics.get("sharpe"),
                "mdd": metrics.get("max_drawdown"),
                "verdict": payload.get("verdict"),
            }
        else:
            # Fallback: newest run directory with metrics.json present.
            runs_dir = artifacts_path / "runs"
            if runs_dir.exists():
                try:
                    dirs = sorted(
                        [d for d in runs_dir.iterdir() if d.is_dir()],
                        key=lambda d: d.stat().st_mtime,
                        reverse=True,
                    )
                    for d in dirs[:50]:
                        mp = d / "metrics.json"
                        if not mp.exists():
                            continue
                        m = json.loads(mp.read_text("utf-8"))
                        summary = m.get("summary") or {}
                        state["latest_run"] = {
                            "name": d.name,
                            "sharpe": summary.get("sharpe"),
                            "mdd": summary.get("max_drawdown"),
                            "verdict": m.get("verdict"),
                        }
                        break
                except Exception:
                    pass

        # Goals (active)
        goals_path = artifacts_path / "brain" / "goals.json"
        if goals_path.exists():
            try:
                goals = json.loads(goals_path.read_text("utf-8"))
                if isinstance(goals, list):
                    state["goals"] = [
                        str(g.get("title") or "")
                        for g in goals
                        if isinstance(g, dict) and str(g.get("status") or "").lower() == "active"
                    ]
            except Exception:
                pass

        # Brain diary (latest)
        diary_path = artifacts_path / "brain" / "diary" / "latest.json"
        if diary_path.exists():
            try:
                diary = json.loads(diary_path.read_text("utf-8"))
                if isinstance(diary, dict):
                    state["brain_diary"] = diary
            except Exception:
                pass

        # Mood: identity.json -> diary.mood -> unknown
        identity_path = artifacts_path / "brain" / "identity.json"
        if identity_path.exists():
            try:
                ident = json.loads(identity_path.read_text("utf-8"))
                if isinstance(ident, dict) and ident.get("mood"):
                    state["mood"] = str(ident.get("mood"))
            except Exception:
                pass
        if state["mood"] == "unknown":
            mood_from_diary = (state.get("brain_diary") or {}).get("mood")
            if mood_from_diary:
                state["mood"] = str(mood_from_diary)

        # Active experiments: pending/running experiment tasks in tasks.db
        tasks_db = artifacts_path / "state" / "tasks.db"
        if tasks_db.exists():
            try:
                conn = sqlite3.connect(str(tasks_db))
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT COUNT(1) FROM tasks WHERE kind=? AND status IN ('pending','running')",
                        ("experiment",),
                    )
                    row = cur.fetchone()
                    if row:
                        state["active_experiments"] = int(row[0])
                finally:
                    conn.close()
            except Exception:
                pass

        # Memory items: count memory_items rows
        mem_db = artifacts_path / "memory" / "memory.db"
        if mem_db.exists():
            try:
                conn = sqlite3.connect(str(mem_db))
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(1) FROM memory_items")
                    row = cur.fetchone()
                    if row:
                        state["memory_items"] = int(row[0])
                finally:
                    conn.close()
            except Exception:
                pass

        return state

    def handle_command(self, command: str, artifacts_dir: str) -> Dict[str, Any]:
        """
        Natural language command dispatcher:
          "run backtest X" -> run_backtest()
          "read file X" -> read_file()
          "list X" -> list_directory()
          "status" -> get_system_state()

        Returns {action, result, message}.
        """
        cmd = (command or "").strip()
        low = cmd.lower()

        def _extract_arg(after_phrase: str) -> Optional[str]:
            m = re.search(rf"\\b{re.escape(after_phrase)}\\b(.*)$", cmd, flags=re.IGNORECASE)
            if not m:
                return None
            tail = (m.group(1) or "").strip()
            if not tail:
                return None
            for pat in (r"`([^`]+)`", r"\"([^\"]+)\"", r"'([^']+)'"):
                mm = re.search(pat, tail)
                if mm:
                    return (mm.group(1) or "").strip() or None
            return tail

        if low == "status" or low.startswith("status "):
            result = self.get_system_state(artifacts_dir)
            lr = result.get("latest_run") or {}
            msg = f"mood={result.get('mood')} latest_run={lr.get('name')}"
            return {"action": "get_system_state", "result": result, "message": msg}

        if "run backtest" in low:
            cfg = _extract_arg("run backtest")
            if not cfg:
                return {
                    "action": "run_backtest",
                    "result": {},
                    "message": "Missing config path (try: run backtest configs/run.json)",
                }
            result = self.run_backtest(cfg, artifacts_dir)
            run_id = result.get("run_id")
            msg = (
                f"backtest_ok run_id={run_id}"
                if result.get("ok")
                else f"backtest_failed: {str(result.get('stderr') or '')[:200]}"
            )
            return {"action": "run_backtest", "result": result, "message": msg}

        if "read file" in low:
            pth = _extract_arg("read file")
            if not pth:
                return {"action": "read_file", "result": {}, "message": "Missing file path (try: read file README.md)"}
            result = self.read_file(pth)
            msg = "read_ok" if not result.get("error") else f"read_failed: {result.get('error')}"
            return {"action": "read_file", "result": result, "message": msg}

        if low.startswith("list") or low.startswith("ls"):
            pth = _extract_arg("list") if low.startswith("list") else _extract_arg("ls")
            if not pth:
                pth = "."
            result = self.list_directory(pth)
            if result.get("error"):
                msg = f"list_failed: {result.get('error')}"
            else:
                msg = f"list_ok dirs={len(result.get('dirs') or [])} files={len(result.get('files') or [])}"
            return {"action": "list_directory", "result": result, "message": msg}

        return {
            "action": "unknown",
            "result": {},
            "message": "I can: 'run backtest <config>', 'read file <path>', 'list <dir>', 'status'",
        }

