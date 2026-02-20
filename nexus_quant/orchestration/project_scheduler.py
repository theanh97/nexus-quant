"""
NEXUS Multi-Project Scheduler.

Runs multiple project brain loops concurrently, each in its own thread.
Manages lifecycle, health checks, and cross-pollination across projects.

Usage:
    scheduler = NexusScheduler.from_config("nexus.yaml")
    scheduler.start()    # start all enabled projects
    scheduler.status()   # check health
    scheduler.stop()     # graceful shutdown
"""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.scheduler")


class ProjectRunner:
    """Runs a single project's brain loop in a thread."""

    def __init__(
        self,
        project_name: str,
        config_path: str,
        artifacts_dir: Path,
        memory_root: Path,
        market: str = "",
        interval: int = 600,
        max_concurrent: int = 2,
    ):
        self.project_name = project_name
        self.config_path = config_path
        self.artifacts_dir = artifacts_dir / project_name
        self.memory_root = memory_root
        self.market = market
        self.interval = interval
        self.max_concurrent = max_concurrent

        self.cycle_count = 0
        self.consecutive_errors = 0
        self.last_result: Optional[Dict[str, Any]] = None
        self.status = "idle"
        self._stop_event = threading.Event()

        # Ensure dirs exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run_loop(self) -> None:
        """Main loop — runs brain cycles with error recovery."""
        from ..brain.loop import NexusAutonomousLoop
        from ..memory.namespace import MemoryNamespace

        logger.info("[%s] Starting brain loop (interval=%ds)", self.project_name, self.interval)
        self.status = "running"

        # Initialize memory namespace for this project
        _mem = MemoryNamespace(self.memory_root, self.project_name, self.market)

        # Initialize brain loop
        loop_obj = NexusAutonomousLoop(
            Path(self.config_path),
            self.artifacts_dir,
        )

        heartbeat_path = self.artifacts_dir / "state" / "brain_heartbeat.json"
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

        while not self._stop_event.is_set():
            try:
                result = loop_obj.run_cycle()
                self.cycle_count += 1
                self.consecutive_errors = 0
                self.last_result = result
                self.status = "running"

                # Write heartbeat
                heartbeat_path.write_text(json.dumps({
                    "project": self.project_name,
                    "ts": time.time(),
                    "cycle": self.cycle_count,
                    "status": "ok",
                    "last_sharpe": result.get("best_sharpe"),
                }), encoding="utf-8")

                logger.info("[%s] Cycle %d: sharpe=%s mood=%s",
                            self.project_name, self.cycle_count,
                            result.get("best_sharpe"), result.get("mood"))

            except Exception as exc:
                self.consecutive_errors += 1
                self.status = "error"
                backoff = min(300, 30 * (2 ** (self.consecutive_errors - 1)))
                logger.error("[%s] Error in cycle %d (%d consecutive): %s",
                             self.project_name, self.cycle_count + 1,
                             self.consecutive_errors, exc)
                traceback.print_exc()

                heartbeat_path.write_text(json.dumps({
                    "project": self.project_name,
                    "ts": time.time(),
                    "cycle": self.cycle_count,
                    "status": "error",
                    "error": str(exc),
                    "consecutive_errors": self.consecutive_errors,
                }), encoding="utf-8")

                if self.consecutive_errors >= 10:
                    logger.warning("[%s] 10 consecutive errors — pausing 10 min", self.project_name)
                    self.status = "paused"
                    self._stop_event.wait(600)
                    self.consecutive_errors = 0
                else:
                    self._stop_event.wait(backoff)
                continue

            self._stop_event.wait(self.interval)

        self.status = "stopped"
        logger.info("[%s] Brain loop stopped", self.project_name)

    def stop(self) -> None:
        """Signal the loop to stop."""
        self._stop_event.set()

    def health(self) -> Dict[str, Any]:
        """Return health info."""
        return {
            "project": self.project_name,
            "status": self.status,
            "cycles": self.cycle_count,
            "consecutive_errors": self.consecutive_errors,
            "last_sharpe": (self.last_result or {}).get("best_sharpe"),
        }


class NexusScheduler:
    """
    Orchestrates multiple ProjectRunner instances concurrently.

    Each project runs its own brain loop in a daemon thread.
    Cross-pollination daemon promotes validated insights across projects.
    """

    def __init__(self, memory_root: Path, artifacts_root: Path):
        self.memory_root = memory_root
        self.artifacts_root = artifacts_root
        self.runners: Dict[str, ProjectRunner] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self._cross_pollination_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @classmethod
    def from_config(cls, config_path: str = "nexus.yaml") -> "NexusScheduler":
        """Create scheduler from nexus.yaml master config."""
        from ..projects import discover_projects, _parse_simple_yaml

        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning("nexus.yaml not found, using defaults")
            config: Dict[str, Any] = {}
        else:
            config = _parse_simple_yaml(config_file.read_text(encoding="utf-8"))

        memory_root = Path(config.get("memory_root", "memory"))
        artifacts_root = Path("artifacts")

        scheduler = cls(memory_root=memory_root, artifacts_root=artifacts_root)

        # Discover projects from nexus_quant/projects/
        projects = discover_projects()
        sched_cfg = config.get("scheduler", {})
        if not isinstance(sched_cfg, dict):
            sched_cfg = {}

        for pname, manifest in projects.items():
            proj_sched = sched_cfg.get(pname, {})
            if not isinstance(proj_sched, dict):
                proj_sched = {}

            enabled = proj_sched.get("enabled", manifest.enabled)
            if not enabled:
                logger.info("Project %s disabled — skipping", pname)
                continue

            interval = proj_sched.get("brain_interval") or manifest.brain_interval
            max_concurrent = proj_sched.get("max_concurrent_runs") or 2

            if not manifest.default_config:
                logger.warning("Project %s has no default_config — skipping", pname)
                continue

            runner = ProjectRunner(
                project_name=pname,
                config_path=manifest.default_config,
                artifacts_dir=artifacts_root,
                memory_root=memory_root,
                market=manifest.market,
                interval=int(interval),
                max_concurrent=int(max_concurrent),
            )
            scheduler.runners[pname] = runner

        return scheduler

    def start(self) -> None:
        """Start all project brain loops in separate threads."""
        logger.info("=== NEXUS Multi-Project Scheduler ===")
        logger.info("Starting %d project(s)", len(self.runners))

        for name, runner in self.runners.items():
            t = threading.Thread(target=runner.run_loop, name=f"nexus-{name}", daemon=True)
            t.start()
            self.threads[name] = t
            logger.info("  [+] %s (interval=%ds, market=%s)", name, runner.interval, runner.market)

        # Start cross-pollination daemon
        self._cross_pollination_thread = threading.Thread(
            target=self._cross_pollination_loop, name="nexus-xpoll", daemon=True
        )
        self._cross_pollination_thread.start()
        logger.info("  [+] Cross-pollination daemon (every 6h)")

    def stop(self) -> None:
        """Gracefully stop all projects."""
        logger.info("Stopping NEXUS Scheduler...")
        self._stop_event.set()
        for name, runner in self.runners.items():
            runner.stop()
        for name, thread in self.threads.items():
            thread.join(timeout=30)
        logger.info("All projects stopped.")

    def status(self) -> Dict[str, Any]:
        """Get status of all running projects."""
        return {
            "scheduler": "running" if not self._stop_event.is_set() else "stopped",
            "project_count": len(self.runners),
            "projects": {name: runner.health() for name, runner in self.runners.items()},
            "threads_alive": {name: t.is_alive() for name, t in self.threads.items()},
        }

    def wait(self) -> None:
        """Block until stopped (Ctrl+C or stop() called)."""
        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(10)
        except KeyboardInterrupt:
            self.stop()

    def _cross_pollination_loop(self) -> None:
        """Periodically scan project memories for insights to promote."""
        from ..memory.namespace import MemoryNamespace

        while not self._stop_event.is_set():
            self._stop_event.wait(21600)  # Every 6 hours
            if self._stop_event.is_set():
                break

            logger.info("[xpoll] Running cross-pollination scan...")
            for name, runner in self.runners.items():
                try:
                    mem = MemoryNamespace(self.memory_root, name, runner.market)
                    candidates = mem.scan_for_promotions()
                    for c in candidates:
                        logger.info("[xpoll] Promote [%s → %s]: %s",
                                    name, c["suggested_target"], c["text"][:80])
                        mem.promote_insight(
                            text=c["text"],
                            from_layer="L2",
                            to_layer=c["suggested_target"],
                            reason="auto-promoted by cross-pollination scan",
                        )
                except Exception as exc:
                    logger.error("[xpoll] Error for %s: %s", name, exc)
