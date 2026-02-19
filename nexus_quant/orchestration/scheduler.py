"""
ExperimentScheduler: Runs N experiments in parallel using ThreadPoolExecutor.

Each experiment is a config override dict applied to a base config.
Results are aggregated into a consolidated ablation artifact.

Usage in Orion:
    scheduler = ExperimentScheduler(
        config_path=config_path,
        artifacts_dir=artifacts_dir,
        max_workers=4,
    )
    results = scheduler.run_batch(experiments)  # List[ExperimentSpec]
"""
from __future__ import annotations

import json
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentSpec:
    """Single experiment specification."""
    experiment_id: str
    config_overrides: Dict[str, Any]
    priority: int = 5      # 1=highest, 10=lowest
    why: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    experiment_id: str
    run_id: Optional[str]
    config_overrides: Dict[str, Any]
    metrics: Dict[str, Any]
    verdict: Dict[str, Any]
    error: Optional[str]
    duration_seconds: float
    why: str


class ExperimentScheduler:
    def __init__(
        self,
        config_path: Path,
        artifacts_dir: Path,
        max_workers: int = 4,
    ) -> None:
        self.config_path = config_path
        self.artifacts_dir = artifacts_dir
        self.max_workers = max(1, min(int(max_workers), 16))
        self._lock = threading.Lock()

    def run_batch(
        self,
        experiments: List[ExperimentSpec],
    ) -> List[ExperimentResult]:
        """Run all experiments in parallel, return results sorted by priority."""
        if not experiments:
            return []
        sorted_exps = sorted(experiments, key=lambda e: e.priority)
        results: List[ExperimentResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._run_one, exp): exp
                for exp in sorted_exps
            }
            for future in as_completed(futures):
                result = future.result()
                with self._lock:
                    results.append(result)
        results.sort(key=lambda r: r.experiment_id)
        return results

    def _run_one(self, spec: ExperimentSpec) -> ExperimentResult:
        """Run a single experiment (called in thread)."""
        import time
        from ..run import run_one
        start = time.time()
        try:
            run_id = run_one(
                self.config_path,
                out_dir=self.artifacts_dir,
                cfg_override=spec.config_overrides,
            )
            # Load metrics from run artifacts
            metrics, verdict = self._load_run_metrics(run_id)
            return ExperimentResult(
                experiment_id=spec.experiment_id,
                run_id=run_id,
                config_overrides=spec.config_overrides,
                metrics=metrics,
                verdict=verdict,
                error=None,
                duration_seconds=round(time.time() - start, 2),
                why=spec.why,
            )
        except Exception as e:
            return ExperimentResult(
                experiment_id=spec.experiment_id,
                run_id=None,
                config_overrides=spec.config_overrides,
                metrics={},
                verdict={"pass": False, "reasons": [str(e)]},
                error=str(e),
                duration_seconds=round(time.time() - start, 2),
                why=spec.why,
            )

    def _load_run_metrics(self, run_id: str) -> tuple[dict, dict]:
        """Load metrics.json from run artifacts."""
        runs_dir = self.artifacts_dir / "runs"
        run_dir = runs_dir / run_id
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text("utf-8"))
                return data.get("summary") or {}, data.get("verdict") or {}
            except Exception:
                pass
        return {}, {}

    def write_batch_report(
        self,
        results: List[ExperimentResult],
        report_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Write consolidated batch report as JSON + MD."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if report_path is None:
            report_dir = self.artifacts_dir / "wisdom" / "batch_runs"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"batch.{ts}.json"

        passed = [r for r in results if r.verdict.get("pass")]
        failed = [r for r in results if not r.verdict.get("pass")]

        # Sort passed by Sharpe
        def _sharpe(r: ExperimentResult) -> float:
            return float(r.metrics.get("sharpe") or 0.0)
        passed_sorted = sorted(passed, key=_sharpe, reverse=True)

        report = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "total": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "best": {
                "experiment_id": passed_sorted[0].experiment_id if passed_sorted else None,
                "metrics": passed_sorted[0].metrics if passed_sorted else {},
                "config_overrides": passed_sorted[0].config_overrides if passed_sorted else {},
                "why": passed_sorted[0].why if passed_sorted else "",
            },
            "results": [
                {
                    "id": r.experiment_id,
                    "run_id": r.run_id,
                    "pass": r.verdict.get("pass"),
                    "sharpe": float(r.metrics.get("sharpe") or 0.0),
                    "calmar": float(r.metrics.get("calmar") or 0.0),
                    "mdd": float(r.metrics.get("max_drawdown") or 0.0),
                    "error": r.error,
                    "duration_s": r.duration_seconds,
                    "why": r.why,
                }
                for r in results
            ],
        }

        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        # Write MD summary
        md_path = report_path.with_suffix(".md")
        lines = [
            "# Batch Experiment Report",
            f"- ts: `{report['ts']}`",
            f"- total: {len(results)}, passed: {len(passed)}, failed: {len(failed)}",
            "",
            "## Best Experiment",
        ]
        if passed_sorted:
            best = passed_sorted[0]
            lines += [
                f"- id: `{best.experiment_id}`",
                f"- sharpe: `{best.metrics.get('sharpe')}`  calmar: `{best.metrics.get('calmar')}`",
                f"- why: {best.why}",
                f"- overrides: `{json.dumps(best.config_overrides)}`",
            ]
        lines += ["", "## All Results"]
        for r in results:
            status = "✓" if r.verdict.get("pass") else "✗"
            lines.append(f"- {status} `{r.experiment_id}` sharpe={r.metrics.get('sharpe')} err={r.error or ''}")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return {"report_path": str(report_path), "md_path": str(md_path), **{k: report[k] for k in ("total", "passed", "failed")}}
