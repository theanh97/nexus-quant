"""
NEXUS Alpha Loop -- Continuous Strategy Improvement Engine.

Runs all strategy configs in sequence, compares results,
promotes the best-performing strategy, and auto-tunes params.

Features:
- Runs all binance configs in configs/ directory
- Compares corrected Sharpe across strategies
- Promotes winner to champion in artifacts/champion.json
- Generates improvement suggestions via LLM (GLM-5)
- Runs every N seconds in autonomous mode
- Writes summary to ledger + reports to console

Usage:
  python3 -m nexus_quant alpha_loop --artifacts artifacts --interval 3600
  python3 -m nexus_quant alpha_loop --artifacts artifacts --interval 3600 --max-cycles 5
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..data.provider_policy import classify_provider

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Periods-per-year reference map
# ---------------------------------------------------------------------------
_PPY_MAP: Dict[str, float] = {
    "1m": 525600.0, "5m": 105120.0, "15m": 35040.0, "30m": 17520.0,
    "1h": 8760.0, "2h": 4380.0, "4h": 2190.0, "8h": 1095.0,
    "1d": 365.0, "1w": 52.0,
}


def _corrected_sharpe(metrics: Dict[str, Any], bar_interval: str = "1h") -> float:
    """Return annualised Sharpe from metrics summary, clamping non-finite values.

    The backtest engine already annualises correctly using ppy derived from
    bar_interval, so no further sqrt-correction is applied here.
    """
    summary = metrics.get("summary") or {}
    raw = float(summary.get("sharpe") or 0.0)
    return raw if math.isfinite(raw) else 0.0


# ---------------------------------------------------------------------------
# LLM helper (GLM-5 via ZAI, graceful fallback)
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, max_tokens: int = 600) -> str:
    """Call GLM-5 via ZAI gateway. Falls back to deterministic text on error."""
    api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = (
        os.environ.get("ZAI_ANTHROPIC_BASE_URL") or "https://api.z.ai/api/anthropic"
    ).rstrip("/")
    model = os.environ.get("ZAI_DEFAULT_MODEL", "glm-5")
    if not api_key:
        return "[alpha_loop] LLM unavailable: no API key set."
    try:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=(
                "You are NEXUS, an autonomous quant trading research engine. "
                "Analyse backtest results and suggest concrete, data-driven param tweaks. "
                "Be specific, concise, and use bullet points."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""
    except Exception as exc:  # pylint: disable=broad-except
        return f"[LLM error: {exc}]"


# ---------------------------------------------------------------------------
# AlphaLoop
# ---------------------------------------------------------------------------

class AlphaLoop:
    """Continuous alpha discovery and strategy improvement engine."""

    _BINANCE_PREFIXES = ("run_binance_",)

    def __init__(
        self,
        artifacts_dir: Path,
        configs_dir: Path,
        interval_seconds: int = 3600,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.configs_dir = Path(configs_dir)
        self.interval_seconds = max(60, int(interval_seconds))
        self.champion_path = self.artifacts_dir / "champion.json"
        self.ledger_path = self.artifacts_dir / "ledger" / "ledger.jsonl"
        self._cycle_count = 0
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    # --- config discovery ---------------------------------------------------

    def _binance_configs(self) -> List[Path]:
        """Return sorted list of binance config paths."""
        return [
            p for p in sorted(self.configs_dir.glob("*.json"))
            if any(p.name.startswith(pfx) for pfx in self._BINANCE_PREFIXES)
        ]

    # --- backtest runner ----------------------------------------------------

    def _run_config(self, config_path: Path) -> Optional[str]:
        """Run one backtest via subprocess; return run_id or None on failure."""
        cmd = [
            sys.executable, "-m", "nexus_quant", "run",
            "--config", str(config_path),
            "--out", str(self.artifacts_dir),
        ]
        log.info("[alpha_loop] Running: %s", config_path.name)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(self.configs_dir.parent),
            )
            if proc.returncode != 0:
                log.warning(
                    "[alpha_loop] %s failed (rc=%d)\n%s",
                    config_path.name, proc.returncode, (proc.stderr or "")[-400:],
                )
                return None
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            run_id = lines[-1] if lines else None
            log.info("[alpha_loop] %s -> %s", config_path.name, run_id)
            return run_id
        except subprocess.TimeoutExpired:
            log.error("[alpha_loop] %s timed out after 600 s", config_path.name)
        except Exception as exc:  # pylint: disable=broad-except
            log.error("[alpha_loop] %s error: %s", config_path.name, exc)
        return None

    # --- metrics / config loaders -------------------------------------------

    def _load_run_metrics(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load metrics.json from a run directory; return dict or None."""
        p = self.artifacts_dir / "runs" / run_id / "metrics.json"
        if not p.exists():
            log.warning("[alpha_loop] metrics.json missing for %s", run_id)
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # pylint: disable=broad-except
            log.error("[alpha_loop] parse metrics %s: %s", run_id, exc)
            return None

    def _load_run_config(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load config.json from a run directory; return dict or None."""
        p = self.artifacts_dir / "runs" / run_id / "config.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # pylint: disable=broad-except
            log.error("[alpha_loop] parse config %s: %s", run_id, exc)
            return None

    # --- champion management ------------------------------------------------

    def get_champion(self) -> Dict[str, Any]:
        """Read artifacts/champion.json; return {} if missing."""
        if not self.champion_path.exists():
            return {}
        try:
            return json.loads(self.champion_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("[alpha_loop] champion.json read error: %s", exc)
            return {}

    def set_champion(
        self,
        run_id: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        """Save champion to artifacts/champion.json."""
        bar_interval = str((config.get("data") or {}).get("bar_interval") or "1h")
        doc = {
            "run_id": run_id,
            "run_name": config.get("run_name"),
            "strategy": (config.get("strategy") or {}).get("name"),
            "params": (config.get("strategy") or {}).get("params") or {},
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "corrected_sharpe": _corrected_sharpe(metrics, bar_interval),
            "summary": metrics.get("summary") or {},
            "bias_check": metrics.get("bias_check") or {},
            "verdict": metrics.get("verdict") or {},
            "walk_forward_stability": (
                (metrics.get("walk_forward") or {}).get("stability") or {}
            ),
        }
        self.champion_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        log.info(
            "[alpha_loop] Champion promoted: %s (sharpe=%.4f)",
            doc["run_name"],
            doc["corrected_sharpe"],
        )

    # --- compare strategies (scan all artifacts/runs/) ----------------------

    def compare_strategies(self) -> List[Dict[str, Any]]:
        """Scan artifacts/runs/ and return completed runs sorted by corrected Sharpe desc."""
        runs_dir = self.artifacts_dir / "runs"
        if not runs_dir.exists():
            return []
        results: List[Dict[str, Any]] = []
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            mf = run_dir / "metrics.json"
            cf = run_dir / "config.json"
            if not mf.exists() or not cf.exists():
                continue
            try:
                metrics = json.loads(mf.read_text(encoding="utf-8"))
                config = json.loads(cf.read_text(encoding="utf-8"))
                provider = str((config.get("data") or {}).get("provider") or "")
                provider_class = classify_provider(provider)
                if provider_class != "real":
                    continue
                bar_iv = str((config.get("data") or {}).get("bar_interval") or "1h")
                sm = metrics.get("summary") or {}
                results.append({
                    "run_id": run_dir.name,
                    "run_name": config.get("run_name"),
                    "strategy": (config.get("strategy") or {}).get("name"),
                    "data_provider": provider,
                    "data_provider_class": provider_class,
                    "corrected_sharpe": _corrected_sharpe(metrics, bar_iv),
                    "sharpe": float(sm.get("sharpe") or 0.0),
                    "calmar": float(sm.get("calmar") or 0.0),
                    "max_drawdown": float(sm.get("max_drawdown") or 0.0),
                    "cagr": float(sm.get("cagr") or 0.0),
                    "verdict_pass": bool((metrics.get("verdict") or {}).get("pass")),
                    "wf_pct_profitable": float(
                        (metrics.get("walk_forward") or {})
                        .get("stability", {})
                        .get("fraction_profitable") or 0.0
                    ),
                })
            except Exception as exc:  # pylint: disable=broad-except
                log.debug("[alpha_loop] skip %s: %s", run_dir.name, exc)
        results.sort(key=lambda x: x["corrected_sharpe"], reverse=True)
        return results

    # --- LLM improvement suggestions ----------------------------------------

    def suggest_improvements(self) -> List[str]:
        """Ask GLM-5 for param tweaks. Returns list of suggestion strings."""
        champion = self.get_champion()
        top3 = self.compare_strategies()[:3]
        prompt = (
            "I am NEXUS, a quant trading AI. My current champion strategy:\n"
            + json.dumps(champion, indent=2)
            + "\n\nTop 3 strategies by corrected Sharpe:\n"
            + json.dumps(top3, indent=2)
            + "\n\nSuggest 3-5 specific parameter tweaks for the NEXT backtest cycle "
            + "to improve out-of-sample Sharpe. For each: param name, "
            + "current -> proposed value, 1-sentence quant rationale. "
            + "Do not suggest changing the strategy class."
        )
        raw = _call_llm(prompt, max_tokens=600)
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # --- ledger append ------------------------------------------------------

    def _append_ledger(self, event: Dict[str, Any]) -> None:
        """Append a JSONL event to the ledger."""
        try:
            with open(self.ledger_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("[alpha_loop] ledger write error: %s", exc)

    # --- core cycle ---------------------------------------------------------

    def run_one_cycle(self) -> Dict[str, Any]:
        """
        One alpha discovery cycle:
        1. Discover all binance configs
        2. Run each backtest (subprocess)
        3. Compare results
        4. Promote champion if Sharpe improved
        5. LLM improvement suggestions
        6. Write ledger event
        7. Return cycle summary dict
        """
        self._cycle_count += 1
        cycle_id = self._cycle_count
        started_at = datetime.now(timezone.utc).isoformat()
        log.info("[alpha_loop] === Cycle %d started at %s ===", cycle_id, started_at)

        configs = self._binance_configs()
        if not configs:
            log.warning("[alpha_loop] No binance configs found in %s", self.configs_dir)
            return {"cycle": cycle_id, "error": "no_configs", "started_at": started_at}

        run_results: List[Dict[str, Any]] = []
        for cfg_path in configs:
            run_id = self._run_config(cfg_path)
            if run_id is None:
                run_results.append({"config": cfg_path.name, "status": "failed"})
                continue
            metrics = self._load_run_metrics(run_id)
            cfg_data = self._load_run_config(run_id)
            if metrics is None or cfg_data is None:
                run_results.append({"config": cfg_path.name, "status": "no_metrics", "run_id": run_id})
                continue
            bar_iv = str((cfg_data.get("data") or {}).get("bar_interval") or "1h")
            sm = metrics.get("summary") or {}
            run_results.append({
                "config": cfg_path.name,
                "status": "ok",
                "run_id": run_id,
                "run_name": cfg_data.get("run_name"),
                "strategy": (cfg_data.get("strategy") or {}).get("name"),
                "corrected_sharpe": _corrected_sharpe(metrics, bar_iv),
                "sharpe": float(sm.get("sharpe") or 0.0),
                "calmar": float(sm.get("calmar") or 0.0),
                "cagr": float(sm.get("cagr") or 0.0),
                "max_drawdown": float(sm.get("max_drawdown") or 0.0),
                "verdict_pass": bool((metrics.get("verdict") or {}).get("pass")),
                "_metrics": metrics,
                "_config": cfg_data,
            })

        ok_runs = sorted(
            [r for r in run_results if r["status"] == "ok"],
            key=lambda x: x["corrected_sharpe"],
            reverse=True,
        )

        new_champion_promoted = False
        champion_run: Optional[Dict[str, Any]] = None
        if ok_runs:
            best = ok_runs[0]
            prev_sharpe = float(self.get_champion().get("corrected_sharpe") or -999.0)
            if best["corrected_sharpe"] > prev_sharpe:
                self.set_champion(
                    run_id=best["run_id"],
                    config=best["_config"],
                    metrics=best["_metrics"],
                )
                new_champion_promoted = True
            champion_run = best

        suggestions: List[str] = []
        try:
            suggestions = self.suggest_improvements()
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("[alpha_loop] suggest_improvements error: %s", exc)

        cycle_summary: Dict[str, Any] = {
            "cycle": cycle_id,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "configs_run": len(configs),
            "runs_ok": len(ok_runs),
            "runs_failed": len(run_results) - len(ok_runs),
            "new_champion_promoted": new_champion_promoted,
            "champion": {
                k: v for k, v in (champion_run or {}).items()
                if not k.startswith("_")
            },
            "rankings": [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in ok_runs
            ],
            "improvement_suggestions": suggestions,
        }

        self._append_ledger({
            "ts": cycle_summary["finished_at"],
            "kind": "alpha_loop_cycle",
            "run_id": (champion_run or {}).get("run_id"),
            "run_name": "alpha_loop",
            "payload": cycle_summary,
        })

        self._print_cycle_report(cycle_summary)
        return cycle_summary

    # --- console report -----------------------------------------------------

    @staticmethod
    def _print_cycle_report(s: Dict[str, Any]) -> None:
        sep = "-" * 64
        print(sep)
        print("[AlphaLoop] Cycle", s["cycle"], "completed @", s["finished_at"])
        print("  Configs run :", s["configs_run"])
        print("  OK / Failed :", s["runs_ok"], "/", s["runs_failed"])
        c = s.get("champion") or {}
        if c:
            print(
                "  Champion    :", c.get("run_name"),
                "| strategy=", c.get("strategy"),
                "| sharpe={:.4f}".format(c.get("corrected_sharpe", 0)),
                "| calmar={:.4f}".format(c.get("calmar", 0)),
                "| cagr={:.4f}".format(c.get("cagr", 0)),
                "| mdd={:.4f}".format(c.get("max_drawdown", 0)),
                "| verdict=", "PASS" if c.get("verdict_pass") else "FAIL",
            )
            print("  New champ?  :", s["new_champion_promoted"])
        print("  Rankings:")
        for i, r in enumerate(s.get("rankings") or [], 1):
            print(
                "   ", i, ".",
                "{:<38}".format(str(r.get("run_name") or "")),
                "sharpe={:+.4f}".format(r.get("corrected_sharpe", 0)),
                "mdd={:.3f}".format(r.get("max_drawdown", 0)),
                "PASS" if r.get("verdict_pass") else "FAIL",
            )
        sug = s.get("improvement_suggestions") or []
        if sug:
            print("  LLM Suggestions:")
            for sg in sug[:5]:
                print("    -", sg)
        print(sep)

    # --- continuous loop ----------------------------------------------------

    def run_loop(self, max_cycles: Optional[int] = None) -> None:
        """Run the alpha loop continuously, sleeping interval_seconds between cycles."""
        log.info(
            "[alpha_loop] Starting loop (interval=%ds, max_cycles=%s)",
            self.interval_seconds,
            max_cycles or "infinite",
        )
        cycle = 0
        while True:
            try:
                self.run_one_cycle()
                cycle += 1
            except KeyboardInterrupt:
                log.info("[alpha_loop] Interrupted by user after %d cycles.", cycle)
                break
            except Exception as exc:  # pylint: disable=broad-except
                log.error(
                    "[alpha_loop] Unhandled error in cycle %d: %s",
                    cycle, exc, exc_info=True,
                )
                time.sleep(min(60, self.interval_seconds // 4))
                cycle += 1
            if max_cycles and cycle >= max_cycles:
                log.info("[alpha_loop] max_cycles=%d reached. Exiting.", max_cycles)
                break
            log.info("[alpha_loop] Sleeping %ds before next cycle...", self.interval_seconds)
            time.sleep(self.interval_seconds)


# ---------------------------------------------------------------------------
# CLI entry point  (wired into nexus_quant cli.py as alpha_loop subcommand)
# ---------------------------------------------------------------------------

def alpha_loop_main(
    artifacts_dir: Path,
    configs_dir: Optional[Path] = None,
    interval_seconds: int = 3600,
    max_cycles: int = 0,
) -> int:
    """
    Entry point for: python3 -m nexus_quant alpha_loop

    Args:
        artifacts_dir: Artifacts output directory.
        configs_dir: Configs directory (defaults to artifacts_dir.parent/configs).
        interval_seconds: Seconds to sleep between cycles.
        max_cycles: Max cycles to run (0 = infinite).

    Returns:
        Shell exit code (0 = OK).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    artifacts_dir = Path(artifacts_dir)
    if configs_dir is None:
        configs_dir = artifacts_dir.parent / "configs"
    configs_dir = Path(configs_dir)

    if not configs_dir.exists():
        print(
            f"[alpha_loop] ERROR: configs_dir not found: {configs_dir}",
            file=sys.stderr,
        )
        return 1

    loop = AlphaLoop(
        artifacts_dir=artifacts_dir,
        configs_dir=configs_dir,
        interval_seconds=interval_seconds,
    )

    if max_cycles == 1:
        loop.run_one_cycle()
    else:
        loop.run_loop(max_cycles=max_cycles if max_cycles > 0 else None)

    return 0
