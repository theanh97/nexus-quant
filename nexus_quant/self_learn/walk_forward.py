from __future__ import annotations

"""
Walk-forward parameter optimisation for NEXUS Quant self-learning.

The :class:`WalkForwardOptimizer` partitions the dataset into a sequence of
rolling (train, validation) windows, runs a parameter search on each train
window, then evaluates the best candidate out-of-sample on the validation
window.  The final recommendation is the parameter set that achieves the best
*median* out-of-sample objective across all windows – the most "stable" choice.

Triggering
----------
Call :func:`run_walk_forward` when ``self_learn_cfg["mode"] == "walk_forward"``.
The return dict is compatible with the ``improve_strategy_params`` output format
so that the rest of the pipeline (ledger, memory, Orion) can treat both paths
uniformly.

Config keys (all under ``self_learn_cfg``)
------------------------------------------
mode                  "walk_forward"          (triggers this module)
window_bars           int  (default 720)       train window length in bars
step_bars             int  (default 240)       bars between window starts
validate_fraction     float (default 0.25)     fraction of window used as OOS
trials_per_window     int  (default 15)        candidate trials per window
objective             str  (default "median_calmar")
                          one of: median_calmar | median_sharpe | calmar | sharpe

Stdlib-only; no external dependencies.
"""

import json
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from ..data.schema import MarketDataset
from ..evaluation.benchmark import BenchmarkConfig, run_benchmark_pack_v1
from ..evaluation.metrics import equity_from_returns, summarize
from ..strategies.registry import make_strategy
from .search import _sample_candidate_cfg  # reuse existing sampler


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _round_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Round float values to 6 d.p. for compact, human-readable output."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        out[k] = round(v, 6) if isinstance(v, float) else v
    return out


def _objective_value(summary: Dict[str, Any], objective: str) -> float:
    """Extract the scalar score for *objective* from a metrics summary dict."""
    key = str(objective)
    if key in summary:
        return float(summary[key])
    # Aliases / fallback chain
    aliases = {
        "median_calmar": ["median_calmar", "calmar"],
        "median_sharpe": ["median_sharpe", "sharpe"],
        "calmar":        ["calmar", "median_calmar"],
        "sharpe":        ["sharpe", "median_sharpe"],
    }
    for candidate_key in aliases.get(key, [key]):
        if candidate_key in summary:
            return float(summary[candidate_key])
    return 0.0


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True))
        fh.write("\n")


# ---------------------------------------------------------------------------
# Dataset slicing
# ---------------------------------------------------------------------------


def _slice_dataset(dataset: MarketDataset, start: int, end: int) -> MarketDataset:
    """
    Return a new :class:`MarketDataset` containing only bars ``[start, end)``.

    *start* and *end* are timeline indices (not timestamps).  All aligned
    series (perp_close, spot_close, volume, mark, index, bid, ask) are sliced
    consistently.  The funding dict is filtered to only timestamps that fall
    within ``timeline[start:end]``.

    The sub-dataset fingerprint encodes the parent fingerprint plus the slice
    bounds so that caching / audit trails remain unique.
    """
    tl = dataset.timeline[start:end]
    if not tl:
        raise ValueError(f"Empty slice: start={start}, end={end}")

    ts_set = set(tl)

    def _sl(series: Optional[Dict[str, List[float]]]) -> Optional[Dict[str, List[float]]]:
        if series is None:
            return None
        return {s: series[s][start:end] for s in dataset.symbols}

    funding_slice: Dict[str, Dict[int, float]] = {}
    for sym in dataset.symbols:
        sym_fund = dataset.funding.get(sym) or {}
        funding_slice[sym] = {ts: rate for ts, rate in sym_fund.items() if ts in ts_set}

    # Build a minimal fingerprint that is still unique per slice.
    fp = f"{dataset.fingerprint}[{start}:{end}]"

    return MarketDataset(
        provider=dataset.provider,
        timeline=list(tl),
        symbols=list(dataset.symbols),
        perp_close={s: dataset.perp_close[s][start:end] for s in dataset.symbols},
        spot_close=_sl(dataset.spot_close),
        funding=funding_slice,
        fingerprint=fp,
        perp_volume=_sl(dataset.perp_volume),
        spot_volume=_sl(dataset.spot_volume),
        perp_mark_close=_sl(dataset.perp_mark_close),
        perp_index_close=_sl(dataset.perp_index_close),
        bid_close=_sl(dataset.bid_close),
        ask_close=_sl(dataset.ask_close),
        meta=dict(dataset.meta),
    )


# ---------------------------------------------------------------------------
# Per-window evaluation
# ---------------------------------------------------------------------------


def _eval_on_dataset(
    sub_dataset: MarketDataset,
    engine: BacktestEngine,
    strategy_cfg: Dict[str, Any],
    bench_cfg: BenchmarkConfig,
    objective: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Run one strategy on *sub_dataset* and return a metrics summary enriched
    with windowed medians when the sub-dataset is large enough.
    """
    strategy = make_strategy(strategy_cfg)
    result = engine.run(dataset=sub_dataset, strategy=strategy, seed=seed)

    ppy = float(bench_cfg.periods_per_year)
    eq = result.equity_curve
    rs = result.returns

    summary = summarize(
        returns=rs,
        equity_curve=eq,
        periods_per_year=ppy,
        trades=result.trades,
    )

    # Add windowed medians if the sub-dataset is large enough.
    wf_cfg = getattr(bench_cfg, "walk_forward", None)
    if wf_cfg and getattr(wf_cfg, "enabled", False):
        win = int(getattr(wf_cfg, "window_bars", 0) or 0)
        step = int(getattr(wf_cfg, "step_bars", 0) or 0)
        if win > 0 and step > 0 and len(rs) >= win:
            calmars: List[float] = []
            sharpes: List[float] = []
            for s0 in range(0, max(0, len(rs) - win + 1), step):
                r_sl = rs[s0: s0 + win]
                e_sl = eq[s0: s0 + win + 1]
                w_sum = summarize(r_sl, e_sl, periods_per_year=ppy, trades=None)
                calmars.append(float(w_sum.get("calmar", 0.0)))
                sharpes.append(float(w_sum.get("sharpe", 0.0)))
            if calmars:
                summary["median_calmar"] = statistics.median(calmars)
            if sharpes:
                summary["median_sharpe"] = statistics.median(sharpes)

    return _round_dict(summary)


# ---------------------------------------------------------------------------
# WalkForwardOptimizer
# ---------------------------------------------------------------------------


@dataclass
class _WindowResult:
    """Internal record for one (train, validate) window outcome."""
    window_idx: int
    train_start: int
    train_end: int
    validate_end: int
    best_cfg: Dict[str, Any]
    best_train_score: float
    best_validate_score: float
    validate_summary: Dict[str, Any]
    n_trials: int


class WalkForwardOptimizer:
    """
    Rolling window parameter optimisation.

    For each window ``[train_start : train_end]`` validated on
    ``[train_end : validate_end]``:

    1. Run ``trials_per_window`` candidate trials on the train split.
    2. Select the best candidate by *objective* (train score).
    3. Evaluate that candidate on the validation window (out-of-sample).
    4. Collect OOS performance across all windows.
    5. Select final params = those that appear most often as the per-window
       winner (most *stable* across rolling windows).

    The returned dict is compatible with :func:`~.search.improve_strategy_params`
    so callers can use both interchangeably.
    """

    def __init__(
        self,
        *,
        dataset: MarketDataset,
        engine: BacktestEngine,
        base_strategy_cfg: Dict[str, Any],
        bench_cfg: BenchmarkConfig,
        risk_cfg: Dict[str, Any],
        self_learn_cfg: Dict[str, Any],
        memory_dir: Path,
        run_id: str,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.engine = engine
        self.base_strategy_cfg = base_strategy_cfg
        self.bench_cfg = bench_cfg
        self.risk_cfg = risk_cfg
        self.self_learn_cfg = self_learn_cfg
        self.memory_dir = Path(memory_dir)
        self.run_id = run_id
        self.seed = seed

        # Parse config keys with safe defaults.
        slc = self_learn_cfg or {}
        self.window_bars: int = max(60, int(slc.get("window_bars") or 720))
        self.step_bars: int = max(1, int(slc.get("step_bars") or 240))
        self.validate_fraction: float = min(
            max(float(slc.get("validate_fraction") or 0.25), 0.05), 0.5
        )
        self.trials_per_window: int = max(1, int(slc.get("trials_per_window") or 15))
        self.objective: str = str(slc.get("objective") or "median_calmar")
        self.max_drawdown_gate: float = float(
            (risk_cfg or {}).get("max_drawdown") or 0.35
        )
        self.max_turnover_gate: float = float(
            (risk_cfg or {}).get("max_turnover_per_rebalance") or 2.0
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Execute the walk-forward optimisation and return a result dict that
        mirrors the ``improve_strategy_params`` output format.

        Returns
        -------
        dict with keys:
            accepted            bool
            objective           str
            best_candidate      dict  (strategy name, params, per-window OOS summary)
            baseline            dict  (baseline metrics on full dataset)
            windows_analyzed    int
            stability_score     float  (fraction of windows won by the selected params)
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.memory_dir / "walk_forward_log.jsonl"

        n_bars = len(self.dataset.timeline)
        train_len = self.window_bars
        # How many bars are reserved for validation within each window.
        validate_len = max(
            10,
            int(round(self.window_bars * self.validate_fraction)),
        )
        # The train portion is the remainder.
        inner_train_len = train_len - validate_len

        if inner_train_len < 30:
            return self._not_enough_data(
                reason="train_portion_too_short",
                detail=f"window_bars={self.window_bars}, validate_fraction={self.validate_fraction}",
            )

        # Build window boundaries: each entry is (train_start, train_end, validate_end)
        windows: List[Tuple[int, int, int]] = []
        pos = 0
        while pos + train_len <= n_bars:
            t_start = pos
            t_end = pos + inner_train_len        # exclusive; validate starts here
            v_end = pos + train_len              # exclusive
            windows.append((t_start, t_end, v_end))
            pos += self.step_bars

        if not windows:
            return self._not_enough_data(
                reason="no_windows_fit",
                detail=f"n_bars={n_bars}, window_bars={self.window_bars}",
            )

        # --- Baseline on full dataset ---
        baseline_summary = _eval_on_dataset(
            sub_dataset=self.dataset,
            engine=self.engine,
            strategy_cfg=self.base_strategy_cfg,
            bench_cfg=self.bench_cfg,
            objective=self.objective,
            seed=self.seed,
        )

        # --- Per-window search ---
        window_results: List[_WindowResult] = []
        rng = random.Random(self.seed + 77_777)

        for w_idx, (t_start, t_end, v_end) in enumerate(windows):
            wr = self._run_window(
                window_idx=w_idx,
                train_start=t_start,
                train_end=t_end,
                validate_end=v_end,
                rng=rng,
                log_path=log_path,
            )
            if wr is not None:
                window_results.append(wr)

        if not window_results:
            return self._no_winners(baseline_summary)

        # --- Aggregate: find most stable params ---
        best_cfg, stability_score, oos_scores = self._select_stable_params(window_results)

        # --- Compute aggregate OOS metrics across windows ---
        oos_summaries = [wr.validate_summary for wr in window_results]
        agg_oos = self._aggregate_oos(oos_summaries)

        accepted = stability_score >= 0.4  # win >= 40% of windows to accept

        # Persist summary
        summary_payload = {
            "run_id": self.run_id,
            "accepted_at": _utc_iso(),
            "accepted": accepted,
            "objective": self.objective,
            "windows_analyzed": len(window_results),
            "total_windows_attempted": len(windows),
            "stability_score": round(stability_score, 4),
            "best_params": best_cfg.get("params", {}),
            "strategy": best_cfg.get("name", ""),
            "oos_aggregate": agg_oos,
        }
        summary_path = self.memory_dir / "walk_forward_summary.json"
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8"
        )

        out: Dict[str, Any] = {
            "accepted": accepted,
            "objective": self.objective,
            "best_candidate": {
                "strategy": best_cfg.get("name", ""),
                "params": best_cfg.get("params", {}),
                "oos_aggregate": agg_oos,
                "stability_score": round(stability_score, 4),
                "oos_scores_per_window": [round(s, 6) for s in oos_scores],
            },
            "baseline": {
                "strategy": self.base_strategy_cfg.get("name", ""),
                "params": self.base_strategy_cfg.get("params", {}),
                "full_dataset": baseline_summary,
            },
            "windows_analyzed": len(window_results),
            "stability_score": round(stability_score, 4),
            "mode": "walk_forward",
            "config": {
                "window_bars": self.window_bars,
                "step_bars": self.step_bars,
                "validate_fraction": self.validate_fraction,
                "trials_per_window": self.trials_per_window,
                "objective": self.objective,
            },
        }
        return out

    # ------------------------------------------------------------------
    # Private: single window
    # ------------------------------------------------------------------

    def _run_window(
        self,
        *,
        window_idx: int,
        train_start: int,
        train_end: int,
        validate_end: int,
        rng: random.Random,
        log_path: Path,
    ) -> Optional[_WindowResult]:
        """
        Run param search on the train slice, then evaluate the best on
        the validate slice.  Returns None if no viable candidate is found.
        """
        try:
            train_ds = _slice_dataset(self.dataset, train_start, train_end)
        except ValueError:
            return None

        try:
            val_ds = _slice_dataset(self.dataset, train_end, validate_end)
        except ValueError:
            return None

        if len(train_ds.timeline) < 20 or len(val_ds.timeline) < 5:
            return None

        best_train_score = float("-inf")
        best_cfg_for_window: Optional[Dict[str, Any]] = None

        for t in range(self.trials_per_window):
            # Reuse the existing candidate sampler from search.py; no priors
            # here (walk-forward is about stability, not exploitation of past
            # accepted params).
            cand_cfg = _sample_candidate_cfg(
                base_strategy_cfg=self.base_strategy_cfg,
                dataset=train_ds,
                rng=rng,
                priors=None,
                exploit_prob=0.0,
            )

            try:
                train_summary = _eval_on_dataset(
                    sub_dataset=train_ds,
                    engine=self.engine,
                    strategy_cfg=cand_cfg,
                    bench_cfg=self.bench_cfg,
                    objective=self.objective,
                    seed=self.seed + window_idx * 1000 + t,
                )
            except Exception:
                continue

            # Apply risk gates on the train window.
            mdd = float(train_summary.get("max_drawdown", 1.0))
            turn_max = float(train_summary.get("turnover_max", 999.0))
            if mdd > self.max_drawdown_gate:
                continue
            if turn_max > self.max_turnover_gate:
                continue

            score = _objective_value(train_summary, self.objective)
            if score > best_train_score:
                best_train_score = score
                best_cfg_for_window = cand_cfg

        if best_cfg_for_window is None:
            return None

        # Evaluate best train candidate on validation window (OOS).
        try:
            val_summary = _eval_on_dataset(
                sub_dataset=val_ds,
                engine=self.engine,
                strategy_cfg=best_cfg_for_window,
                bench_cfg=self.bench_cfg,
                objective=self.objective,
                seed=self.seed + window_idx * 1000 + 9_999,
            )
        except Exception:
            return None

        val_score = _objective_value(val_summary, self.objective)

        record = {
            "window_idx": window_idx,
            "train_start": train_start,
            "train_end": train_end,
            "validate_end": validate_end,
            "n_trials": self.trials_per_window,
            "best_train_score": round(best_train_score, 6),
            "validate_score": round(val_score, 6),
            "best_params": best_cfg_for_window.get("params", {}),
            "strategy": best_cfg_for_window.get("name", ""),
        }
        _append_jsonl(log_path, record)

        return _WindowResult(
            window_idx=window_idx,
            train_start=train_start,
            train_end=train_end,
            validate_end=validate_end,
            best_cfg=best_cfg_for_window,
            best_train_score=best_train_score,
            best_validate_score=val_score,
            validate_summary=val_summary,
            n_trials=self.trials_per_window,
        )

    # ------------------------------------------------------------------
    # Private: stability aggregation
    # ------------------------------------------------------------------

    def _select_stable_params(
        self,
        window_results: List[_WindowResult],
    ) -> Tuple[Dict[str, Any], float, List[float]]:
        """
        Identify which candidate param set wins the most windows (highest OOS
        score) and return ``(best_cfg, stability_score, oos_scores_for_that_cfg)``.

        *stability_score* is the fraction of windows in which this config
        produced the best OOS objective value relative to all other windows'
        best candidates.

        Strategy: since each window independently selects the best train
        candidate, we group windows by their winning cfg (serialised to JSON
        as a canonical key) and count how often each cfg was chosen.  The cfg
        with the highest count wins.  Ties are broken by median OOS score.
        """
        # Map: canonical_params_json -> list of (window_idx, oos_score, cfg)
        groups: Dict[str, List[Tuple[int, float, Dict[str, Any]]]] = {}
        for wr in window_results:
            key = json.dumps(wr.best_cfg.get("params", {}), sort_keys=True)
            groups.setdefault(key, []).append(
                (wr.window_idx, wr.best_validate_score, wr.best_cfg)
            )

        best_key: Optional[str] = None
        best_count = -1
        best_med_oos = float("-inf")

        for k, entries in groups.items():
            count = len(entries)
            med_oos = statistics.median(e[1] for e in entries)
            if count > best_count or (count == best_count and med_oos > best_med_oos):
                best_count = count
                best_med_oos = med_oos
                best_key = k

        # Collect OOS scores for the winning cfg (across all windows where it
        # appeared), then also compute per-window OOS scores for the selected cfg.
        winning_entries = groups.get(best_key or "", [])
        winning_cfg = winning_entries[0][2] if winning_entries else self.base_strategy_cfg
        oos_scores = [e[1] for e in winning_entries]

        stability_score = float(best_count) / float(len(window_results))
        return (winning_cfg, stability_score, oos_scores)

    # ------------------------------------------------------------------
    # Private: OOS aggregation across windows
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_oos(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute median, mean, and min across key OOS metrics over all windows.
        Returns a compact dict suitable for JSON serialisation.
        """
        if not summaries:
            return {}

        keys = ["calmar", "sharpe", "sortino", "max_drawdown", "cagr", "total_return",
                "median_calmar", "median_sharpe"]
        out: Dict[str, Any] = {"n_windows": len(summaries)}
        for k in keys:
            vals = [float(s[k]) for s in summaries if k in s]
            if vals:
                out[f"{k}_median"] = round(statistics.median(vals), 6)
                out[f"{k}_mean"] = round(statistics.mean(vals), 6)
                out[f"{k}_min"] = round(min(vals), 6)
                out[f"{k}_max"] = round(max(vals), 6)
        return out

    # ------------------------------------------------------------------
    # Private: failure stubs (compatible output shape)
    # ------------------------------------------------------------------

    def _not_enough_data(self, *, reason: str, detail: str) -> Dict[str, Any]:
        return {
            "accepted": False,
            "objective": self.objective,
            "best_candidate": {"reason": reason, "detail": detail},
            "baseline": {
                "strategy": self.base_strategy_cfg.get("name", ""),
                "params": self.base_strategy_cfg.get("params", {}),
            },
            "windows_analyzed": 0,
            "stability_score": 0.0,
            "mode": "walk_forward",
            "config": {
                "window_bars": self.window_bars,
                "step_bars": self.step_bars,
                "validate_fraction": self.validate_fraction,
                "trials_per_window": self.trials_per_window,
                "objective": self.objective,
            },
        }

    def _no_winners(self, baseline_summary: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "accepted": False,
            "objective": self.objective,
            "best_candidate": {"reason": "no_window_produced_a_valid_candidate"},
            "baseline": {
                "strategy": self.base_strategy_cfg.get("name", ""),
                "params": self.base_strategy_cfg.get("params", {}),
                "full_dataset": baseline_summary,
            },
            "windows_analyzed": 0,
            "stability_score": 0.0,
            "mode": "walk_forward",
            "config": {
                "window_bars": self.window_bars,
                "step_bars": self.step_bars,
                "validate_fraction": self.validate_fraction,
                "trials_per_window": self.trials_per_window,
                "objective": self.objective,
            },
        }


# ---------------------------------------------------------------------------
# Module-level convenience entry point
# ---------------------------------------------------------------------------


def run_walk_forward(
    *,
    dataset: MarketDataset,
    engine: BacktestEngine,
    base_strategy_cfg: Dict[str, Any],
    bench_cfg: BenchmarkConfig,
    risk_cfg: Dict[str, Any],
    self_learn_cfg: Dict[str, Any],
    memory_dir: Path,
    run_id: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Convenience wrapper: instantiate :class:`WalkForwardOptimizer` and call
    :meth:`~WalkForwardOptimizer.run`.

    This is the primary public interface for the walk-forward mode.  The
    returned dict is compatible with the ``improve_strategy_params`` output
    format used throughout the rest of the pipeline.

    Parameters
    ----------
    dataset:
        Full aligned market dataset.
    engine:
        Configured :class:`~nexus_quant.backtest.engine.BacktestEngine`.
    base_strategy_cfg:
        Dict with ``"name"`` and ``"params"`` keys for the strategy to optimise.
    bench_cfg:
        :class:`~nexus_quant.evaluation.benchmark.BenchmarkConfig`.
    risk_cfg:
        Risk gate config (``max_drawdown``, ``max_turnover_per_rebalance``).
    self_learn_cfg:
        Self-learn config section; must include ``"mode": "walk_forward"``.
        See module docstring for all recognised keys.
    memory_dir:
        Directory where logs and the final summary are written.
    run_id:
        Unique run identifier threaded through for traceability.
    seed:
        Master RNG seed.

    Returns
    -------
    dict
        Walk-forward result with keys ``accepted``, ``objective``,
        ``best_candidate``, ``baseline``, ``windows_analyzed``,
        ``stability_score``.
    """
    optimizer = WalkForwardOptimizer(
        dataset=dataset,
        engine=engine,
        base_strategy_cfg=base_strategy_cfg,
        bench_cfg=bench_cfg,
        risk_cfg=risk_cfg,
        self_learn_cfg=self_learn_cfg,
        memory_dir=memory_dir,
        run_id=run_id,
        seed=seed,
    )
    return optimizer.run()
