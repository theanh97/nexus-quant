"""
Options P&L Engine — Proper Gamma/Theta Model

Economic model for short-vol (short straddle / delta-neutral) positions:

For a delta-hedged short straddle, daily PnL decomposition:
    PnL_bar = Theta_income - Gamma_cost
    where:
        Theta_income  ~ IV_atm² * leverage * dt / 2  [option time decay collected]
        Gamma_cost    ~ 0.5 * Gamma * (ΔS)² / S²   [delta-hedge rebalancing cost]
        Net           ≈ 0.5 * (IV² - RV_bar²) * leverage * dt

VRP return per bar (no SCALE factor — proper calibration):
    VRP_pnl_bar = 0.5 * (IV² - RV_bar_ann²) * dt * |weight|

where:
    IV         = implied vol for this bar (annualized, e.g. 0.76 for 76%)
    RV_bar_ann = realized vol for single bar = |log_ret| * sqrt(bars_per_year)
    dt         = 1 / bars_per_year

Expected annual gross return at 1x leverage:
    Theta income ≈ 0.5 * IV² ≈ 28.9% (BTC IV=76%)
    Gamma cost   ≈ 0.5 * E[RV²] ≈ 11.4% (avg daily move 2.5%)
    Net VRP      ≈ 17.5% gross → Sharpe ~1.0-1.8 after costs

This correctly:
    - Always positive in expectation when IV > RV (VRP thesis)
    - Has volatility proportional to gamma (vol of vol)
    - Suffers large losses during vol spikes (RV >> IV)

Usage:
    from nexus_quant.projects.crypto_options.options_engine import OptionsBacktestEngine
    engine = OptionsBacktestEngine(cost_model, bars_per_year=365)
    result = engine.run(dataset, strategy)
    metrics = compute_metrics(result.equity_curve, result.returns, bars_per_year=365)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from nexus_quant.backtest.engine import BacktestConfig, BacktestResult
from nexus_quant.backtest.costs import ExecutionCostModel
from nexus_quant.data.schema import MarketDataset
from nexus_quant.strategies.base import Strategy, Weights


class OptionsBacktestEngine:
    """Backtest engine for delta-neutral options strategies.

    Replaces price-based PnL with proper variance risk premium PnL:
        PnL_bar = |weight| * 0.5 * (IV² - RV_bar²) * dt

    For VRP strategy (short vol): weight < 0, so:
        PnL = (-weight) * 0.5 * (IV² - RV_bar²) * dt
    Positive when IV > RV_bar (expected outcome for short vol)

    For directional strategies (skew MR, term structure):
    Falls back to standard price PnL (delta exposure on underlying).
    """

    def __init__(
        self,
        costs: ExecutionCostModel,
        bars_per_year: int = 365,
        use_options_pnl: bool = True,
        iv_smooth_bars: int = 3,
        skew_sensitivity_mult: float = 1.0,
    ) -> None:
        self.costs = costs
        self.bars_per_year = bars_per_year
        self.use_options_pnl = use_options_pnl
        self.iv_smooth_bars = iv_smooth_bars
        self.skew_sensitivity_mult = skew_sensitivity_mult

        # Build base config for the standard engine fallback
        self._base_cfg = BacktestConfig(costs=costs)

    def run(
        self, dataset: MarketDataset, strategy: Strategy, seed: int = 0
    ) -> BacktestResult:
        """Run options backtest.

        VRP strategy: gamma/theta model.
        Skew strategy: vega/skew-change model.
        All other strategies: standard price PnL (delta proxy).
        """
        strategy_name = (strategy.name or "").lower()
        is_vrp = "vrp" in strategy_name or "variance" in strategy_name
        is_skew = "skew" in strategy_name
        is_term = "term" in strategy_name
        is_butterfly = "butterfly" in strategy_name
        is_iv_mr = "iv_mr" in strategy_name
        is_pcr = "pcr" in strategy_name or "put_call" in strategy_name

        if self.use_options_pnl and is_vrp:
            return self._run_vrp(dataset, strategy)
        elif self.use_options_pnl and is_skew:
            return self._run_skew(dataset, strategy, feature_key="skew_25d")
        elif self.use_options_pnl and is_term:
            return self._run_skew(dataset, strategy, feature_key="term_spread")
        elif self.use_options_pnl and is_butterfly:
            return self._run_skew(dataset, strategy, feature_key="butterfly_25d")
        elif self.use_options_pnl and is_iv_mr:
            return self._run_skew(dataset, strategy, feature_key="iv_atm")
        elif self.use_options_pnl and is_pcr:
            return self._run_skew(dataset, strategy, feature_key="put_call_ratio")
        else:
            from nexus_quant.backtest.engine import BacktestEngine
            return BacktestEngine(self._base_cfg).run(dataset, strategy, seed)

    def _run_vrp(self, dataset: MarketDataset, strategy: Strategy) -> BacktestResult:
        """Run VRP strategy with proper options PnL (gamma/theta model)."""
        syms = dataset.symbols
        n = len(dataset.timeline)
        dt = 1.0 / self.bars_per_year

        equity = 1.0
        weights: Weights = {s: 0.0 for s in syms}
        equity_curve: List[float] = [1.0]
        returns_list: List[float] = []
        trades: List[Dict[str, Any]] = []

        options_pnl_total = 0.0
        cost_pnl_total = 0.0

        for idx in range(1, n):
            prev_equity = equity

            # ── Step 1: Options PnL for current bar ──────────────────────
            bar_pnl = 0.0
            for sym in syms:
                w = weights.get(sym, 0.0)
                if abs(w) < 1e-10:
                    continue

                # Realized vol for this single bar (annualized)
                closes = dataset.perp_close.get(sym, [])
                if idx < len(closes) and closes[idx - 1] > 0 and closes[idx] > 0:
                    log_ret = math.log(closes[idx] / closes[idx - 1])
                    rv_bar = abs(log_ret) * math.sqrt(self.bars_per_year)
                else:
                    rv_bar = 0.0

                # IV for previous bar (what we "priced" the option at)
                iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                iv = self._get_smoothed_iv(iv_series, idx - 1)

                if iv is None or iv <= 0:
                    # No IV data: fall back to price PnL
                    if idx < len(closes) and closes[idx - 1] > 0:
                        bar_pnl += w * (closes[idx] / closes[idx - 1] - 1.0)
                    continue

                # Proper options PnL: Gamma/Theta model (no arbitrary SCALE)
                #
                # For a delta-hedged short straddle, P&L per bar:
                #   Theta income = 0.5 * IV² * dt            [always positive]
                #   Gamma cost   = 0.5 * RV_bar_ann² * dt    [scales with realized move²]
                #   Net          = 0.5 * (IV² - RV_bar_ann²) * dt
                #
                # Where RV_bar_ann = |log_ret| * sqrt(bars_per_year)  (annualized single-bar)
                # dt = 1 / bars_per_year
                #
                # Expected annual PnL (BTC, IV=76%, avg daily move=2.5%):
                #   Theta: 0.5 * 0.76² = 28.9% gross annual
                #   Gamma: 0.5 * (0.025*√365)² ≈ 0.5 * 0.228 = 11.4% annual cost
                #   Net VRP: ≈ 17.5% gross annual at 1x leverage
                #   Break-even daily move: IV/√bars_per_year = 76%/√365 ≈ 4%
                #   Tail: 10% crash day → loss ≈ 0.5*(0.58-3.65)/365 ≈ -0.42%/day

                # rv_bar = abs(log_ret) * sqrt(bars_per_year) — already computed above
                vrp_pnl = 0.5 * (iv ** 2 - rv_bar ** 2) * dt

                # Short vol = negative weight → (-w) > 0 → positive when IV > RV_bar
                bar_pnl += (-w) * vrp_pnl

            dp = equity * bar_pnl
            equity += dp
            options_pnl_total += dp

            # ── Step 2: Rebalance ─────────────────────────────────────────
            if strategy.should_rebalance(dataset, idx):
                target = strategy.target_weights(dataset, idx, weights)
                for s in syms:
                    target.setdefault(s, 0.0)

                turnover = sum(
                    abs(float(target.get(s, 0.0)) - float(weights.get(s, 0.0)))
                    for s in syms
                )
                bd = self.costs.cost(equity=equity, turnover=turnover)
                cost = float(bd.get("cost", 0.0))
                equity -= cost
                cost_pnl_total -= cost
                equity = max(equity, 0.0)

                weights = {s: float(target.get(s, 0.0)) for s in syms}
                if turnover > 1e-6:
                    trades.append({
                        "idx": idx,
                        "ts_epoch": dataset.timeline[idx],
                        "turnover": turnover,
                        **bd,
                    })

            equity_curve.append(equity)
            bar_ret = (equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
            returns_list.append(bar_ret)

        return BacktestResult(
            strategy=strategy.describe(),
            timeline=list(dataset.timeline[:len(equity_curve)]),
            equity_curve=equity_curve,
            returns=returns_list,
            trades=trades,
            breakdown={
                "options_pnl": options_pnl_total,
                "cost_pnl": cost_pnl_total,
                "funding_pnl": 0.0,
                "price_pnl": 0.0,
                "model": "gamma_theta_vrp",
            },
            data_fingerprint=dataset.fingerprint,
            code_fingerprint="options_engine_v1",
        )

    def _run_skew(self, dataset: MarketDataset, strategy: Strategy, feature_key: str = "skew_25d") -> BacktestResult:
        """Run vega-based strategy with feature-change P&L model.

        Works for skew_25d, term_spread, butterfly_25d, or any vega-like feature.

        P&L for a 25-delta risk reversal (skew trade):
            When "short skew" (sold puts, bought calls):
                PnL = -weight × Δ(skew_25d) × vega_sensitivity × dt_scale
                Positive when skew decreases (mean-reverts down)

            When "long skew" (bought puts, sold calls):
                PnL = weight × Δ(skew_25d) × vega_sensitivity × dt_scale
                Positive when skew increases (mean-reverts up)

        vega_sensitivity calibration:
            A 25d risk reversal has vega exposure proportional to IV level.
            Per unit notional, ~0.5 vega per 1% IV change.
            Normalized to portfolio return: vega_sens ≈ IV_atm (rough scaling).
        """
        syms = dataset.symbols
        n = len(dataset.timeline)
        dt = 1.0 / self.bars_per_year

        equity = 1.0
        weights: Weights = {s: 0.0 for s in syms}
        equity_curve: List[float] = [1.0]
        returns_list: List[float] = []
        trades: List[Dict[str, Any]] = []

        skew_pnl_total = 0.0
        cost_pnl_total = 0.0

        for idx in range(1, n):
            prev_equity = equity

            # ── Step 1: Skew P&L for current bar ──────────────────────
            bar_pnl = 0.0
            for sym in syms:
                w = weights.get(sym, 0.0)
                if abs(w) < 1e-10:
                    continue

                # Feature change for this bar (skew, term spread, etc.)
                skew_series = dataset.features.get(feature_key, {}).get(sym, [])
                if idx >= len(skew_series) or idx - 1 >= len(skew_series):
                    continue

                skew_now = skew_series[idx]
                skew_prev = skew_series[idx - 1]
                if skew_now is None or skew_prev is None:
                    continue

                d_skew = float(skew_now) - float(skew_prev)

                # Vega sensitivity: proportional to IV level
                iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                iv = self._get_smoothed_iv(iv_series, idx - 1)
                if iv is None or iv <= 0:
                    iv = 0.70  # default

                # Risk reversal P&L model:
                # PnL = weight × d_skew × sensitivity
                # weight < 0 means "short skew" → profits when skew decreases (d_skew < 0)
                # weight > 0 means "long skew" → profits when skew increases (d_skew > 0)
                # w * d_skew naturally gives correct sign (both negative → positive)
                # sensitivity = iv * sqrt(dt) — normalized to account for time scaling
                sensitivity = iv * math.sqrt(dt) * self.skew_sensitivity_mult
                skew_pnl = w * d_skew * sensitivity

                bar_pnl += skew_pnl

            dp = equity * bar_pnl
            equity += dp
            skew_pnl_total += dp

            # ── Step 2: Rebalance ─────────────────────────────────────
            if strategy.should_rebalance(dataset, idx):
                target = strategy.target_weights(dataset, idx, weights)
                for s in syms:
                    target.setdefault(s, 0.0)

                turnover = sum(
                    abs(float(target.get(s, 0.0)) - float(weights.get(s, 0.0)))
                    for s in syms
                )
                bd = self.costs.cost(equity=equity, turnover=turnover)
                cost = float(bd.get("cost", 0.0))
                equity -= cost
                cost_pnl_total -= cost
                equity = max(equity, 0.0)

                weights = {s: float(target.get(s, 0.0)) for s in syms}
                if turnover > 1e-6:
                    trades.append({
                        "idx": idx,
                        "ts_epoch": dataset.timeline[idx],
                        "turnover": turnover,
                        **bd,
                    })

            equity_curve.append(equity)
            bar_ret = (equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
            returns_list.append(bar_ret)

        return BacktestResult(
            strategy=strategy.describe(),
            timeline=list(dataset.timeline[:len(equity_curve)]),
            equity_curve=equity_curve,
            returns=returns_list,
            trades=trades,
            breakdown={
                "skew_pnl": skew_pnl_total,
                "cost_pnl": cost_pnl_total,
                "funding_pnl": 0.0,
                "price_pnl": 0.0,
                "model": "vega_skew_change",
            },
            data_fingerprint=dataset.fingerprint,
            code_fingerprint="options_engine_skew_v1",
        )

    def _get_smoothed_iv(self, iv_series: List, idx: int) -> Optional[float]:
        """Rolling mean of IV to reduce noise."""
        if not iv_series or idx >= len(iv_series):
            return None
        start = max(0, idx - self.iv_smooth_bars + 1)
        vals = [v for v in iv_series[start:idx + 1] if v is not None]
        return sum(vals) / len(vals) if vals else None


# ── Metrics calculator ────────────────────────────────────────────────────────

def compute_metrics(
    equity_curve: List[float],
    returns: List[float],
    bars_per_year: int = 365,
) -> Dict[str, float]:
    """Compute performance metrics from equity curve and returns.

    Returns:
        dict with: sharpe, sortino, cagr, max_drawdown, calmar, vol,
                   n_bars, final_equity, win_rate
    """
    n = len(equity_curve)
    if n < 2:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}

    final_eq = equity_curve[-1]
    n_years = (n - 1) / bars_per_year
    cagr = (final_eq ** (1.0 / n_years) - 1.0) if n_years > 0 and final_eq > 0 else 0.0

    # Sharpe
    if returns:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / max(len(returns) - 1, 1)
        vol = var_r ** 0.5
        sharpe = (mean_r / vol * math.sqrt(bars_per_year)) if vol > 0 else 0.0

        # Sortino (downside deviation)
        neg_rets = [r for r in returns if r < 0]
        if neg_rets:
            down_var = sum(r ** 2 for r in neg_rets) / len(neg_rets)
            down_std = down_var ** 0.5
            sortino = (mean_r / down_std * math.sqrt(bars_per_year)) if down_std > 0 else 0.0
        else:
            sortino = sharpe

        win_rate = sum(1 for r in returns if r > 0) / len(returns)
    else:
        sharpe = sortino = vol = win_rate = 0.0

    # Max drawdown
    peak = 1.0
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (eq / peak - 1.0) if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    calmar = abs(cagr / max_dd) if max_dd < 0 else 0.0

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "vol": round(vol * math.sqrt(bars_per_year), 4),
        "n_bars": n,
        "final_equity": round(final_eq, 4),
        "win_rate": round(win_rate, 4),
    }


# ── Walk-forward validation ────────────────────────────────────────────────────

def run_yearly_wf(
    provider_cfg: Dict[str, Any],
    strategy_cls,
    strategy_params: Dict[str, Any],
    years: List[int],
    costs: ExecutionCostModel,
    use_options_pnl: bool = True,
    bars_per_year: int = 365,
    seed: int = 42,
    skew_sensitivity_mult: float = 1.0,
) -> Dict[str, Any]:
    """Run per-year walk-forward validation.

    Args:
        provider_cfg: base provider config
        strategy_cls: strategy class
        strategy_params: strategy params
        years: list of years to evaluate
        costs: cost model
        use_options_pnl: True = gamma/theta model
        bars_per_year: 365 for daily
        seed: random seed
        skew_sensitivity_mult: multiplier for skew P&L sensitivity

    Returns:
        {"yearly": {...}, "summary": {...}}
    """
    from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider

    engine = OptionsBacktestEngine(
        costs=costs,
        bars_per_year=bars_per_year,
        use_options_pnl=use_options_pnl,
        skew_sensitivity_mult=skew_sensitivity_mult,
    )

    yearly: Dict[str, Any] = {}
    sharpes: List[float] = []

    for yr in years:
        cfg = dict(provider_cfg)
        cfg["start"] = f"{yr}-01-01"
        cfg["end"] = f"{yr}-12-31"

        try:
            provider = DeribitRestProvider(cfg, seed=seed)
            dataset = provider.load()
            if len(dataset.timeline) < 10:
                yearly[str(yr)] = {"error": "insufficient data", "sharpe": 0.0}
                sharpes.append(0.0)
                continue

            strategy = strategy_cls(params=strategy_params)
            result = engine.run(dataset, strategy, seed=seed)
            m = compute_metrics(result.equity_curve, result.returns, bars_per_year)

            yearly[str(yr)] = {
                "sharpe": m["sharpe"],
                "cagr_pct": round(m["cagr"] * 100, 2),
                "mdd_pct": round(m["max_drawdown"] * 100, 2),
                "calmar": m["calmar"],
                "n_bars": m["n_bars"],
                "passed": m["sharpe"] > 0.5,
            }
            sharpes.append(m["sharpe"])
        except Exception as e:
            yearly[str(yr)] = {"error": str(e), "sharpe": 0.0, "passed": False}
            sharpes.append(0.0)

    avg = round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0
    mn = round(min(sharpes), 4) if sharpes else 0.0
    passed = avg >= 1.0 and mn >= 0.5

    return {
        "yearly": yearly,
        "summary": {
            "avg_sharpe": avg,
            "min_sharpe": mn,
            "years_positive": sum(1 for s in sharpes if s > 0),
            "years_above_1": sum(1 for s in sharpes if s >= 1.0),
            "total_years": len(sharpes),
            "passed": passed,
        },
    }


from typing import Dict, Any, List
