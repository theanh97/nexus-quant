from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .metrics import equity_from_returns, summarize
from ..backtest.engine import BacktestResult
from ..data.schema import MarketDataset


@dataclass(frozen=True)
class WalkForwardConfig:
    enabled: bool = True
    window_bars: int = 720
    step_bars: int = 240


@dataclass(frozen=True)
class BenchmarkConfig:
    version: str = "v1"
    periods_per_year: float = 365.0
    walk_forward: WalkForwardConfig = WalkForwardConfig()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BenchmarkConfig":
        version = str(d.get("version") or "v1")
        ann = d.get("annualization") or {}
        ppy = float(ann.get("periods_per_year") or 365.0)
        wf = d.get("walk_forward") or {}
        walk = WalkForwardConfig(
            enabled=bool(wf.get("enabled", True)),
            window_bars=int(wf.get("window_bars") or 720),
            step_bars=int(wf.get("step_bars") or 240),
        )
        return BenchmarkConfig(version=version, periods_per_year=ppy, walk_forward=walk)


def _buy_hold_returns(dataset: MarketDataset, symbol: str) -> List[float]:
    closes = dataset.perp_close[symbol]
    out = []
    for i in range(1, len(closes)):
        c0 = closes[i - 1]
        c1 = closes[i]
        out.append((c1 / c0) - 1.0 if c0 != 0 else 0.0)
    return out


def _equal_weight_buy_hold_returns(dataset: MarketDataset) -> List[float]:
    syms = dataset.symbols
    n = max(1, len(syms))
    out = []
    for idx in range(1, len(dataset.timeline)):
        r = 0.0
        for s in syms:
            c0 = dataset.close(s, idx - 1)
            c1 = dataset.close(s, idx)
            r += (c1 / c0) - 1.0 if c0 != 0 else 0.0
        out.append(r / float(n))
    return out


def run_benchmark_pack_v1(dataset: MarketDataset, result: BacktestResult, bench_cfg: BenchmarkConfig) -> Dict[str, Any]:
    periods_per_year = float(bench_cfg.periods_per_year)

    # Strategy summary
    btc_sym = "BTCUSDT" if "BTCUSDT" in dataset.symbols else dataset.symbols[0]
    btc_ret = _buy_hold_returns(dataset, btc_sym)
    summary = summarize(
        returns=result.returns,
        equity_curve=result.equity_curve,
        periods_per_year=periods_per_year,
        trades=result.trades,
        benchmark_returns=btc_ret,
    )

    # Baselines
    baselines: Dict[str, Any] = {}
    baselines["btc_buy_hold"] = summarize(
        returns=btc_ret,
        equity_curve=equity_from_returns(btc_ret, start=1.0),
        periods_per_year=periods_per_year,
        trades=None,
    )
    eq_ret = _equal_weight_buy_hold_returns(dataset)
    baselines["equal_weight_buy_hold"] = summarize(
        returns=eq_ret,
        equity_curve=equity_from_returns(eq_ret, start=1.0),
        periods_per_year=periods_per_year,
        trades=None,
        benchmark_returns=btc_ret,
    )

    wf = _walk_forward(result, periods_per_year=periods_per_year, cfg=bench_cfg.walk_forward, btc_ret=btc_ret)

    return {
        "version": "v1",
        "meta": {
            "periods_per_year": periods_per_year,
            "btc_benchmark_symbol": btc_sym,
            "data_provider": dataset.provider,
            "data_fingerprint": dataset.fingerprint,
        },
        "summary": _round_summary(summary),
        "baselines": {k: _round_summary(v) for k, v in baselines.items()},
        "walk_forward": wf,
    }


def _walk_forward(
    result: BacktestResult, periods_per_year: float, cfg: WalkForwardConfig, btc_ret: List[float]
) -> Dict[str, Any]:
    if not cfg.enabled:
        return {"enabled": False}
    window = int(cfg.window_bars)
    step = int(cfg.step_bars)
    if window <= 0 or step <= 0:
        return {"enabled": False, "reason": "invalid window/step"}

    rs = result.returns
    eq = result.equity_curve
    windows: List[Dict[str, Any]] = []

    for start in range(0, max(0, len(rs) - window + 1), step):
        r_slice = rs[start : start + window]
        eq_slice = eq[start : start + window + 1]
        b_slice = btc_ret[start : start + window] if len(btc_ret) >= start + window else None
        windows.append(
            summarize(
                returns=r_slice,
                equity_curve=eq_slice,
                periods_per_year=periods_per_year,
                trades=None,
                benchmark_returns=b_slice,
            )
        )

    if not windows:
        return {"enabled": True, "windows": [], "stability": {"windows": 0}}

    # Simple stability stats
    n = len(windows)
    frac_profitable = sum(1 for w in windows if float(w.get("cagr", 0.0)) > 0) / float(n)
    frac_calmar_pos = sum(1 for w in windows if float(w.get("calmar", 0.0)) > 0) / float(n)
    frac_mdd_ok = sum(1 for w in windows if float(w.get("max_drawdown", 1.0)) < 0.35) / float(n)

    return {
        "enabled": True,
        "config": {"window_bars": window, "step_bars": step},
        "stability": {
            "windows": n,
            "fraction_profitable": round(frac_profitable, 4),
            "fraction_calmar_positive": round(frac_calmar_pos, 4),
            "fraction_mdd_lt_0_35": round(frac_mdd_ok, 4),
        },
        "windows": [_round_summary(w) for w in windows],
    }


def _round_summary(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, 6)
        else:
            out[k] = v
    return out

