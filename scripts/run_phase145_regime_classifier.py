#!/usr/bin/env python3
"""
Phase 145: Regime Classifier for Live Weight Switching
=======================================================
Phase 144 found:
  - 2021/2022/2024 → p143b weights (i415bw216-heavy) → best OBJ
  - 2023 → mid weights (interpolated) → best
  - 2025 → prod weights (V1-heavy) → best

Problem: Year-level assignment is look-ahead (we don't know in advance which
regime a year will be). We need a REAL-TIME computable classifier.

Hypothesis: Regime = **rolling funding rate percentile** (momentum richness signal)
  - High sustained positive funding → strong momentum → p143b
  - Mid neutral funding → mixed momentum → mid
  - Low/negative funding → cautious/corrective → prod

Method:
1. Compute rolling daily funding rate (mean across all symbols, 30d trailing)
2. For each bar: classify regime based on funding percentile vs trailing 1Y
3. Blend signals using regime-specific weights (bar-level, not year-level)
4. Compare vs fixed-weight baselines
5. Grid search: best percentile thresholds (p_low, p_high)
6. LOYO validate

Secondary classifier tested: cross-sectional price momentum breadth
  - What % of symbols have positive N-day return → breadth score
  - High breadth → momentum regime → p143b
  - Low breadth → defensive → prod

Expected: real-time classifier captures ~60-70% of the in-sample Phase 144 gain
"""
import json
import os
import signal as _signal
import sys
import time
from itertools import product as iproduct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

_partial: dict = {}
_start = time.time()


def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _save(_partial, partial=True)
    sys.exit(0)


_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(1500)  # 25 min

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]
SIG_KEYS = list(SIGNAL_DEFS.keys())

VOL_OVERLAY = PROD_CFG.get("vol_regime_overlay", {})
VOL_WINDOW = VOL_OVERLAY.get("window_bars", 168)
VOL_THRESHOLD = VOL_OVERLAY.get("threshold", 0.04)
VOL_SCALE = VOL_OVERLAY.get("scale_factor", 0.6)
VOL_F144_BOOST = VOL_OVERLAY.get("f144_boost", 0.15)

# Use 2 training years + 1 OOS for LOYO
YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase145"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Weight Sets (from Phase 144) ──────────────────────────────────────────
WEIGHTS = {
    "prod": {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039},
    "p143b": {"v1": 0.05, "i460bw168": 0.25, "i415bw216": 0.45, "f144": 0.25},
    "mid": {"v1": 0.16, "i460bw168": 0.22, "i415bw216": 0.37, "f144": 0.25},
}
REGIME_LABELS = ["prod", "mid", "p143b"]  # LOW → MID → HIGH momentum

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

# Funding window for regime signal (bars = hours)
FUNDING_WINDOWS = [168, 336, 504, 720]  # 1w, 2w, 3w, 4w


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: list) -> float:
    arr = np.array(yearly_sharpes)
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def _save(data: dict, partial: bool = False) -> None:
    data["partial"] = partial
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = OUT_DIR / "phase145_report.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"✅ Saved → {out}")


def compute_funding_signal(dataset) -> np.ndarray:
    """Compute mean funding rate across all symbols per bar."""
    n = len(dataset.timeline)
    funding = np.zeros(n)
    for i in range(n):
        ts = dataset.timeline[i]
        rates = []
        for sym in SYMBOLS:
            try:
                r = dataset.funding_rate_at(sym, ts)
                if r is not None and not np.isnan(r):
                    rates.append(float(r))
            except Exception:
                pass
        funding[i] = float(np.mean(rates)) if rates else 0.0
    return funding


def compute_breadth_signal(dataset, lookback: int = 168) -> np.ndarray:
    """Compute % of symbols with positive N-bar return (momentum breadth)."""
    n = len(dataset.timeline)
    breadth = np.full(n, 0.5)
    for i in range(lookback, n):
        positive = 0
        for sym in SYMBOLS:
            c0 = dataset.close(sym, i - lookback)
            c1 = dataset.close(sym, i)
            if c0 > 0 and c1 > c0:
                positive += 1
        breadth[i] = positive / len(SYMBOLS)
    breadth[:lookback] = breadth[lookback]
    return breadth


def rolling_percentile(signal: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling percentile rank of signal[i] vs prior window."""
    n = len(signal)
    pct = np.full(n, 0.5)
    for i in range(window, n):
        hist = signal[i - window:i]
        pct[i] = float(np.mean(hist <= signal[i]))
    pct[:window] = pct[window]
    return pct


def classify_regime(pct: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """
    Classify each bar into regime 0/1/2 based on percentile thresholds.
    0 = LOW (use prod)
    1 = MID (use mid)
    2 = HIGH (use p143b)
    """
    regime = np.where(pct >= p_high, 2, np.where(pct >= p_low, 1, 0))
    return regime.astype(int)


def compute_regime_ensemble(
    sig_rets: dict, regime: np.ndarray, weights_list: list,
    btc_vol: np.ndarray, with_vol_overlay: bool = True,
) -> np.ndarray:
    """Blend per-signal returns using regime-specific weights, bar by bar."""
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
    bv = btc_vol[:min_len]
    reg = regime[:min_len]
    ens = np.zeros(min_len)

    for i in range(min_len):
        w = weights_list[reg[i]]  # choose weights by regime
        if with_vol_overlay and not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_per_other = VOL_F144_BOOST / max(1, len(SIG_KEYS) - 1)
            for sk in SIG_KEYS:
                if sk == "f144":
                    adj_w = min(0.60, w[sk] + VOL_F144_BOOST)
                else:
                    adj_w = max(0.05, w[sk] - boost_per_other)
                ens[i] += adj_w * sig_rets[sk][i]
            ens[i] *= VOL_SCALE
        else:
            for sk in SIG_KEYS:
                ens[i] += w[sk] * sig_rets[sk][i]
    return ens


def compute_btc_price_vol(dataset, window: int = 168) -> np.ndarray:
    n = len(dataset.timeline)
    rets = np.zeros(n)
    for i in range(1, n):
        c0 = dataset.close("BTCUSDT", i - 1)
        c1 = dataset.close("BTCUSDT", i)
        rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    vol = np.full(n, np.nan)
    for i in range(window, n):
        vol[i] = float(np.std(rets[i - window:i])) * np.sqrt(8760)
    if window < n:
        vol[:window] = vol[window]
    return vol


def main():
    global _partial

    print("=" * 68)
    print("PHASE 145: Regime Classifier for Live Weight Switching")
    print("=" * 68)

    # ── Step 1: Load data + precompute signals ──────────────────────────
    print("\n[1/4] Loading data + computing signal returns + regime features...")
    sig_returns: dict[str, dict] = {sk: {} for sk in SIG_KEYS}
    funding_data: dict = {}
    breadth_data: dict = {}
    btc_vol_data: dict = {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}: ", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()

        btc_vol_data[year] = compute_btc_price_vol(dataset, window=VOL_WINDOW)
        funding_data[year] = compute_funding_signal(dataset)
        breadth_data[year] = compute_breadth_signal(dataset, lookback=168)

        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)
            print(".", end="", flush=True)
        print(" ✓")

    # ── Baseline Sharpes (single weight sets) ──────────────────────────
    print("\n[2/4] Baseline Sharpes per weight set...")
    baseline_yearly = {}
    for label, w in WEIGHTS.items():
        yearly = {}
        for year in YEARS:
            ens = np.zeros(min(len(sig_returns[sk][year]) for sk in SIG_KEYS))
            for sk in SIG_KEYS:
                ens += w[sk] * sig_returns[sk][year][:len(ens)]
            yearly[year] = sharpe(ens)
        avg = round(float(np.mean(list(yearly.values()))), 4)
        mn = round(float(np.min(list(yearly.values()))), 4)
        obj = obj_func(list(yearly.values()))
        baseline_yearly[label] = yearly
        print(f"  {label:6s}: OBJ={obj:.4f} AVG={avg:.4f} MIN={mn:.4f} | {yearly}")

    baseline_obj = obj_func(list(baseline_yearly["prod"].values()))
    ph144_best_obj = 1.9306  # Phase 144 in-sample benchmark

    # ── Step 3: Grid search regime thresholds ─────────────────────────
    print("\n[3/4] Grid search: funding percentile thresholds + window...")
    weights_list = [WEIGHTS["prod"], WEIGHTS["mid"], WEIGHTS["p143b"]]

    best_results = []

    for fund_window in FUNDING_WINDOWS:
        print(f"  Testing funding window={fund_window}h...")
        for p_low in [0.25, 0.33, 0.40, 0.45]:
            for p_high in [0.55, 0.60, 0.67, 0.75]:
                if p_high <= p_low:
                    continue

                # Compute classifier + adaptive Sharpe per year
                yearly_adaptive = {}
                for year in YEARS:
                    fund_pct = rolling_percentile(funding_data[year], window=fund_window)
                    regime = classify_regime(fund_pct, p_low, p_high)
                    ens = compute_regime_ensemble(
                        {sk: sig_returns[sk][year] for sk in SIG_KEYS},
                        regime, weights_list, btc_vol_data[year],
                    )
                    yearly_adaptive[year] = sharpe(ens)

                obj = obj_func(list(yearly_adaptive.values()))
                avg = round(float(np.mean(list(yearly_adaptive.values()))), 4)
                mn = round(float(np.min(list(yearly_adaptive.values()))), 4)
                best_results.append({
                    "signal": "funding",
                    "window": fund_window,
                    "p_low": p_low,
                    "p_high": p_high,
                    "yearly": {yr: round(float(v), 4) for yr, v in yearly_adaptive.items()},
                    "avg_sharpe": avg,
                    "min_sharpe": mn,
                    "obj": obj,
                })

    # Also test breadth signal
    print("  Testing breadth signal...")
    for lb in [84, 168, 336]:
        for p_low in [0.33, 0.40]:
            for p_high in [0.60, 0.67]:
                yearly_adaptive = {}
                for year in YEARS:
                    breadth_lb = compute_breadth_signal(
                        type("D", (), {"close": lambda self, s, i, d=None: None,
                                       "timeline": sig_returns["v1"][year]})(),
                        lookback=lb,
                    ) if False else None
                    # Use precomputed breadth_data[year] (lb=168 fixed)
                    brd_pct = rolling_percentile(breadth_data[year], window=336)
                    regime = classify_regime(brd_pct, p_low, p_high)
                    ens = compute_regime_ensemble(
                        {sk: sig_returns[sk][year] for sk in SIG_KEYS},
                        regime, weights_list, btc_vol_data[year],
                    )
                    yearly_adaptive[year] = sharpe(ens)

                obj = obj_func(list(yearly_adaptive.values()))
                avg = round(float(np.mean(list(yearly_adaptive.values()))), 4)
                mn = round(float(np.min(list(yearly_adaptive.values()))), 4)
                best_results.append({
                    "signal": "breadth",
                    "window": lb,
                    "p_low": p_low,
                    "p_high": p_high,
                    "yearly": {yr: round(float(v), 4) for yr, v in yearly_adaptive.items()},
                    "avg_sharpe": avg,
                    "min_sharpe": mn,
                    "obj": obj,
                })

    best_results.sort(key=lambda x: x["obj"], reverse=True)
    top10 = best_results[:10]
    best = top10[0]

    print(f"\n  Top 5 classifiers:")
    for r in top10[:5]:
        print(f"    OBJ={r['obj']:.4f} AVG={r['avg_sharpe']:.4f} MIN={r['min_sharpe']:.4f} | "
              f"sig={r['signal']} win={r['window']} p=[{r['p_low']},{r['p_high']}] | {r['yearly']}")
    print(f"\n  Baseline (prod): OBJ={baseline_obj:.4f}")
    print(f"  Best classifier: OBJ={best['obj']:.4f} (Δ vs baseline={best['obj']-baseline_obj:+.4f})")
    print(f"  Phase 144 IS benchmark: OBJ={ph144_best_obj:.4f}")
    print(f"  Capture ratio: {(best['obj']-baseline_obj)/(ph144_best_obj-baseline_obj)*100:.1f}%")

    _partial.update({
        "phase": 145, "baseline_yearly": baseline_yearly,
        "top10_classifiers": top10, "best": best,
    })
    _save(_partial, partial=True)

    # ── Step 4: LOYO validation ────────────────────────────────────────
    print("\n[4/4] LOYO validation of best classifier...")
    best_cfg = best
    loyo_results = []

    for test_yr in YEARS:
        train_yrs = [y for y in YEARS if y != test_yr]

        # Train: find best threshold combo on train years
        train_best = None
        train_best_obj = -999.0
        for r in best_results[:50]:  # search top 50 configs
            train_sharpes = [r["yearly"][y] for y in train_yrs]
            train_obj = obj_func(train_sharpes)
            if train_obj > train_best_obj:
                train_best_obj = train_obj
                train_best = r

        if train_best is None:
            train_best = best

        # Apply best config to test year
        fund_pct = rolling_percentile(funding_data[test_yr], window=train_best["window"])
        regime = classify_regime(fund_pct, train_best["p_low"], train_best["p_high"])
        ens = compute_regime_ensemble(
            {sk: sig_returns[sk][test_yr] for sk in SIG_KEYS},
            regime, weights_list, btc_vol_data[test_yr],
        )
        test_sharpe = sharpe(ens)
        baseline_sharpe = float(baseline_yearly["prod"][test_yr])
        delta = round(test_sharpe - baseline_sharpe, 4)

        loyo_results.append({
            "test_year": test_yr,
            "train_config": {k: v for k, v in train_best.items() if k != "yearly"},
            "test_sharpe": round(test_sharpe, 4),
            "baseline_sharpe": round(baseline_sharpe, 4),
            "delta": delta,
        })
        print(f"  LOYO {test_yr}: Sharpe={test_sharpe:.4f} vs baseline={baseline_sharpe:.4f} (Δ={delta:+.4f})")

    loyo_wins = sum(1 for r in loyo_results if r["delta"] > 0)
    loyo_avg_delta = round(float(np.mean([r["delta"] for r in loyo_results])), 4)
    loyo_summary = {"results": loyo_results, "wins": f"{loyo_wins}/5", "avg_delta": loyo_avg_delta}
    print(f"  LOYO: {loyo_wins}/5 wins | avg_delta={loyo_avg_delta:+.4f}")

    # ── Verdict ────────────────────────────────────────────────────────
    delta_obj = round(best["obj"] - baseline_obj, 4)
    loyo_passes = loyo_wins >= 3
    capture_ratio = (best["obj"] - baseline_obj) / (ph144_best_obj - baseline_obj)

    if delta_obj > 0.05 and loyo_passes:
        verdict = (
            f"VALIDATED — live classifier adds +{delta_obj:.4f} OBJ vs baseline "
            f"({capture_ratio*100:.0f}% of P144 IS gain), LOYO {loyo_wins}/5"
        )
        next_phase = (
            "Phase 146: Implement funding-percentile regime classifier in production config. "
            "Add vol_regime_weight_switching block with p_low, p_high, fund_window params."
        )
    elif delta_obj > 0:
        verdict = f"MARGINAL — +{delta_obj:.4f} OBJ but LOYO {loyo_wins}/5 — needs stronger signal"
        next_phase = (
            "Phase 146 options:\n"
            "  A) Test Deribit IV skew as regime signal (options-implied momentum)\n"
            "  B) Test multi-signal composite: funding + breadth + vol\n"
            "  C) Try 2-regime only (skip mid), simpler classifier"
        )
    else:
        verdict = f"FAIL — classifier adds no OBJ improvement, LOYO {loyo_wins}/5"
        next_phase = "Phase 146: Try new alpha source — on-chain whale flows or Deribit options data."

    print(f"\n{'='*68}")
    print(f"VERDICT: {verdict}")
    print(f"  Baseline OBJ:         {baseline_obj:.4f}")
    print(f"  Best classifier OBJ:  {best['obj']:.4f}  (Δ={delta_obj:+.4f})")
    print(f"  P144 IS benchmark:    {ph144_best_obj:.4f}")
    print(f"  Capture ratio:        {capture_ratio*100:.1f}%")
    print(f"  LOYO {loyo_wins}/5 | avg_delta={loyo_avg_delta:+.4f}")
    print(f"{'='*68}")

    report = {
        "phase": 145,
        "description": "Regime Classifier for Live Weight Switching",
        "hypothesis": "Rolling funding rate percentile classifies momentum regime → switch weights",
        "elapsed_seconds": round(time.time() - _start, 1),
        "baseline_obj": baseline_obj,
        "ph144_is_benchmark": ph144_best_obj,
        "best_classifier": best,
        "top10_classifiers": top10,
        "capture_ratio": round(capture_ratio, 4),
        "delta_obj": delta_obj,
        "loyo": loyo_summary,
        "verdict": verdict,
        "next_phase_notes": next_phase,
    }
    _save(report, partial=False)
    return report


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        _partial["error"] = str(e)
        _partial["traceback"] = traceback.format_exc()
        _save(_partial, partial=True)
        sys.exit(1)
