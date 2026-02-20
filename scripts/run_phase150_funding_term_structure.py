#!/usr/bin/env python3
"""
Phase 150: Funding Rate Term Structure Spread Overlay
======================================================
Production state: OBJ=1.8886 (v2.3.0, breadth regime + funding dispersion + vol)

Background:
  - P135: Funding LEVEL overlay → NO IMPROVEMENT
  - P148: Funding DISPERSION (cross-sectional std) → MARGINAL +0.0086
  - P149: Return correlation → NO IMPROVEMENT

New hypothesis: Funding Rate TERM STRUCTURE SPREAD (short vs medium)
  When very recent funding (8h or 24h) is much HIGHER than medium-term (72h or 144h avg),
  it signals a SPIKE in demand for leverage = overcrowded positioning.
  These spikes typically mean-revert → reduce ensemble leverage.

  Conversely, when recent funding is LOWER than medium-term, demand is cooling
  despite momentum → potential for recovery bounce → can slightly boost.

  This is distinct from P135 (absolute level) and P148 (cross-sym dispersion).
  It uses the TIME dimension of funding dynamics.

Computed:
  spread = mean_funding_24h - mean_funding_144h  (across all symbols)
  When spread > 75th pct → scale×0.70 (overcrowded spike, reduce)
  When spread < 25th pct → scale×1.10 (cooling, slight boost)

Variants:
  A. reduce_high_spread_75:  scale×0.70 when spread_pct > 75th pct
  B. reduce_high_spread_90:  scale×0.70 when spread_pct > 90th pct
  C. boost_low_spread_25:    scale×1.10 when spread_pct < 25th pct
  D. bidirectional: A + C combined
  E. smooth: scale = 1 - 0.30 × (spread_pct - 0.5)
  F. stronger_reduce_90: scale×0.55 when spread_pct > 90th pct

Pass: OBJ > 1.8886 + LOYO >= 3/5
"""
import json
import os
import signal as _signal
import sys
import time
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
_signal.alarm(1500)

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]
SIG_KEYS = list(SIGNAL_DEFS.keys())

VOL_OVERLAY = PROD_CFG.get("vol_regime_overlay", {})
VOL_WINDOW = VOL_OVERLAY.get("window_bars", 168)
VOL_THRESHOLD = VOL_OVERLAY.get("threshold", 0.5)
VOL_SCALE = VOL_OVERLAY.get("scale_factor", 0.5)
VOL_F144_BOOST = VOL_OVERLAY.get("f144_boost", 0.2)

# Breadth regime params (P146)
BREADTH_P_LOW = 0.33
BREADTH_P_HIGH = 0.67
BREADTH_LB = 168

WEIGHTS = {
    "prod":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039},
    "p143b": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f144": 0.25},
    "mid":   {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f144": 0.25},
}
WEIGHTS_LIST = [WEIGHTS["prod"], WEIGHTS["mid"], WEIGHTS["p143b"]]

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase150"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})


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
    out = OUT_DIR / "phase150_report.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"✅ Saved → {out}")


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


def compute_funding_ts_spread(dataset, short_w: int = 24, long_w: int = 144) -> np.ndarray:
    """Funding rate term structure spread: mean_short - mean_long across all symbols."""
    n = len(dataset.timeline)
    fund_series = np.zeros(n)
    for i in range(n):
        ts = dataset.timeline[i]
        rates = []
        for sym in SYMBOLS:
            try:
                r = dataset.funding_rate_at(sym, ts)
                if r is not None and not np.isnan(float(r)):
                    rates.append(float(r))
            except Exception:
                pass
        fund_series[i] = float(np.mean(rates)) if rates else 0.0

    spread = np.zeros(n)
    for i in range(long_w, n):
        short_avg = float(np.mean(fund_series[max(0, i - short_w):i]))
        long_avg = float(np.mean(fund_series[i - long_w:i]))
        spread[i] = short_avg - long_avg
    spread[:long_w] = spread[long_w] if long_w < n else 0.0
    return spread


def compute_breadth(dataset, lookback: int = 168) -> np.ndarray:
    n = len(dataset.timeline)
    breadth = np.full(n, 0.5)
    for i in range(lookback, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - lookback)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    breadth[:lookback] = breadth[lookback] if lookback < n else 0.5
    return breadth


def rolling_percentile(signal: np.ndarray, window: int) -> np.ndarray:
    n = len(signal)
    pct = np.full(n, 0.5)
    for i in range(window, n):
        hist = signal[i - window:i]
        pct[i] = float(np.mean(hist <= signal[i]))
    pct[:window] = pct[window]
    return pct


def compute_ensemble(sig_rets, breadth_pct, btc_vol, spread_pct=None,
                     reduce_thresh=None, reduce_scale=0.70,
                     boost_thresh=None, boost_scale=1.10,
                     smooth_alpha=0.0) -> np.ndarray:
    """
    Full ensemble with breadth regime + vol overlay + funding term structure overlay.
    """
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
    bv = btc_vol[:min_len]
    brd_p = breadth_pct[:min_len]
    sp_p = spread_pct[:min_len] if spread_pct is not None else np.full(min_len, 0.5)
    ens = np.zeros(min_len)

    for i in range(min_len):
        # 1. Choose weights by breadth regime
        if brd_p[i] >= BREADTH_P_HIGH:
            w = WEIGHTS["p143b"]
        elif brd_p[i] >= BREADTH_P_LOW:
            w = WEIGHTS["mid"]
        else:
            w = WEIGHTS["prod"]

        # 2. Apply vol overlay
        scale = 1.0
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_other = VOL_F144_BOOST / max(1, len(SIG_KEYS) - 1)
            for sk in SIG_KEYS:
                if sk == "f144":
                    adj_w = min(0.60, w[sk] + VOL_F144_BOOST)
                else:
                    adj_w = max(0.05, w[sk] - boost_other)
                ens[i] += adj_w * sig_rets[sk][i]
            ens[i] *= VOL_SCALE
        else:
            for sk in SIG_KEYS:
                ens[i] += w[sk] * sig_rets[sk][i]

        # 3. Apply funding term structure overlay
        sp = sp_p[i]
        if smooth_alpha > 0:
            scale = 1.0 - smooth_alpha * (sp - 0.5)
        else:
            if reduce_thresh is not None and sp >= reduce_thresh:
                scale = reduce_scale
            elif boost_thresh is not None and sp <= boost_thresh:
                scale = boost_scale
        ens[i] *= scale

    return ens


def main():
    global _partial

    print("=" * 68)
    print("PHASE 150: Funding Rate Term Structure Spread Overlay")
    print("=" * 68)

    # ── Step 1: Load data ──────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    sig_returns: dict = {sk: {} for sk in SIG_KEYS}
    breadth_data: dict = {}
    spread_data: dict = {}
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
        breadth_data[year] = compute_breadth(dataset, lookback=BREADTH_LB)
        spread_data[year] = compute_funding_ts_spread(dataset, short_w=24, long_w=144)

        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)
            print(".", end="", flush=True)
        print(" ✓")

    # ── Step 2: IS comparison ─────────────────────────────────────────
    print("\n[2/4] Comparing funding term structure variants (IS 2021-2025)...")

    BASELINE_OBJ = 1.8886

    def yearly_sharpes(spread_pct_data=None, reduce_thresh=None, reduce_scale=0.70,
                       boost_thresh=None, boost_scale=1.10, smooth_alpha=0.0) -> dict:
        yearly = {}
        for yr in YEARS:
            sp_p = rolling_percentile(spread_data[yr], window=336) if spread_pct_data is None else spread_pct_data[yr]
            brd_p = rolling_percentile(breadth_data[yr], window=168)
            ens = compute_ensemble(
                {sk: sig_returns[sk][yr] for sk in SIG_KEYS},
                brd_p, btc_vol_data[yr], sp_p,
                reduce_thresh=reduce_thresh, reduce_scale=reduce_scale,
                boost_thresh=boost_thresh, boost_scale=boost_scale,
                smooth_alpha=smooth_alpha,
            )
            yearly[yr] = sharpe(ens)
        return yearly

    # Precompute spread percentiles
    spread_pct = {yr: rolling_percentile(spread_data[yr], window=336) for yr in YEARS}

    variants = {
        "baseline": dict(spread_pct_data=None, reduce_thresh=None, boost_thresh=None),
        "reduce_high_spread_75": dict(spread_pct_data=spread_pct, reduce_thresh=0.75, reduce_scale=0.70),
        "reduce_high_spread_90": dict(spread_pct_data=spread_pct, reduce_thresh=0.90, reduce_scale=0.70),
        "boost_low_spread_25":   dict(spread_pct_data=spread_pct, boost_thresh=0.25, boost_scale=1.10),
        "bidirectional_75_25":   dict(spread_pct_data=spread_pct, reduce_thresh=0.75, reduce_scale=0.70,
                                      boost_thresh=0.25, boost_scale=1.10),
        "smooth_30":             dict(spread_pct_data=spread_pct, smooth_alpha=0.30),
        "stronger_reduce_90":    dict(spread_pct_data=spread_pct, reduce_thresh=0.90, reduce_scale=0.55),
        "reduce_high_spread_80_scale60": dict(spread_pct_data=spread_pct, reduce_thresh=0.80, reduce_scale=0.60),
    }

    results = []
    for label, kwargs in variants.items():
        ys = yearly_sharpes(**kwargs)
        obj = obj_func(list(ys.values()))
        delta = round(obj - BASELINE_OBJ, 4)
        results.append({"label": label, "yearly": {yr: round(float(v), 4) for yr, v in ys.items()},
                        "obj": obj, "delta": delta})
        flag = "✅" if delta > 0 else "❌"
        print(f"  {flag} {label:40s} OBJ={obj:.4f} Δ={delta:+.5f}")

    results.sort(key=lambda x: x["obj"], reverse=True)
    best = results[0]
    print(f"\n  Best: {best['label']} → OBJ={best['obj']:.4f} (Δ={best['delta']:+.4f})")

    _partial.update({"phase": 150, "results": results, "best": best})
    _save(_partial, partial=True)

    # ── Step 3: LOYO (if improvement found) ──────────────────────────
    if best["obj"] <= BASELINE_OBJ:
        print("\n  No improvement vs baseline — skipping LOYO/WF")
        verdict = (
            f"NO IMPROVEMENT — all funding term structure variants below baseline OBJ={BASELINE_OBJ}. "
            f"Best: {best['label']} OBJ={best['obj']:.4f} (Δ={best['delta']:+.4f}). "
            f"Funding rate temporal dynamics do not add alpha to current stack."
        )
        next_phase = (
            "Phase 151 options:\n"
            "  A) New data source: taker buy ratio (buyer-initiated volume %)\n"
            "  B) Open interest dynamics: OI growth rate as momentum indicator\n"
            "  C) Cross-exchange funding arbitrage: Bybit vs Binance spread\n"
            "  D) Regime detection using Markov switching model on returns"
        )
        report = {
            "phase": 150,
            "description": "Funding Rate Term Structure Spread Overlay",
            "elapsed_seconds": round(time.time() - _start, 1),
            "baseline_obj": BASELINE_OBJ,
            "results": results,
            "best": best,
            "verdict": verdict,
            "next_phase_notes": next_phase,
        }
        _save(report, partial=False)
        print(f"\n{'='*68}")
        print(f"VERDICT: {verdict[:100]}...")
        print(f"{'='*68}")
        return report

    print("\n[3/4] LOYO validation of best variant...")
    # Extract best kwargs
    best_label = best["label"]
    best_kwargs = variants.get(best_label, {})

    loyo_results = []
    for test_yr in YEARS:
        train_yrs = [y for y in YEARS if y != test_yr]
        # Find best variant on train years
        best_train = None
        best_train_obj = -999.0
        for r in results:
            t_sharpes = [r["yearly"][y] for y in train_yrs]
            t_obj = obj_func(t_sharpes)
            if t_obj > best_train_obj:
                best_train_obj = t_obj
                best_train = r

        test_sharpe = best_train["yearly"][test_yr]
        baseline_sharpe = results[0]["yearly"][test_yr] if results[0]["label"] != "baseline" else results[1]["yearly"][test_yr]
        # Find baseline sharpe
        for r in results:
            if r["label"] == "baseline":
                baseline_sharpe = r["yearly"][test_yr]
                break
        delta = round(test_sharpe - baseline_sharpe, 4)
        loyo_results.append({
            "test_year": test_yr, "chosen": best_train["label"],
            "test_sharpe": round(test_sharpe, 4),
            "baseline_sharpe": round(baseline_sharpe, 4),
            "delta": delta,
        })
        print(f"  LOYO {test_yr}: {best_train['label']} Sharpe={test_sharpe:.4f} vs baseline={baseline_sharpe:.4f} (Δ={delta:+.4f})")

    loyo_wins = sum(1 for r in loyo_results if r["delta"] > 0)
    loyo_avg = round(float(np.mean([r["delta"] for r in loyo_results])), 4)
    print(f"  LOYO: {loyo_wins}/5 wins | avg_delta={loyo_avg:+.4f}")

    if best["delta"] > 0.005 and loyo_wins >= 3:
        verdict = f"VALIDATED — term structure spread adds +{best['delta']:.4f} OBJ, LOYO {loyo_wins}/5"
    else:
        verdict = f"MARGINAL/FAIL — OBJ delta={best['delta']:+.4f}, LOYO {loyo_wins}/5 — KEEP baseline"

    print(f"\n{'='*68}")
    print(f"VERDICT: {verdict}")
    print(f"  Baseline OBJ: {BASELINE_OBJ:.4f}")
    print(f"  Best OBJ:     {best['obj']:.4f} (Δ={best['delta']:+.4f})")
    print(f"{'='*68}")

    report = {
        "phase": 150,
        "description": "Funding Rate Term Structure Spread Overlay",
        "elapsed_seconds": round(time.time() - _start, 1),
        "baseline_obj": BASELINE_OBJ,
        "results": results,
        "best": best,
        "loyo": {"results": loyo_results, "wins": f"{loyo_wins}/5", "avg_delta": loyo_avg},
        "verdict": verdict,
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
