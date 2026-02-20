#!/usr/bin/env python3
"""
Phase 151: Universe Expansion — 10 → 15 Major Crypto Perps
============================================================
Phase 150: cross-sectional skewness NO IMPROVEMENT
Phases 149-150: overlays diminishing returns. Structural change needed.

Hypothesis — More Symbols = Better Cross-Sectional Discrimination:
  Current: 10 symbols, k=4 per side → 4/10 = 40% coverage (less selective)
  Expanded: 15 symbols, k=4 per side → 4/15 = 27% coverage (more selective)

  Benefits:
  1. Idio momentum gets more coins to rank → stronger signal-to-noise
  2. Better diversification across the portfolio
  3. More funding rate variation across 15 coins → F144 has more alpha

  Added symbols (all major perps, liquid since ≥2020):
    MATICUSDT - MATIC/Polygon (liquid since 2021)
    LTCUSDT   - Litecoin (liquid since 2018)
    BCHUSDT   - Bitcoin Cash (liquid since 2018)
    ATOMUSDT  - Cosmos (liquid since 2020)
    NEARUSDT  - NEAR Protocol (liquid since 2020)

Test approach:
  1. Run baseline (10 symbols) with full production stack
  2. Run expanded (15 symbols) with SAME weights
  3. Also test 15-sym with re-tuned k_per_side (k=5 might be better at 15 syms)
  4. LOYO + WF comparison

Note: Does NOT re-optimize weights (would require full grid search).
      Tests whether structural expansion alone improves OBJ.
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
_signal.alarm(2400)  # 40min timeout — loading 15 syms × 5 years is slower

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS_10 = PROD_CFG["data"]["symbols"]  # current 10
SYMBOLS_15 = SYMBOLS_10 + ["UNIUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "NEARUSDT"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]
SIG_KEYS = list(SIGNAL_DEFS.keys())

VOL_OVERLAY = PROD_CFG.get("vol_regime_overlay", {})
VOL_WINDOW = VOL_OVERLAY.get("window_bars", 168)
VOL_THRESHOLD = VOL_OVERLAY.get("threshold", 0.5)
VOL_SCALE = VOL_OVERLAY.get("scale_factor", 0.5)
VOL_F144_BOOST = VOL_OVERLAY.get("f144_boost", 0.2)

BREADTH_LOOKBACK = 168
PCT_WINDOW = 336
P_LOW = 0.33
P_HIGH = 0.67
FUND_DISP_BOOST_SCALE = 1.15
FUND_DISP_PCT_THRESHOLD = 0.75

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase151"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "prod":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039},
    "p143b": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f144": 0.25},
    "mid":   {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f144": 0.25},
}
WEIGHTS_LIST = [WEIGHTS["prod"], WEIGHTS["mid"], WEIGHTS["p143b"]]

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
    out = OUT_DIR / "phase151_report.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"✅ Saved → {out}")


def compute_overlays(dataset, symbols) -> tuple:
    """Compute per-bar overlays: btc_vol, breadth_regime, fund_std_pct."""
    n = len(dataset.timeline)

    # BTC vol (always from BTCUSDT regardless of universe)
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = dataset.close("BTCUSDT", i - 1)
        c1 = dataset.close("BTCUSDT", i)
        btc_rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n:
        btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]

    # Breadth using the given universe
    breadth = np.full(n, 0.5)
    for i in range(BREADTH_LOOKBACK, n):
        pos = sum(
            1 for sym in symbols
            if (c0 := dataset.close(sym, i - BREADTH_LOOKBACK)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(symbols)
    breadth[:BREADTH_LOOKBACK] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        hist = breadth[i - PCT_WINDOW:i]
        brd_pct[i] = float(np.mean(hist <= breadth[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    breadth_regime = np.where(brd_pct >= P_HIGH, 2, np.where(brd_pct >= P_LOW, 1, 0)).astype(int)

    # Funding dispersion (over available symbols in dataset)
    fund_std_raw = np.zeros(n)
    for i in range(n):
        ts = dataset.timeline[i]
        rates = []
        for sym in symbols:
            try:
                r = dataset.last_funding_rate_before(sym, ts)
                rates.append(r)
            except Exception:
                pass
        fund_std_raw[i] = float(np.std(rates)) if len(rates) > 1 else 0.0
    fund_std_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        hist = fund_std_raw[i - PCT_WINDOW:i]
        fund_std_pct[i] = float(np.mean(hist <= fund_std_raw[i]))
    fund_std_pct[:PCT_WINDOW] = 0.5

    return btc_vol, breadth_regime, fund_std_pct


def compute_ensemble(sig_rets, btc_vol, breadth_regime, fund_std_pct) -> np.ndarray:
    """Full production ensemble (breadth + vol + funding dispersion)."""
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
    bv = btc_vol[:min_len]
    reg = breadth_regime[:min_len]
    fsp = fund_std_pct[:min_len]
    ens = np.zeros(min_len)

    for i in range(min_len):
        w = WEIGHTS_LIST[int(reg[i])]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F144_BOOST / max(1, len(SIG_KEYS) - 1)
            base = 0.0
            for sk in SIG_KEYS:
                adj_w = min(0.60, w[sk] + VOL_F144_BOOST) if sk == "f144" else max(0.05, w[sk] - boost)
                base += adj_w * sig_rets[sk][i]
            ret_i = base * VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in SIG_KEYS)
        if fsp[i] > FUND_DISP_PCT_THRESHOLD:
            ret_i *= FUND_DISP_BOOST_SCALE
        ens[i] = ret_i

    return ens


def build_configs(k_per_side: int, symbols: list) -> dict:
    """Build strategy configs for a given universe + k_per_side."""
    return {
        "v1": {
            "strategy": "nexus_alpha_v1",
            "params": {
                "k_per_side": 2,
                "w_carry": 0.35, "w_mom": 0.45, "w_mean_reversion": 0.2,
                "momentum_lookback_bars": 336,
                "mean_reversion_lookback_bars": 72,
                "vol_lookback_bars": 168,
                "target_gross_leverage": 0.35,
                "rebalance_interval_bars": 60,
            }
        },
        "i460bw168": {
            "strategy": "idio_momentum_alpha",
            "params": {
                "k_per_side": k_per_side,
                "lookback_bars": 460, "beta_window_bars": 168,
                "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
            }
        },
        "i415bw216": {
            "strategy": "idio_momentum_alpha",
            "params": {
                "k_per_side": k_per_side,
                "lookback_bars": 415, "beta_window_bars": 216,
                "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
            }
        },
        "f144": {
            "strategy": "funding_momentum_alpha",
            "params": {
                "k_per_side": 2,
                "funding_lookback_bars": 144,
                "direction": "contrarian",
                "target_gross_leverage": 0.25,
                "rebalance_interval_bars": 24,
            }
        },
    }


def main():
    global _partial

    print("=" * 72)
    print("PHASE 151: Universe Expansion — 10 → 15 Major Crypto Perps")
    print("=" * 72)
    print(f"  10-sym: {SYMBOLS_10}")
    print(f"  15-sym: {SYMBOLS_15}")

    # Test configurations
    test_configs = [
        ("10sym_k4",  SYMBOLS_10, 4),  # production baseline
        ("15sym_k4",  SYMBOLS_15, 4),  # expanded universe, same k
        ("15sym_k5",  SYMBOLS_15, 5),  # expanded universe, k=5 (more selective)
    ]

    yearly_results: dict = {}
    obj_results: dict = {}

    for label, symbols, k in test_configs:
        print(f"\n[Loading] {label} ({len(symbols)} symbols, k_per_side={k})...")
        sig_configs = build_configs(k, symbols)

        yr_sharpes = {}
        for year, (start, end) in YEAR_RANGES.items():
            print(f"  {year}: ", end="", flush=True)
            cfg_data = {
                "provider": "binance_rest_v1", "symbols": symbols,
                "start": start, "end": end, "bar_interval": "1h",
                "cache_dir": ".cache/binance_rest",
            }
            provider = make_provider(cfg_data, seed=42)
            dataset = provider.load()
            bv, br, fsp = compute_overlays(dataset, symbols)

            sig_rets = {}
            for sk in SIG_KEYS:
                bt_cfg = BacktestConfig(costs=COST_MODEL)
                strat = make_strategy({"name": sig_configs[sk]["strategy"], "params": sig_configs[sk]["params"]})
                engine = BacktestEngine(bt_cfg)
                result = engine.run(dataset, strat)
                sig_rets[sk] = np.array(result.returns, dtype=np.float64)
                print(".", end="", flush=True)

            ens = compute_ensemble(sig_rets, bv, br, fsp)
            yr_sharpes[year] = sharpe(ens)
            print(f" Sharpe={yr_sharpes[year]:.4f}")

        obj = obj_func(list(yr_sharpes.values()))
        yearly_results[label] = yr_sharpes
        obj_results[label] = obj
        print(f"  → OBJ={obj:.4f} | {yr_sharpes}")

        _partial.update({"phase": 151, "results_so_far": {
            k_: {"yearly": yearly_results[k_], "obj": obj_results[k_]}
            for k_ in yearly_results
        }})
        _save(_partial, partial=True)

    # ── Compare ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("COMPARISON:")
    baseline_obj = obj_results["10sym_k4"]
    for label, obj in obj_results.items():
        delta = round(obj - baseline_obj, 4)
        flag = " ← BASELINE" if label == "10sym_k4" else (" ✅" if obj > baseline_obj else "")
        print(f"  {label:15s} OBJ={obj:.4f} Δ={delta:+.4f}{flag}")

    # Best expanded config
    expanded_best = max(
        [(l, o) for l, o in obj_results.items() if l != "10sym_k4"],
        key=lambda x: x[1],
        default=(None, -999),
    )
    best_label, best_obj = expanded_best

    if best_obj > baseline_obj:
        verdict = (
            f"IMPROVEMENT — {best_label} OBJ={best_obj:.4f} "
            f"(+{best_obj - baseline_obj:.4f} vs 10-sym baseline)"
        )
        next_phase = (
            f"Phase 152: LOYO + WF validation of {best_label} universe expansion. "
            f"If passes → update production config with {best_label.split('_')[0]} symbols."
        )
        validated = True
    else:
        verdict = (
            f"NO IMPROVEMENT — universe expansion does not help. "
            f"Best expanded OBJ={best_obj:.4f} vs baseline={baseline_obj:.4f}."
        )
        next_phase = (
            "Phase 152: Explore alternative directions — "
            "(a) Re-optimize I415/I460 lookback for 2025 specifically, "
            "(b) Test shorter funding lookback (F72, F96) for 2025 crypto regime, "
            "(c) Accept current system and deploy."
        )
        validated = False

    print(f"\nVERDICT: {verdict}")
    print("=" * 72)

    report = {
        "phase": 151,
        "description": "Universe Expansion — 10 → 15 major crypto perps",
        "hypothesis": "More symbols → better idio signal discrimination",
        "elapsed_seconds": round(time.time() - _start, 1),
        "symbols_10": SYMBOLS_10,
        "symbols_15": SYMBOLS_15,
        "results": {l: {"yearly": yearly_results[l], "obj": obj_results[l]} for l in yearly_results},
        "baseline_obj": baseline_obj,
        "best_expanded": best_label,
        "best_obj": best_obj,
        "validated": validated,
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
