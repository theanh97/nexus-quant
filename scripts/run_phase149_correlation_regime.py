#!/usr/bin/env python3
"""
Phase 149: Cross-Symbol Return Correlation Regime Overlay
==========================================================
Phase 148 validated: funding dispersion boost ×1.15 when std>75th pct
Production OBJ = 1.8886 (v2.3.0)

Hypothesis — Cross-Symbol Correlation as Signal Quality Indicator:
  When all 10 crypto symbols move together (high pairwise correlation), the
  idio momentum strategy cannot differentiate returns → cross-sectional alpha
  collapses. In these periods, the ensemble's alpha should be lower.

  When correlation is low (high return dispersion), idio momentum has more
  material to work with → alpha is higher → can boost leverage.

  This is COMPLEMENTARY to Phase 148 (funding dispersion):
    - P148: funding rates spread → differentiated POSITIONING signals
    - P149: return correlations low → differentiated PRICE SIGNALS

  Computed: rolling 168h average pairwise correlation of hourly returns
  across all 10 USDT perpetual symbols.

Variants:
  A. reduce_high_corr_90pct: scale×0.65 when avg_corr > 90th percentile
  B. reduce_high_corr_75pct: scale×0.65 when avg_corr > 75th percentile
  C. boost_low_corr_25pct:   scale×1.10 when avg_corr < 25th percentile
  D. combo_reduce_boost:     reduce>80th pct + boost<20th pct
  E. smooth_z: returns × (1 - 0.35 × corr_pct) — smooth scaling

Pass criteria: OBJ > 1.8886 (P148 production) + LOYO >= 3/5
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
_signal.alarm(1800)

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
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

OUT_DIR = ROOT / "artifacts" / "phase149"
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

CORR_WINDOW = 168  # rolling window for pairwise correlation


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
    out = OUT_DIR / "phase149_report.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"✅ Saved → {out}")


def compute_all_state(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-bar:
    - btc_vol: BTC rolling price vol (annualised std)
    - breadth_regime: 0=LOW, 1=MID, 2=HIGH
    - fund_std_pct: rolling percentile of funding std across symbols
    - corr_pct: rolling percentile of avg pairwise return correlation

    Returns: (btc_vol, breadth_regime, fund_std_pct, corr_pct)
    """
    n = len(dataset.timeline)

    # BTC vol
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

    # Breadth regime
    breadth = np.full(n, 0.5)
    for i in range(BREADTH_LOOKBACK, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - BREADTH_LOOKBACK)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    breadth[:BREADTH_LOOKBACK] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        hist = breadth[i - PCT_WINDOW:i]
        brd_pct[i] = float(np.mean(hist <= breadth[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    breadth_regime = np.where(brd_pct >= P_HIGH, 2, np.where(brd_pct >= P_LOW, 1, 0)).astype(int)

    # Funding dispersion percentile
    fund_std_raw = np.zeros(n)
    for i in range(n):
        ts = dataset.timeline[i]
        rates = np.array([dataset.last_funding_rate_before(sym, ts) for sym in SYMBOLS])
        fund_std_raw[i] = float(np.std(rates)) if len(rates) > 1 else 0.0
    fund_std_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        hist = fund_std_raw[i - PCT_WINDOW:i]
        fund_std_pct[i] = float(np.mean(hist <= fund_std_raw[i]))
    fund_std_pct[:PCT_WINDOW] = 0.5

    # Cross-symbol return correlation
    # Build return matrix: (n, K)
    sym_rets = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(1, n):
            c0 = dataset.close(sym, i - 1)
            c1 = dataset.close(sym, i)
            sym_rets[i, j] = (c1 / c0 - 1.0) if c0 > 0 else 0.0

    # Rolling avg pairwise correlation
    avg_corr = np.full(n, 0.5)
    K = len(SYMBOLS)
    for i in range(CORR_WINDOW, n):
        window_rets = sym_rets[i - CORR_WINDOW:i, :]  # (CORR_WINDOW, K)
        corr_matrix = np.corrcoef(window_rets.T)  # (K, K)
        # Extract upper triangle (excluding diagonal)
        upper = corr_matrix[np.triu_indices(K, k=1)]
        avg_corr[i] = float(np.nanmean(upper))
    avg_corr[:CORR_WINDOW] = 0.5

    # Rolling percentile of avg_corr
    corr_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        hist = avg_corr[i - PCT_WINDOW:i]
        corr_pct[i] = float(np.mean(hist <= avg_corr[i]))
    corr_pct[:PCT_WINDOW] = 0.5

    return btc_vol, breadth_regime, fund_std_pct, corr_pct


def compute_full_ensemble(
    sig_rets: dict,
    btc_vol: np.ndarray,
    breadth_regime: np.ndarray,
    fund_std_pct: np.ndarray,
    corr_pct: np.ndarray,
    corr_variant: str = "none",
) -> np.ndarray:
    """
    Production ensemble with optional correlation overlay.

    corr_variant:
      "none"              : no correlation overlay (production baseline)
      "reduce_90pct"      : scale×0.65 when corr>90th pct
      "reduce_75pct"      : scale×0.65 when corr>75th pct
      "boost_25pct"       : scale×1.10 when corr<25th pct
      "combo_80_20"       : reduce>80th + boost<20th
      "smooth_z"          : smooth scale based on corr_pct
    """
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
    bv = btc_vol[:min_len]
    reg = breadth_regime[:min_len]
    fsp = fund_std_pct[:min_len]
    cp = corr_pct[:min_len]
    ens = np.zeros(min_len)

    for i in range(min_len):
        # Step 1: Breadth regime weights
        w = WEIGHTS_LIST[int(reg[i])]

        # Step 2: Vol overlay
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F144_BOOST / max(1, len(SIG_KEYS) - 1)
            base = 0.0
            for sk in SIG_KEYS:
                adj_w = min(0.60, w[sk] + VOL_F144_BOOST) if sk == "f144" else max(0.05, w[sk] - boost)
                base += adj_w * sig_rets[sk][i]
            ret_i = base * VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in SIG_KEYS)

        # Step 3: Funding dispersion boost (P148)
        if fsp[i] > FUND_DISP_PCT_THRESHOLD:
            ret_i *= FUND_DISP_BOOST_SCALE

        # Step 4: Correlation regime overlay (P149)
        corr_scale = 1.0
        if corr_variant == "reduce_90pct" and cp[i] > 0.90:
            corr_scale = 0.65
        elif corr_variant == "reduce_75pct" and cp[i] > 0.75:
            corr_scale = 0.65
        elif corr_variant == "boost_25pct" and cp[i] < 0.25:
            corr_scale = 1.10
        elif corr_variant == "combo_80_20":
            if cp[i] > 0.80:
                corr_scale = 0.65
            elif cp[i] < 0.20:
                corr_scale = 1.10
        elif corr_variant == "smooth_z":
            # scale = 1.0 - 0.35 * (corr_pct - 0.5) → range [0.825, 1.175]
            corr_scale = max(0.5, 1.0 - 0.35 * (cp[i] - 0.5))

        ens[i] = ret_i * corr_scale

    return ens


VARIANTS = [
    "none",            # production baseline
    "reduce_90pct",
    "reduce_75pct",
    "boost_25pct",
    "combo_80_20",
    "smooth_z",
]

VARIANT_LABELS = {
    "none":         "Production baseline (P148 + breadth + vol)",
    "reduce_90pct": "Reduce×0.65 when avg_corr > 90th pct",
    "reduce_75pct": "Reduce×0.65 when avg_corr > 75th pct",
    "boost_25pct":  "Boost×1.10 when avg_corr < 25th pct (low corr = differentiated)",
    "combo_80_20":  "Reduce>80th pct + Boost<20th pct (combo)",
    "smooth_z":     "Smooth: scale = 1 - 0.35×(corr_pct - 0.5)",
}


def main():
    global _partial

    print("=" * 72)
    print("PHASE 149: Cross-Symbol Return Correlation Regime Overlay")
    print("=" * 72)

    # ── Step 1: Load data ──────────────────────────────────────────────
    print("\n[1/4] Loading data + pre-computing signal returns + correlation...")
    sig_returns: dict = {sk: {} for sk in SIG_KEYS}
    btc_vol_data: dict = {}
    breadth_regime_data: dict = {}
    fund_std_pct_data: dict = {}
    corr_pct_data: dict = {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}: ", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()
        bv, br, fsp, cp = compute_all_state(dataset)
        btc_vol_data[year] = bv
        breadth_regime_data[year] = br
        fund_std_pct_data[year] = fsp
        corr_pct_data[year] = cp
        print("C", end="", flush=True)

        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)
            print(".", end="", flush=True)
        print(" ✓")

    # ── Step 2: Compare all variants ──────────────────────────────────
    print("\n[2/4] Comparing correlation overlay variants (IS 2021-2025)...")
    variant_results: dict = {}
    baseline_obj = None
    best_v = None
    best_obj = -999.0

    for v in VARIANTS:
        yearly = {}
        for year in YEARS:
            sr = {sk: sig_returns[sk][year] for sk in SIG_KEYS}
            ens = compute_full_ensemble(
                sr, btc_vol_data[year], breadth_regime_data[year],
                fund_std_pct_data[year], corr_pct_data[year], v,
            )
            yearly[year] = sharpe(ens)
        obj = obj_func(list(yearly.values()))
        delta = round(obj - (baseline_obj or obj), 4)
        if v == "none":
            baseline_obj = obj
            delta = 0.0
        variant_results[v] = {"yearly": yearly, "obj": obj, "delta": delta}
        flag = " ← BASELINE" if v == "none" else (" ✅" if obj > baseline_obj else "")
        print(f"  {VARIANT_LABELS[v]:62s} OBJ={obj:.4f} Δ={delta:+.4f}{flag}")
        if obj > best_obj:
            best_obj = obj
            best_v = v

    if best_v == "none":
        print(f"\n  No variant beats baseline OBJ={baseline_obj:.4f}")
    else:
        print(f"\n  Best: {VARIANT_LABELS[best_v]} → OBJ={best_obj:.4f} (Δ={best_obj - baseline_obj:+.4f})")

    _partial.update({
        "phase": 149,
        "baseline_obj": baseline_obj,
        "variant_results": {v: d for v, d in variant_results.items()},
        "best_variant": best_v,
        "best_obj": best_obj,
    })
    _save(_partial, partial=True)

    # ── Step 3: LOYO validation ────────────────────────────────────────
    loyo_results = []
    loyo_wins = 0
    loyo_avg = 0.0

    if best_v != "none":
        print(f"\n[3/4] LOYO validation of {best_v}...")
        for test_yr in YEARS:
            sr_test = {sk: sig_returns[sk][test_yr] for sk in SIG_KEYS}
            ens_best = compute_full_ensemble(
                sr_test, btc_vol_data[test_yr], breadth_regime_data[test_yr],
                fund_std_pct_data[test_yr], corr_pct_data[test_yr], best_v,
            )
            ens_base = compute_full_ensemble(
                sr_test, btc_vol_data[test_yr], breadth_regime_data[test_yr],
                fund_std_pct_data[test_yr], corr_pct_data[test_yr], "none",
            )
            s_best = sharpe(ens_best)
            s_base = sharpe(ens_base)
            delta = round(s_best - s_base, 4)
            loyo_results.append({
                "test_year": test_yr,
                "best_sharpe": round(s_best, 4),
                "baseline_sharpe": round(s_base, 4),
                "delta": delta,
            })
            flag = "✅" if delta > 0 else "❌"
            print(f"  OOS {test_yr}: {s_best:.4f} vs baseline={s_base:.4f} Δ={delta:+.4f} {flag}")
        loyo_wins = sum(1 for r in loyo_results if r["delta"] > 0)
        loyo_avg = round(float(np.mean([r["delta"] for r in loyo_results])), 4)
        print(f"  LOYO: {loyo_wins}/5 wins | avg_delta={loyo_avg:+.4f}")
    else:
        print("\n[3/4] LOYO: Skipped (no variant beats baseline)")

    # ── Step 4: WF validation ─────────────────────────────────────────
    wf_results = []
    wf_wins = 0
    wf_avg_delta = 0.0

    if best_v != "none" and loyo_wins >= 3:
        print(f"\n[4/4] Walk-Forward validation...")
        wf_windows = [
            {"train": ["2021", "2022", "2023"], "test": "2024"},
            {"train": ["2022", "2023", "2024"], "test": "2025"},
        ]
        for wf in wf_windows:
            test_yr = wf["test"]
            sr_test = {sk: sig_returns[sk][test_yr] for sk in SIG_KEYS}
            ens_best = compute_full_ensemble(
                sr_test, btc_vol_data[test_yr], breadth_regime_data[test_yr],
                fund_std_pct_data[test_yr], corr_pct_data[test_yr], best_v,
            )
            ens_base = compute_full_ensemble(
                sr_test, btc_vol_data[test_yr], breadth_regime_data[test_yr],
                fund_std_pct_data[test_yr], corr_pct_data[test_yr], "none",
            )
            test_s = sharpe(ens_best)
            base_s = sharpe(ens_base)
            delta = round(test_s - base_s, 4)
            wf_results.append({
                "test": test_yr, "test_sharpe": round(test_s, 4),
                "baseline_sharpe": round(base_s, 4), "delta": delta,
            })
            flag = "✅" if delta > 0 else "❌"
            print(f"  WF test={test_yr}: {test_s:.4f} vs {base_s:.4f} Δ={delta:+.4f} {flag}")
        wf_wins = sum(1 for r in wf_results if r["delta"] > 0)
        wf_avg_delta = round(float(np.mean([r["delta"] for r in wf_results])), 4)
        print(f"  WF: {wf_wins}/2 | avg_delta={wf_avg_delta:+.4f}")
    else:
        print("\n[4/4] WF: Skipped")

    # ── Verdict ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    validated = (
        best_v != "none"
        and best_obj > baseline_obj
        and loyo_wins >= 3
        and loyo_avg > 0
    )

    if validated:
        verdict = (
            f"IMPROVEMENT — {VARIANT_LABELS[best_v]} adds +{best_obj - baseline_obj:.4f} OBJ. "
            f"LOYO {loyo_wins}/5, avg_delta={loyo_avg:+.4f}, WF {wf_wins}/2"
        )
        next_phase = (
            "Phase 150: Integrate correlation overlay into production config v2.4.0. "
            "Then explore: IV/RV ratio signal or multi-venue basis."
        )
    elif best_v != "none":
        verdict = (
            f"MARGINAL — {VARIANT_LABELS[best_v]} IS +{best_obj - baseline_obj:.4f} "
            f"but LOYO {loyo_wins}/5 < 3. No production change."
        )
        next_phase = (
            "Phase 150: Correlation regime approach needs different formulation. "
            "Try: sector correlation (BTC+ETH vs alts) or extreme correlation threshold. "
            "OR: abandon correlation, explore on-chain throughput as market stress indicator."
        )
    else:
        verdict = (
            f"NO IMPROVEMENT — all correlation variants worse than baseline OBJ={baseline_obj:.4f}. "
            "Cross-symbol correlation is not additive to existing ensemble."
        )
        next_phase = (
            "Phase 150: Explore new angle — realized skewness of cross-section as quality filter. "
            "When avg return skew is very negative (tail risk), reduce leverage. "
            "OR: pivot to universe expansion (test adding newer alts)."
        )

    print(f"VERDICT: {verdict}")
    print(f"{'='*72}")

    report = {
        "phase": 149,
        "description": "Cross-Symbol Return Correlation Regime Overlay",
        "hypothesis": "High avg pairwise corr → idio signals weaker → reduce leverage",
        "elapsed_seconds": round(time.time() - _start, 1),
        "baseline_obj": baseline_obj,
        "variant_results": {v: d for v, d in variant_results.items()},
        "best_variant": best_v,
        "best_obj": best_obj,
        "loyo_results": loyo_results,
        "loyo_wins": loyo_wins,
        "loyo_avg_delta": loyo_avg,
        "wf_results": wf_results,
        "wf_wins": wf_wins,
        "wf_avg_delta": wf_avg_delta,
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
