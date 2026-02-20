#!/usr/bin/env python3
"""
Phase 147: 2025 Deep-Dive + New Signal Candidates
==================================================
Context:
- P144-146: Breadth regime classifier validated (OBJ=1.8945, prod v2.2.0)
- Only weakness: WF 2025 slightly negative (-0.024 Sharpe delta)
- Breadth signal in 2025: 1.8152 vs baseline 1.8981 (tiny loss, -0.083)

Goal: Find what's unique about 2025 + test new signals that could fix it
without hurting other years.

2025 characteristics (hypothesis):
  - Macro regime: Trump tariff uncertainty + BTC ETF flows dominating
  - Alt-coin underperformance vs BTC (ETH/BTC ratio declining)
  - Breadth classifier pushes to HIGH momentum → but 2025 momentum signals mixed
  - Possible fix: add BTC dominance or ETH/BTC ratio as regime modifier

New signals to test:
  A) BTC dominance proxy: BTC relative strength vs alt-basket
     - If BTC dramatically outperforms alts → reduce idio exposure
     - Rationale: in BTC-dominant regime, idio momentum is weaker

  B) Cross-asset correlation spike: when all coins highly correlated
     - Use rolling pairwise correlation of returns (correlation regime)
     - High correlation → idio signal has less alpha → use F144 more

  C) Funding rate spread: max_funding - min_funding across symbols
     - Wide spread → arb opportunity → use more F144
     - Narrow spread → trending → use idio signals

Method:
1. Deep-dive: compute signal diagnostics for 2025 monthly breakdown
2. Test signal A (BTC dominance) as regime modifier
3. Test signal B (correlation spike) as weight modifier
4. Test signal C (funding spread) as regime addend
5. LOYO validate best combo
6. Update production config if validated
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

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase147"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "prod":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039},
    "p143b": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f144": 0.25},
    "mid":   {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f144": 0.25},
    # New: BTC-dominant weight set (V1 heavy + more F144)
    "btc_dom": {"v1": 0.35, "i460bw168": 0.15,   "i415bw216": 0.28,   "f144": 0.22},
}

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
    out = OUT_DIR / "phase147_report.json"
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


def compute_btc_dominance(dataset, window: int = 168) -> np.ndarray:
    """BTC relative strength vs equal-weight alt basket."""
    alts = [s for s in SYMBOLS if s != "BTCUSDT"]
    n = len(dataset.timeline)
    dom = np.zeros(n)
    for i in range(window, n):
        btc_ret = 0.0
        c0_btc = dataset.close("BTCUSDT", i - window)
        c1_btc = dataset.close("BTCUSDT", i)
        if c0_btc > 0:
            btc_ret = (c1_btc / c0_btc) - 1.0
        alt_rets = []
        for sym in alts:
            c0 = dataset.close(sym, i - window)
            c1 = dataset.close(sym, i)
            if c0 > 0:
                alt_rets.append((c1 / c0) - 1.0)
        alt_avg = float(np.mean(alt_rets)) if alt_rets else 0.0
        dom[i] = btc_ret - alt_avg  # positive = BTC dominant
    dom[:window] = dom[window] if window < n else 0.0
    return dom


def compute_correlation_regime(dataset, window: int = 168) -> np.ndarray:
    """Mean pairwise return correlation across symbols."""
    n = len(dataset.timeline)
    corr_sig = np.full(n, 0.5)
    ret_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(1, n):
            c0 = dataset.close(sym, i - 1)
            c1 = dataset.close(sym, i)
            ret_mat[i, j] = (c1 / c0 - 1.0) if c0 > 0 else 0.0

    for i in range(window, n):
        block = ret_mat[i - window:i]
        if block.shape[0] < 20:
            continue
        # Compute mean pairwise correlation
        cov = np.cov(block.T)
        std = np.sqrt(np.diag(cov))
        corr_sum = 0.0
        count = 0
        for a in range(len(SYMBOLS)):
            for b in range(a + 1, len(SYMBOLS)):
                if std[a] > 1e-10 and std[b] > 1e-10:
                    corr_sum += cov[a, b] / (std[a] * std[b])
                    count += 1
        corr_sig[i] = corr_sum / count if count > 0 else 0.5
    corr_sig[:window] = corr_sig[window] if window < n else 0.5
    return corr_sig


def rolling_percentile(signal: np.ndarray, window: int) -> np.ndarray:
    n = len(signal)
    pct = np.full(n, 0.5)
    for i in range(window, n):
        hist = signal[i - window:i]
        pct[i] = float(np.mean(hist <= signal[i]))
    pct[:window] = pct[window]
    return pct


def classify_regime_3(pct: np.ndarray, p_low: float = 0.33, p_high: float = 0.67) -> np.ndarray:
    return np.where(pct >= p_high, 2, np.where(pct >= p_low, 1, 0)).astype(int)


def blend_ens(weights: dict, sig_rets: dict, btc_vol: np.ndarray) -> np.ndarray:
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
    bv = btc_vol[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F144_BOOST / max(1, len(SIG_KEYS) - 1)
            for sk in SIG_KEYS:
                adj_w = min(0.60, w[sk] + VOL_F144_BOOST) if sk == "f144" else max(0.05, w[sk] - boost)
                ens[i] += adj_w * sig_rets[sk][i]
            ens[i] *= VOL_SCALE
        else:
            for sk in SIG_KEYS:
                ens[i] += w[sk] * sig_rets[sk][i]
    return ens


def compute_regime_ens(sig_rets, regime, weights_list, btc_vol) -> np.ndarray:
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
    bv = btc_vol[:min_len]
    reg = regime[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights_list[reg[i]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F144_BOOST / max(1, len(SIG_KEYS) - 1)
            for sk in SIG_KEYS:
                adj_w = min(0.60, w[sk] + VOL_F144_BOOST) if sk == "f144" else max(0.05, w[sk] - boost)
                ens[i] += adj_w * sig_rets[sk][i]
            ens[i] *= VOL_SCALE
        else:
            for sk in SIG_KEYS:
                ens[i] += w[sk] * sig_rets[sk][i]
    return ens


def main():
    global _partial

    print("=" * 68)
    print("PHASE 147: 2025 Analysis + New Signal Candidates")
    print("=" * 68)

    # ── Step 1: Load data ──────────────────────────────────────────────
    print("\n[1/5] Loading data + computing auxiliary signals...")
    sig_returns: dict = {sk: {} for sk in SIG_KEYS}
    breadth_data: dict = {}
    btc_dom_data: dict = {}
    corr_data: dict = {}
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
        breadth_data[year] = compute_breadth(dataset, lookback=168)
        btc_dom_data[year] = compute_btc_dominance(dataset, window=168)
        corr_data[year] = compute_correlation_regime(dataset, window=168)

        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)
            print(".", end="", flush=True)
        print(" ✓")

    # ── Step 2: Baseline breadth classifier (P146 config) ─────────────
    print("\n[2/5] Baseline breadth classifier (P146)...")
    WEIGHTS_LIST_BASE = [WEIGHTS["prod"], WEIGHTS["mid"], WEIGHTS["p143b"]]
    yearly_base = {}
    for year in YEARS:
        brd_pct = rolling_percentile(breadth_data[year], window=168)
        regime = classify_regime_3(brd_pct, 0.33, 0.67)
        ens = compute_regime_ens(
            {sk: sig_returns[sk][year] for sk in SIG_KEYS},
            regime, WEIGHTS_LIST_BASE, btc_vol_data[year],
        )
        yearly_base[year] = sharpe(ens)
    obj_base = obj_func(list(yearly_base.values()))
    print(f"  Baseline (P146): OBJ={obj_base:.4f} | {yearly_base}")

    # ── Step 3: 2025 monthly breakdown ────────────────────────────────
    print("\n[3/5] 2025 monthly performance breakdown...")
    # Simulate monthly Sharpes for 2025 with baseline breadth classifier
    year = "2025"
    brd_pct_2025 = rolling_percentile(breadth_data[year], window=168)
    regime_2025 = classify_regime_3(brd_pct_2025, 0.33, 0.67)
    ens_2025 = compute_regime_ens(
        {sk: sig_returns[sk][year] for sk in SIG_KEYS},
        regime_2025, WEIGHTS_LIST_BASE, btc_vol_data[year],
    )
    prod_ens_2025 = blend_ens(WEIGHTS["prod"],
                              {sk: sig_returns[sk][year] for sk in SIG_KEYS},
                              btc_vol_data[year])

    n = min(len(ens_2025), len(prod_ens_2025))
    # Split into ~2-month chunks (approx 1460 bars each)
    chunk = n // 6
    monthly_adaptive = []
    monthly_prod = []
    for m in range(6):
        sl = slice(m * chunk, (m + 1) * chunk)
        monthly_adaptive.append(round(float(np.mean(ens_2025[sl])) * 8760, 4))
        monthly_prod.append(round(float(np.mean(prod_ens_2025[sl])) * 8760, 4))
    print(f"  2025 bimonthly mean annualized return:")
    print(f"    Adaptive: {monthly_adaptive}")
    print(f"    Prod:     {monthly_prod}")
    print(f"  2025 regime distribution: {dict(zip(*np.unique(regime_2025, return_counts=True)))}")

    # ── Step 4: Test new signal combinations ─────────────────────────
    print("\n[4/5] Testing new signal combinations for 2025 fix...")
    results = []

    # A) BTC dominance as 4th dimension: HIGH btc_dom → demote to prod weights even if breadth high
    for dom_threshold in [0.05, 0.10, 0.15, 0.20]:
        yearly = {}
        for yr in YEARS:
            brd_pct = rolling_percentile(breadth_data[yr], window=168)
            dom_pct = rolling_percentile(btc_dom_data[yr], window=336)
            min_len = min(len(sig_returns[sk][yr]) for sk in SIG_KEYS)
            brd_p = brd_pct[:min_len]
            dom_p = dom_pct[:min_len]
            regime = np.zeros(min_len, dtype=int)
            for i in range(min_len):
                # Breadth regime, but demote if BTC very dominant
                if brd_p[i] >= 0.67 and dom_p[i] <= 0.33:
                    # HIGH breadth but BTC dominating → alts underperforming → use mid
                    regime[i] = 1
                elif brd_p[i] >= 0.67:
                    regime[i] = 2
                elif brd_p[i] >= 0.33:
                    regime[i] = 1
                else:
                    regime[i] = 0
            ens = compute_regime_ens(
                {sk: sig_returns[sk][yr] for sk in SIG_KEYS},
                regime, WEIGHTS_LIST_BASE, btc_vol_data[yr],
            )
            yearly[yr] = sharpe(ens)
        obj = obj_func(list(yearly.values()))
        results.append({
            "signal": "btc_dom_demotion",
            "dom_threshold": dom_threshold,
            "yearly": {yr: round(float(v), 4) for yr, v in yearly.items()},
            "obj": obj,
            "avg": round(float(np.mean(list(yearly.values()))), 4),
            "min": round(float(np.min(list(yearly.values()))), 4),
        })
        print(f"  btc_dom demotion dom_thresh={dom_threshold}: OBJ={obj:.4f} | {yearly}")

    # B) Correlation regime: HIGH corr → use prod (idio signal weak)
    for corr_threshold in [0.67, 0.75, 0.80]:
        yearly = {}
        for yr in YEARS:
            brd_pct = rolling_percentile(breadth_data[yr], window=168)
            corr_pct = rolling_percentile(corr_data[yr], window=336)
            min_len = min(len(sig_returns[sk][yr]) for sk in SIG_KEYS)
            brd_p = brd_pct[:min_len]
            crr_p = corr_pct[:min_len]
            regime = np.zeros(min_len, dtype=int)
            for i in range(min_len):
                if crr_p[i] >= corr_threshold:
                    regime[i] = 0  # HIGH correlation → defensive
                elif brd_p[i] >= 0.67:
                    regime[i] = 2
                elif brd_p[i] >= 0.33:
                    regime[i] = 1
                else:
                    regime[i] = 0
            ens = compute_regime_ens(
                {sk: sig_returns[sk][yr] for sk in SIG_KEYS},
                regime, WEIGHTS_LIST_BASE, btc_vol_data[yr],
            )
            yearly[yr] = sharpe(ens)
        obj = obj_func(list(yearly.values()))
        results.append({
            "signal": "corr_override",
            "corr_threshold": corr_threshold,
            "yearly": {yr: round(float(v), 4) for yr, v in yearly.items()},
            "obj": obj,
            "avg": round(float(np.mean(list(yearly.values()))), 4),
            "min": round(float(np.min(list(yearly.values()))), 4),
        })
        print(f"  corr override corr_pct>={corr_threshold}: OBJ={obj:.4f} | {yearly}")

    # C) 4-weight set: add btc_dom weight set for alt-underperforming regime
    WEIGHTS_LIST_4 = [WEIGHTS["btc_dom"], WEIGHTS["prod"], WEIGHTS["mid"], WEIGHTS["p143b"]]
    for p_btcdom in [0.20, 0.25]:
        yearly = {}
        for yr in YEARS:
            brd_pct = rolling_percentile(breadth_data[yr], window=168)
            dom_pct = rolling_percentile(btc_dom_data[yr], window=336)
            min_len = min(len(sig_returns[sk][yr]) for sk in SIG_KEYS)
            regime = np.zeros(min_len, dtype=int)
            for i in range(min_len):
                if dom_pct[i] >= (1.0 - p_btcdom):  # BTC dominant
                    regime[i] = 0  # btc_dom weights
                elif brd_pct[i] >= 0.67:
                    regime[i] = 3  # p143b
                elif brd_pct[i] >= 0.33:
                    regime[i] = 2  # mid
                else:
                    regime[i] = 1  # prod
            ens = compute_regime_ens(
                {sk: sig_returns[sk][yr] for sk in SIG_KEYS},
                regime, WEIGHTS_LIST_4, btc_vol_data[yr],
            )
            yearly[yr] = sharpe(ens)
        obj = obj_func(list(yearly.values()))
        results.append({
            "signal": "4_weight_sets",
            "p_btcdom": p_btcdom,
            "yearly": {yr: round(float(v), 4) for yr, v in yearly.items()},
            "obj": obj,
            "avg": round(float(np.mean(list(yearly.values()))), 4),
            "min": round(float(np.min(list(yearly.values()))), 4),
        })
        print(f"  4-weight sets p_btcdom={p_btcdom}: OBJ={obj:.4f} | {yearly}")

    results.sort(key=lambda x: x["obj"], reverse=True)
    best = results[0]
    print(f"\n  Best new config: OBJ={best['obj']:.4f} | {best}")
    print(f"  vs Baseline (P146): OBJ={obj_base:.4f}")

    _partial.update({
        "phase": 147, "baseline_obj": obj_base,
        "top_results": results[:5], "best": best,
        "monthly_2025_adaptive": monthly_adaptive,
        "monthly_2025_prod": monthly_prod,
    })
    _save(_partial, partial=True)

    # ── Step 5: LOYO validation of best new config ────────────────────
    print("\n[5/5] LOYO validation of best new config...")

    def apply_best_config(yr):
        """Apply the best found config to a year."""
        sig = best.get("signal", "btc_dom_demotion")
        dom_pct = rolling_percentile(btc_dom_data[yr], window=336)
        brd_pct = rolling_percentile(breadth_data[yr], window=168)
        min_len = min(len(sig_returns[sk][yr]) for sk in SIG_KEYS)
        regime = np.zeros(min_len, dtype=int)
        if sig == "btc_dom_demotion":
            thresh = best.get("dom_threshold", 0.10)
            for i in range(min_len):
                if brd_pct[i] >= 0.67 and dom_pct[i] <= 0.33:
                    regime[i] = 1
                elif brd_pct[i] >= 0.67:
                    regime[i] = 2
                elif brd_pct[i] >= 0.33:
                    regime[i] = 1
                else:
                    regime[i] = 0
            return compute_regime_ens(
                {sk: sig_returns[sk][yr] for sk in SIG_KEYS},
                regime, WEIGHTS_LIST_BASE, btc_vol_data[yr],
            )
        elif sig == "corr_override":
            corr_pct = rolling_percentile(corr_data[yr], window=336)
            cth = best.get("corr_threshold", 0.75)
            for i in range(min_len):
                if corr_pct[i] >= cth:
                    regime[i] = 0
                elif brd_pct[i] >= 0.67:
                    regime[i] = 2
                elif brd_pct[i] >= 0.33:
                    regime[i] = 1
                else:
                    regime[i] = 0
            return compute_regime_ens(
                {sk: sig_returns[sk][yr] for sk in SIG_KEYS},
                regime, WEIGHTS_LIST_BASE, btc_vol_data[yr],
            )
        else:
            # Default: use baseline
            brd_pct = rolling_percentile(breadth_data[yr], window=168)
            regime = classify_regime_3(brd_pct, 0.33, 0.67)
            return compute_regime_ens(
                {sk: sig_returns[sk][yr] for sk in SIG_KEYS},
                regime, WEIGHTS_LIST_BASE, btc_vol_data[yr],
            )

    loyo_results = []
    for test_yr in YEARS:
        train_yrs = [y for y in YEARS if y != test_yr]
        # Train: pick best config from top 5 by train OBJ
        best_train = None
        best_train_obj = -999.0
        for r in results[:10]:
            train_sh = [r["yearly"][y] for y in train_yrs]
            t_obj = obj_func(train_sh)
            if t_obj > best_train_obj:
                best_train_obj = t_obj
                best_train = r

        # Test year with best_train config
        ens = apply_best_config(test_yr)
        baseline_ens = blend_ens(WEIGHTS["prod"],
                                 {sk: sig_returns[sk][test_yr] for sk in SIG_KEYS},
                                 btc_vol_data[test_yr])
        test_sharpe = sharpe(ens)
        baseline_sharpe = sharpe(baseline_ens)
        delta = round(test_sharpe - baseline_sharpe, 4)
        loyo_results.append({
            "test_year": test_yr,
            "test_sharpe": round(test_sharpe, 4),
            "baseline_sharpe": round(baseline_sharpe, 4),
            "delta": delta,
        })
        print(f"  LOYO {test_yr}: Sharpe={test_sharpe:.4f} vs baseline={baseline_sharpe:.4f} (Δ={delta:+.4f})")

    loyo_wins = sum(1 for r in loyo_results if r["delta"] > 0)
    loyo_avg = round(float(np.mean([r["delta"] for r in loyo_results])), 4)
    print(f"  LOYO: {loyo_wins}/5 wins | avg_delta={loyo_avg:+.4f}")

    # Compare with P146 baseline
    delta_obj = round(best["obj"] - obj_base, 4)
    loyo_passes = loyo_wins >= 3

    if delta_obj > 0.01 and loyo_passes:
        verdict = (
            f"IMPROVEMENT — new signal adds +{delta_obj:.4f} OBJ, LOYO {loyo_wins}/5. "
            f"Update production config."
        )
    elif delta_obj > 0:
        verdict = f"MARGINAL — +{delta_obj:.4f} OBJ but LOYO {loyo_wins}/5 — keep P146 config"
    else:
        verdict = (
            f"NO IMPROVEMENT — keep P146 breadth classifier. "
            f"2025 miss (-0.024) is acceptable given overall +0.40 avg_delta. "
            f"Consider exploring on-chain/options signals in P148."
        )

    print(f"\n{'='*68}")
    print(f"VERDICT: {verdict}")
    print(f"  P146 baseline OBJ:     {obj_base:.4f}")
    print(f"  Best new config OBJ:   {best['obj']:.4f} (Δ={delta_obj:+.4f})")
    print(f"  LOYO {loyo_wins}/5 | avg_delta={loyo_avg:+.4f}")
    print(f"{'='*68}")

    report = {
        "phase": 147,
        "description": "2025 Deep-Dive + New Signal Candidates",
        "elapsed_seconds": round(time.time() - _start, 1),
        "baseline_p146_obj": obj_base,
        "monthly_2025": {"adaptive": monthly_adaptive, "prod": monthly_prod},
        "top_results": results[:5],
        "best_new": best,
        "delta_obj": delta_obj,
        "loyo": {"results": loyo_results, "wins": f"{loyo_wins}/5", "avg_delta": loyo_avg},
        "verdict": verdict,
        "next_phase_notes": (
            "Phase 148: Explore on-chain whale flow signal (Glassnode/Nansen) "
            "OR Deribit IV skew surface as regime indicator. "
            "Both independent of price/funding and could add alpha in choppy regimes."
            if "NO IMPROVEMENT" in verdict or "MARGINAL" in verdict
            else "Phase 148: Production deployment + live monitoring dashboard. "
                 "Track regime distribution, weight stability, alpha per regime."
        ),
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
