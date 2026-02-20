#!/usr/bin/env python3
"""
Phase 152: Funding Momentum Lookback Tuning
============================================
Production state: OBJ=2.0079 (P150: breadth + vol + funding_dispersion + funding_term_structure)
Phase 151: Universe expansion → NO IMPROVEMENT (15-sym OBJ=1.33 vs 10-sym 1.89)

Context — 2025 weakness:
  P150 full stack: 2025 Sharpe ≈ 1.85 (weakest year)
  Hypothesis: 2025 crypto regime has FASTER funding dynamics — funding mean-reversion
  cycles shortened due to more sophisticated participants and faster information flow.
  Current F144 (144h lookback) may be too slow to capture 2025 funding cycles.

Test:
  Baseline: F144 (current production) — full P150 stack
  Variants: F48, F72, F96, F120, F144 (baseline), F168, F192
  Full production stack on every variant (vol + breadth + fund_dispersion + fund_ts)
  Precompute v1, i460bw168, i415bw216 once → test each f-lookback variant separately

Pass criteria: OBJ > 2.0079 AND LOYO >= 3/5 wins
               OR: OBJ > 2.0079 AND 2025 Sharpe improvement > 0.05
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
_signal.alarm(2700)  # 45min

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]

# ── Overlay params from production config ──────────────────────────────────
VOL_OV = PROD_CFG.get("vol_regime_overlay", {})
VOL_WINDOW    = VOL_OV.get("window_bars", 168)
VOL_THRESHOLD = VOL_OV.get("threshold", 0.5)
VOL_SCALE     = VOL_OV.get("scale_factor", 0.5)
VOL_F144_BOOST = VOL_OV.get("f144_boost", 0.2)

BRS = PROD_CFG.get("breadth_regime_switching", {})
BREADTH_LB    = BRS.get("breadth_lookback_bars", 168)
PCT_WINDOW    = BRS.get("rolling_percentile_window", 336)
P_LOW         = BRS.get("p_low", 0.33)
P_HIGH        = BRS.get("p_high", 0.67)

FDO = PROD_CFG.get("funding_dispersion_overlay", {})
FUND_DISP_PCT_THRESH = FDO.get("boost_threshold_pct", 0.75)
FUND_DISP_BOOST      = FDO.get("boost_scale", 1.15)

FTS = PROD_CFG.get("funding_term_structure_overlay", {})
FTS_SHORT_W       = FTS.get("short_window_bars", 24)
FTS_LONG_W        = FTS.get("long_window_bars", 144)
FTS_PCT_WINDOW    = FTS.get("rolling_percentile_window", 336)
FTS_REDUCE_THRESH = FTS.get("reduce_threshold", 0.75)
FTS_REDUCE_SCALE  = FTS.get("reduce_scale", 0.7)
FTS_BOOST_THRESH  = FTS.get("boost_threshold", 0.25)
FTS_BOOST_SCALE   = FTS.get("boost_scale", 1.1)

WEIGHTS = {
    "prod":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039},
    "p143b": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f144": 0.25},
    "mid":   {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f144": 0.25},
}

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

BASE_SIGS = ["v1", "i460bw168", "i415bw216"]  # precomputed once
F_SIG = "f144"  # varied per test
ALL_SIGS = BASE_SIGS + [F_SIG]

F_LOOKBACKS = [48, 72, 96, 120, 144, 168, 192]  # 144 = production baseline

OUT_DIR = ROOT / "artifacts" / "phase152"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

BASELINE_OBJ = 2.0079  # P150 production


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
    out = OUT_DIR / "phase152_report.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"✅ Saved → {out}")


def rolling_percentile(signal: np.ndarray, window: int) -> np.ndarray:
    n = len(signal)
    pct = np.full(n, 0.5)
    for i in range(window, n):
        hist = signal[i - window:i]
        pct[i] = float(np.mean(hist <= signal[i]))
    if window < n:
        pct[:window] = pct[window]
    return pct


def compute_btc_vol(dataset) -> np.ndarray:
    n = len(dataset.timeline)
    rets = np.zeros(n)
    for i in range(1, n):
        c0 = dataset.close("BTCUSDT", i - 1)
        c1 = dataset.close("BTCUSDT", i)
        rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        vol[i] = float(np.std(rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n:
        vol[:VOL_WINDOW] = vol[VOL_WINDOW]
    return vol


def compute_breadth(dataset) -> np.ndarray:
    n = len(dataset.timeline)
    breadth = np.full(n, 0.5)
    for i in range(BREADTH_LB, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - BREADTH_LB)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    if BREADTH_LB < n:
        breadth[:BREADTH_LB] = breadth[BREADTH_LB]
    return breadth


def compute_funding_dispersion(dataset) -> np.ndarray:
    """Cross-sectional std of funding rates across all symbols."""
    n = len(dataset.timeline)
    fund_std = np.zeros(n)
    for i in range(n):
        ts = dataset.timeline[i]
        rates = []
        for sym in SYMBOLS:
            try:
                r = dataset.last_funding_rate_before(sym, ts)
                if r is not None and not np.isnan(float(r)):
                    rates.append(float(r))
            except Exception:
                pass
        fund_std[i] = float(np.std(rates)) if len(rates) > 1 else 0.0
    return fund_std


def compute_funding_ts_spread(dataset) -> np.ndarray:
    """Term structure spread: mean_short_w - mean_long_w across all symbols."""
    n = len(dataset.timeline)
    fund_level = np.zeros(n)
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
        fund_level[i] = float(np.mean(rates)) if rates else 0.0
    spread = np.zeros(n)
    for i in range(FTS_LONG_W, n):
        s_avg = float(np.mean(fund_level[max(0, i - FTS_SHORT_W):i]))
        l_avg = float(np.mean(fund_level[i - FTS_LONG_W:i]))
        spread[i] = s_avg - l_avg
    if FTS_LONG_W < n:
        spread[:FTS_LONG_W] = spread[FTS_LONG_W]
    return spread


def compute_ensemble(sig_rets: dict, btc_vol, breadth_pct,
                     fund_disp_pct, fts_pct) -> np.ndarray:
    """Full P150 production stack."""
    min_len = min(len(sig_rets[sk]) for sk in ALL_SIGS)
    bv  = btc_vol[:min_len]
    brd = breadth_pct[:min_len]
    fdo = fund_disp_pct[:min_len]
    fts = fts_pct[:min_len]
    ens = np.zeros(min_len)

    n_others = len(ALL_SIGS) - 1  # for vol overlay redistribution

    for i in range(min_len):
        # 1. Breadth regime → weight set
        if brd[i] >= P_HIGH:
            w = WEIGHTS["p143b"]
        elif brd[i] >= P_LOW:
            w = WEIGHTS["mid"]
        else:
            w = WEIGHTS["prod"]

        # 2. Vol overlay
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_other = VOL_F144_BOOST / max(1, n_others)
            ret_i = 0.0
            for sk in ALL_SIGS:
                if sk == F_SIG:
                    adj_w = min(0.60, w[sk] + VOL_F144_BOOST)
                else:
                    adj_w = max(0.05, w[sk] - boost_other)
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in ALL_SIGS)

        # 3. Funding dispersion overlay (P148)
        if fdo[i] > FUND_DISP_PCT_THRESH:
            ret_i *= FUND_DISP_BOOST

        # 4. Funding term structure overlay (P150)
        sp = fts[i]
        if sp >= FTS_REDUCE_THRESH:
            ret_i *= FTS_REDUCE_SCALE
        elif sp <= FTS_BOOST_THRESH:
            ret_i *= FTS_BOOST_SCALE

        ens[i] = ret_i

    return ens


def main():
    global _partial

    print("=" * 72)
    print("PHASE 152: Funding Momentum Lookback Tuning")
    print(f"  Baseline: F144, OBJ={BASELINE_OBJ}")
    print(f"  Testing F lookbacks: {F_LOOKBACKS}")
    print("=" * 72)

    # ── Step 1: Precompute base signals + overlays ───────────────────────
    print("\n[1/3] Precomputing base signals (v1, i460bw168, i415bw216) + overlays...")

    base_sig_rets: dict = {sk: {} for sk in BASE_SIGS}  # sk → year → np.array
    f_sig_rets: dict = {lb: {} for lb in F_LOOKBACKS}   # lb → year → np.array
    btc_vols, breadth_raw, fund_disp_raw, fts_raw = {}, {}, {}, {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}: loading+computing... ", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()

        btc_vols[year]     = compute_btc_vol(dataset)
        breadth_raw[year]  = compute_breadth(dataset)
        fund_disp_raw[year] = compute_funding_dispersion(dataset)
        fts_raw[year]       = compute_funding_ts_spread(dataset)

        # Base signals
        for sk in BASE_SIGS:
            sig_def = SIGNAL_DEFS[sk]
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            result = BacktestEngine(bt_cfg).run(dataset, strat)
            base_sig_rets[sk][year] = np.array(result.returns, dtype=np.float64)
            print("b", end="", flush=True)

        # F144 variants — run each lookback
        f_params_base = dict(SIGNAL_DEFS[F_SIG]["params"])
        for lb in F_LOOKBACKS:
            params = dict(f_params_base)
            params["funding_lookback_bars"] = lb
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": SIGNAL_DEFS[F_SIG]["strategy"], "params": params})
            result = BacktestEngine(bt_cfg).run(dataset, strat)
            f_sig_rets[lb][year] = np.array(result.returns, dtype=np.float64)
            print("f", end="", flush=True)

        print(f" ✓")

    # ── Step 2: Compute rolling overlays + evaluate each F variant ───────
    print("\n[2/3] Evaluating F lookback variants with full P150 stack...")

    yearly_results: dict = {}
    obj_results: dict = {}

    for lb in F_LOOKBACKS:
        label = f"F{lb}"
        yr_sharpes = {}
        for year in YEARS:
            # Rolling overlays
            brd_pct = rolling_percentile(breadth_raw[year], PCT_WINDOW)
            fdo_pct = rolling_percentile(fund_disp_raw[year], PCT_WINDOW)
            fts_pct = rolling_percentile(fts_raw[year], FTS_PCT_WINDOW)

            sig_rets = {sk: base_sig_rets[sk][year] for sk in BASE_SIGS}
            sig_rets[F_SIG] = f_sig_rets[lb][year]

            ens = compute_ensemble(sig_rets, btc_vols[year], brd_pct, fdo_pct, fts_pct)
            yr_sharpes[year] = sharpe(ens)

        obj = obj_func(list(yr_sharpes.values()))
        yearly_results[label] = yr_sharpes
        obj_results[label] = obj
        delta = round(obj - BASELINE_OBJ, 4)
        flag = " ← BASELINE" if lb == 144 else (" ✅" if obj > BASELINE_OBJ else "")
        print(f"  {label:6s} OBJ={obj:.4f} Δ={delta:+.4f} | {yr_sharpes}{flag}")

        _partial.update({
            "phase": 152,
            "results_so_far": {
                k_: {"yearly": yearly_results[k_], "obj": obj_results[k_]}
                for k_ in yearly_results
            }
        })
        _save(_partial, partial=True)

    # ── Step 3: LOYO validation for best non-baseline ───────────────────
    print("\n[3/3] LOYO validation for best variant...")

    baseline_label = "F144"
    best_label = max(
        [l for l in obj_results if l != baseline_label],
        key=lambda l: obj_results[l],
        default=None,
    )
    best_obj = obj_results.get(best_label, -999)
    best_lb  = int(best_label[1:]) if best_label else 144

    loyo_wins = 0
    loyo_deltas = []
    loyo_detail = {}
    if best_label and best_obj > BASELINE_OBJ:
        print(f"  Running LOYO for {best_label} (OBJ={best_obj:.4f})...")
        for held_out in YEARS:
            train_years = [y for y in YEARS if y != held_out]
            train_ens_base = []
            train_ens_best = []
            for yr in train_years:
                brd_pct = rolling_percentile(breadth_raw[yr], PCT_WINDOW)
                fdo_pct = rolling_percentile(fund_disp_raw[yr], PCT_WINDOW)
                fts_pct = rolling_percentile(fts_raw[yr], FTS_PCT_WINDOW)

                sig_base = {sk: base_sig_rets[sk][yr] for sk in BASE_SIGS}
                sig_base[F_SIG] = f_sig_rets[144][yr]
                ens_b = compute_ensemble(sig_base, btc_vols[yr], brd_pct, fdo_pct, fts_pct)
                train_ens_base.extend(ens_b.tolist())

                sig_best = {sk: base_sig_rets[sk][yr] for sk in BASE_SIGS}
                sig_best[F_SIG] = f_sig_rets[best_lb][yr]
                ens_x = compute_ensemble(sig_best, btc_vols[yr], brd_pct, fdo_pct, fts_pct)
                train_ens_best.extend(ens_x.tolist())

            s_base = sharpe(np.array(train_ens_base))
            s_best = sharpe(np.array(train_ens_best))
            delta = round(s_best - s_base, 4)
            loyo_deltas.append(delta)
            win = delta > 0
            loyo_wins += int(win)
            loyo_detail[f"loo_{held_out}"] = {"base": s_base, "best": s_best, "delta": delta, "win": win}
            print(f"    LOO-{held_out}: base={s_base:.4f} {best_label}={s_best:.4f} Δ={delta:+.4f} {'✅' if win else '❌'}")
    else:
        print(f"  Best expanded ({best_label}) does not improve OBJ — skipping LOYO.")
        loyo_wins = 0

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("COMPARISON (full P150 stack):")
    baseline_obj = obj_results[baseline_label]
    for label, obj in sorted(obj_results.items(), key=lambda x: -x[1]):
        delta = round(obj - baseline_obj, 4)
        flag = " ← BASELINE" if label == baseline_label else (" ✅" if obj > baseline_obj else "")
        print(f"  {label:6s} OBJ={obj:.4f} Δ={delta:+.4f}{flag}")

    validated = best_obj > BASELINE_OBJ and loyo_wins >= 3
    if validated:
        verdict = (
            f"VALIDATED — {best_label} OBJ={best_obj:.4f} "
            f"(+{best_obj - BASELINE_OBJ:.4f} vs P150 baseline {BASELINE_OBJ}), "
            f"LOYO {loyo_wins}/5"
        )
        next_phase = (
            f"Phase 153: WF validation of {best_label}. "
            f"If passes → update production config funding_lookback_bars={best_lb}."
        )
    elif best_obj > BASELINE_OBJ:
        verdict = (
            f"MARGINAL — {best_label} OBJ={best_obj:.4f} (+{best_obj - BASELINE_OBJ:.4f}) "
            f"but LOYO only {loyo_wins}/5 — insufficient."
        )
        next_phase = (
            "Phase 153: Try ensemble blending (e.g. 50% F144 + 50% F72) or "
            "test adaptive lookback switching (shorter during high-vol regimes)."
        )
    else:
        verdict = (
            f"NO IMPROVEMENT — all F variants <= baseline. "
            f"Best: {best_label} OBJ={best_obj:.4f} vs baseline {BASELINE_OBJ}. "
            "Current F144 is optimal."
        )
        next_phase = (
            "Phase 153: Explore signal-level enhancements: "
            "(a) Open Interest momentum signal, "
            "(b) Taker buy ratio signal (directional demand), "
            "(c) Momentum decay filter (exit winners earlier in 2025). "
            "OR: Accept P150 system (OBJ=2.0079) as production-ready and deploy."
        )

    print(f"\nVERDICT: {verdict}")
    print("=" * 72)

    report = {
        "phase": 152,
        "description": "Funding momentum lookback tuning — F48 to F192",
        "hypothesis": "Shorter F lookback captures faster 2025 funding cycles",
        "elapsed_seconds": round(time.time() - _start, 1),
        "baseline_obj": BASELINE_OBJ,
        "f_lookbacks_tested": F_LOOKBACKS,
        "results": {l: {"yearly": yearly_results[l], "obj": obj_results[l]} for l in yearly_results},
        "best_variant": best_label,
        "best_obj": best_obj,
        "loyo_wins": loyo_wins,
        "loyo_avg_delta": round(float(np.mean(loyo_deltas)), 4) if loyo_deltas else None,
        "loyo_detail": loyo_detail,
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
