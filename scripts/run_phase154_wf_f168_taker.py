#!/usr/bin/env python3
"""
Phase 154: WF Validation of F168 + Taker Buy Alpha as 5th Signal
=================================================================
Production state: OBJ≈2.07 (v2.6.0: F168 confirmed, breadth+vol+fund_disp+fund_ts)
Phase 153: F168 +0.032 on v2.5.0 stack; rebalance compression harmful.

Two validation tracks:

TRACK A — F168 Walk-Forward Validation
  WF Window 1: Train 2021-2023 → Test 2024
  WF Window 2: Train 2022-2024 → Test 2025
  Compare F168 vs F144 on BOTH windows — need Δ > 0 in at least 2/2 for CONFIRM

TRACK B — Taker Buy Ratio as 5th Signal
  taker_buy_alpha: ratio_lookback=48h, k=2, leverage=0.20
  Blend weight: add 10% taker_buy, reduce others proportionally
  Requires: dataset.taker_buy_volume and dataset.perp_volume (may not be cached)
  If data unavailable → mark SKIP and log note.

Pass: WF 2/2 F168 wins → confirms v2.6.0 prod update.
      Taker: OBJ > current + LOYO 3/5 → Phase 155 full test.
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
_signal.alarm(3000)  # 50min

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]

VOL_OV = PROD_CFG.get("vol_regime_overlay", {})
VOL_WINDOW     = VOL_OV.get("window_bars", 168)
VOL_THRESHOLD  = VOL_OV.get("threshold", 0.5)
VOL_SCALE      = VOL_OV.get("scale_factor", 0.5)
VOL_F144_BOOST = VOL_OV.get("f144_boost", 0.2)

BRS = PROD_CFG.get("breadth_regime_switching", {})
BREADTH_LB = BRS.get("breadth_lookback_bars", 168)
PCT_WINDOW = BRS.get("rolling_percentile_window", 336)
P_LOW      = BRS.get("p_low", 0.33)
P_HIGH     = BRS.get("p_high", 0.67)

FDO = PROD_CFG.get("funding_dispersion_overlay", {})
FUND_DISP_PCT   = FDO.get("boost_threshold_pct", 0.75)
FUND_DISP_SCALE = FDO.get("boost_scale", 1.15)

FTS = PROD_CFG.get("funding_term_structure_overlay", {})
FTS_SHORT_W = FTS.get("short_window_bars", 24)
FTS_LONG_W  = FTS.get("long_window_bars", 144)
FTS_PCT_WIN = FTS.get("rolling_percentile_window", 336)
FTS_RT      = FTS.get("reduce_threshold", 0.70)
FTS_RS      = FTS.get("reduce_scale", 0.60)
FTS_BT      = FTS.get("boost_threshold", 0.30)
FTS_BS      = FTS.get("boost_scale", 1.15)

WEIGHTS = {
    "prod":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039},
    "p143b": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f144": 0.25},
    "mid":   {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f144": 0.25},
}

# 5-signal weight sets (adding taker_buy_alpha at 10%, reduce others proportionally)
WEIGHTS_5SIG = {
    "prod":  {"v1": 0.2472, "i460bw168": 0.1770, "i415bw216": 0.2922, "f144": 0.1836, "taker": 0.10},
    "p143b": {"v1": 0.045,  "i460bw168": 0.225,  "i415bw216": 0.405,  "f144": 0.225,  "taker": 0.10},
    "mid":   {"v1": 0.144,  "i460bw168": 0.198,  "i415bw216": 0.333,  "f144": 0.225,  "taker": 0.10},
}

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

# WF Windows: train on 3 years, test on 1
WF_WINDOWS = [
    {"train": ["2021", "2022", "2023"], "test": "2024"},
    {"train": ["2022", "2023", "2024"], "test": "2025"},
]

BASELINE_OBJ = 2.0761  # F168 on v2.5.0 from P153
F168_LB = int(PROD_CFG["ensemble"]["signals"]["f144"]["params"]["funding_lookback_bars"])  # 168 after v2.6.0
F144_LB = 144

OUT_DIR = ROOT / "artifacts" / "phase154"
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
    out = OUT_DIR / "phase154_report.json"
    out.write_text(json.dumps(data, indent=2, default=str))
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


def compute_fund_disp(dataset) -> np.ndarray:
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


def compute_fund_ts(dataset) -> np.ndarray:
    n = len(dataset.timeline)
    level = np.zeros(n)
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
        level[i] = float(np.mean(rates)) if rates else 0.0
    spread = np.zeros(n)
    for i in range(FTS_LONG_W, n):
        s_avg = float(np.mean(level[max(0, i - FTS_SHORT_W):i]))
        l_avg = float(np.mean(level[i - FTS_LONG_W:i]))
        spread[i] = s_avg - l_avg
    if FTS_LONG_W < n:
        spread[:FTS_LONG_W] = spread[FTS_LONG_W]
    return spread


def blend(sig_rets: dict, bv, brd_pct, fdo_pct, fts_pct, weight_table) -> np.ndarray:
    """Full v2.6.0 production ensemble with configurable weight table."""
    sig_keys = list(sig_rets.keys())
    f_key = "f144"  # weight key is always "f144" in WEIGHTS
    min_len = min(len(sig_rets[sk]) for sk in sig_keys)
    bv_  = bv[:min_len]
    brd_ = brd_pct[:min_len]
    fdo_ = fdo_pct[:min_len]
    fts_ = fts_pct[:min_len]
    ens  = np.zeros(min_len)
    n_others = len(sig_keys) - 1

    for i in range(min_len):
        if brd_[i] >= P_HIGH:
            w = weight_table["p143b"]
        elif brd_[i] >= P_LOW:
            w = weight_table["mid"]
        else:
            w = weight_table["prod"]

        if not np.isnan(bv_[i]) and bv_[i] > VOL_THRESHOLD:
            boost_other = VOL_F144_BOOST / max(1, n_others)
            ret_i = 0.0
            for sk in sig_keys:
                w_key = "taker" if sk == "taker_buy_alpha" else sk
                if w_key == f_key:
                    adj_w = min(0.60, w.get(w_key, 0.0) + VOL_F144_BOOST)
                else:
                    adj_w = max(0.0, w.get(w_key, 0.0) - boost_other)
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w.get("taker" if sk == "taker_buy_alpha" else sk, 0.0) * sig_rets[sk][i] for sk in sig_keys)

        if fdo_[i] > FUND_DISP_PCT:
            ret_i *= FUND_DISP_SCALE

        sp = fts_[i]
        if sp >= FTS_RT:
            ret_i *= FTS_RS
        elif sp <= FTS_BT:
            ret_i *= FTS_BS

        ens[i] = ret_i

    return ens


def load_year(year):
    start, end = YEAR_RANGES[year]
    cfg_data = {
        "provider": "binance_rest_v1", "symbols": SYMBOLS,
        "start": start, "end": end, "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    provider = make_provider(cfg_data, seed=42)
    return provider.load()


def main():
    global _partial

    print("=" * 72)
    print("PHASE 154: WF Validation F168 + Taker Buy 5th Signal")
    print(f"  F168_LB={F168_LB} (production), F144_LB={F144_LB} (comparison)")
    print("=" * 72)

    # ── Load all years ───────────────────────────────────────────────────
    print("\n[1/4] Loading datasets...")
    datasets = {}
    btc_vols, breadth_raw, fund_disp_raw, fts_raw = {}, {}, {}, {}
    for year in YEARS:
        print(f"  {year}: ", end="", flush=True)
        ds = load_year(year)
        datasets[year]       = ds
        btc_vols[year]       = compute_btc_vol(ds)
        breadth_raw[year]    = compute_breadth(ds)
        fund_disp_raw[year]  = compute_fund_disp(ds)
        fts_raw[year]        = compute_fund_ts(ds)
        print("✓")

    # ── Track A: WF Validation of F168 vs F144 ──────────────────────────
    print("\n[2/4] TRACK A — Walk-Forward Validation F168 vs F144...")

    def run_year_with_f(year, f_lb):
        brd_pct = rolling_percentile(breadth_raw[year], PCT_WINDOW)
        fdo_pct = rolling_percentile(fund_disp_raw[year], PCT_WINDOW)
        fts_pct = rolling_percentile(fts_raw[year], FTS_PCT_WIN)
        sig_rets = {}
        for sk in ["v1", "i460bw168", "i415bw216"]:
            sig_def = SIGNAL_DEFS[sk]
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": sig_def["strategy"], "params": dict(sig_def["params"])})
            result = BacktestEngine(bt_cfg).run(datasets[year], strat)
            sig_rets[sk] = np.array(result.returns, dtype=np.float64)
        f_params = dict(SIGNAL_DEFS["f144"]["params"])
        f_params["funding_lookback_bars"] = f_lb
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat = make_strategy({"name": SIGNAL_DEFS["f144"]["strategy"], "params": f_params})
        result = BacktestEngine(bt_cfg).run(datasets[year], strat)
        sig_rets["f144"] = np.array(result.returns, dtype=np.float64)
        ens = blend(sig_rets, btc_vols[year], brd_pct, fdo_pct, fts_pct, WEIGHTS)
        return sharpe(ens)

    wf_results = []
    wf_wins = 0
    for wf in WF_WINDOWS:
        train_yrs = wf["train"]
        test_yr   = wf["test"]
        print(f"\n  WF: train={train_yrs} test={test_yr}")

        train_ens_f144 = []
        train_ens_f168 = []
        for yr in train_yrs:
            print(f"    train {yr}: ", end="", flush=True)
            s144 = run_year_with_f(yr, F144_LB)
            s168 = run_year_with_f(yr, F168_LB)
            train_ens_f144.append(s144)
            train_ens_f168.append(s168)
            print(f"F144={s144:.4f} F168={s168:.4f}")

        print(f"    test  {test_yr}: ", end="", flush=True)
        test_s144 = run_year_with_f(test_yr, F144_LB)
        test_s168 = run_year_with_f(test_yr, F168_LB)
        delta = round(test_s168 - test_s144, 4)
        win = delta > 0
        wf_wins += int(win)
        print(f"F144={test_s144:.4f} F168={test_s168:.4f} Δ={delta:+.4f} {'✅' if win else '❌'}")

        wf_results.append({
            "train": train_yrs, "test": test_yr,
            "train_sharpes_f144": train_ens_f144,
            "train_sharpes_f168": train_ens_f168,
            "test_sharpe_f144": test_s144,
            "test_sharpe_f168": test_s168,
            "test_delta": delta, "win": win,
        })

    wf_avg_delta = round(float(np.mean([r["test_delta"] for r in wf_results])), 4)
    f168_wf_passes = wf_wins == 2
    print(f"\n  WF Summary: {wf_wins}/2 wins, avg_delta={wf_avg_delta:+.4f} → {'PASS ✅' if f168_wf_passes else 'FAIL ❌'}")
    _partial.update({"track_a_wf": wf_results, "wf_wins": wf_wins, "wf_avg_delta": wf_avg_delta})
    _save(_partial, partial=True)

    # ── Track B: Taker Buy Alpha as 5th Signal ──────────────────────────
    print("\n[3/4] TRACK B — Taker Buy Alpha as 5th signal...")

    taker_available = False
    taker_results = {}
    taker_obj = 0.0

    # Check if taker data is available
    try:
        ds2021 = datasets["2021"]
        has_taker = (
            hasattr(ds2021, "taker_buy_volume")
            and ds2021.taker_buy_volume is not None
            and len(ds2021.taker_buy_volume) > 0
        )
        if not has_taker:
            print("  ⚠️ taker_buy_volume not available in dataset — SKIP Track B")
        else:
            taker_available = True
    except Exception as e:
        print(f"  ⚠️ taker check error: {e} — SKIP Track B")

    if taker_available:
        taker_params = {
            "k_per_side": 2,
            "ratio_lookback_bars": 48,
            "vol_lookback_bars": 168,
            "target_gross_leverage": 0.20,
            "rebalance_interval_bars": 24,
        }
        yr_sharpes_taker = {}
        yr_sharpes_base  = {}
        for year in YEARS:
            brd_pct = rolling_percentile(breadth_raw[year], PCT_WINDOW)
            fdo_pct = rolling_percentile(fund_disp_raw[year], PCT_WINDOW)
            fts_pct = rolling_percentile(fts_raw[year], FTS_PCT_WIN)

            sig_rets_base = {}
            sig_rets_5    = {}
            for sk in ["v1", "i460bw168", "i415bw216"]:
                sig_def = SIGNAL_DEFS[sk]
                bt_cfg = BacktestConfig(costs=COST_MODEL)
                strat = make_strategy({"name": sig_def["strategy"], "params": dict(sig_def["params"])})
                result = BacktestEngine(bt_cfg).run(datasets[year], strat)
                rets = np.array(result.returns, dtype=np.float64)
                sig_rets_base[sk] = rets
                sig_rets_5[sk] = rets

            # F168 for base
            f_params = dict(SIGNAL_DEFS["f144"]["params"])
            f_params["funding_lookback_bars"] = F168_LB
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": SIGNAL_DEFS["f144"]["strategy"], "params": f_params})
            result = BacktestEngine(bt_cfg).run(datasets[year], strat)
            f_rets = np.array(result.returns, dtype=np.float64)
            sig_rets_base["f144"] = f_rets
            sig_rets_5["f144"]    = f_rets

            # Taker buy
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": "taker_buy_alpha", "params": taker_params})
            result = BacktestEngine(bt_cfg).run(datasets[year], strat)
            sig_rets_5["taker_buy_alpha"] = np.array(result.returns, dtype=np.float64)

            ens_base = blend(sig_rets_base, btc_vols[year], brd_pct, fdo_pct, fts_pct, WEIGHTS)
            ens_5    = blend(sig_rets_5, btc_vols[year], brd_pct, fdo_pct, fts_pct, WEIGHTS_5SIG)
            yr_sharpes_base[year]  = sharpe(ens_base)
            yr_sharpes_taker[year] = sharpe(ens_5)
            print(f"  {year}: base={yr_sharpes_base[year]:.4f} taker5={yr_sharpes_taker[year]:.4f} Δ={yr_sharpes_taker[year]-yr_sharpes_base[year]:+.4f}")

        base_obj  = obj_func(list(yr_sharpes_base.values()))
        taker_obj = obj_func(list(yr_sharpes_taker.values()))
        taker_delta = round(taker_obj - base_obj, 4)
        loyo_wins_taker = sum(1 for yr in YEARS if yr_sharpes_taker[yr] > yr_sharpes_base[yr])
        taker_results = {
            "base_yearly": yr_sharpes_base, "base_obj": base_obj,
            "taker_yearly": yr_sharpes_taker, "taker_obj": taker_obj,
            "delta": taker_delta, "loyo_wins": loyo_wins_taker,
        }
        print(f"\n  Taker5 OBJ={taker_obj:.4f} vs base={base_obj:.4f} Δ={taker_delta:+.4f} | LOYO {loyo_wins_taker}/5")
    else:
        taker_results = {"skipped": True, "reason": "taker_buy_volume not in dataset"}
        print("  Track B: SKIPPED (taker data unavailable)")

    _partial.update({"track_b_taker": taker_results})
    _save(_partial, partial=True)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 154 SUMMARY:")
    print(f"  TRACK A (F168 WF): {wf_wins}/2 wins, avg_delta={wf_avg_delta:+.4f} → {'CONFIRMED ✅' if f168_wf_passes else 'WEAK ⚠️'}")
    if taker_available:
        loyo_t = taker_results.get("loyo_wins", 0)
        td = taker_results.get("delta", 0)
        print(f"  TRACK B (Taker 5th): OBJ delta={td:+.4f} LOYO={loyo_t}/5 → {'EXPLORE ✅' if td > 0 and loyo_t >= 3 else 'NO IMPROVEMENT'}")
    else:
        print("  TRACK B: SKIPPED — taker data not cached")

    # Verdict
    if f168_wf_passes:
        wf_verdict = f"F168 WF CONFIRMED — {wf_wins}/2 wins, avg_delta={wf_avg_delta:+.4f}. Production v2.6.0 validated."
    else:
        wf_verdict = f"F168 WF WEAK — {wf_wins}/2 wins. Keep v2.6.0 (consistent IS wins) but flag for live monitoring."

    taker_verdict = "SKIPPED"
    if taker_available:
        td = taker_results.get("delta", 0)
        lt = taker_results.get("loyo_wins", 0)
        taker_verdict = (
            f"EXPLORE: taker5 Δ={td:+.4f} LOYO={lt}/5 → Phase 155 full test."
            if td > 0 and lt >= 3
            else f"NO IMPROVEMENT: taker5 Δ={td:+.4f} LOYO={lt}/5"
        )

    next_phase = (
        "Phase 155: Open Interest (OI) momentum signal — test if OI delta direction predicts returns. "
        "OI data from Binance Vision or provider. Also consider momentum decay filter."
        if not taker_available or taker_results.get("delta", 0) <= 0
        else f"Phase 155: Full LOYO+WF validation of taker_buy_alpha as 5th signal."
    )

    print(f"\nVERDICT A: {wf_verdict}")
    print(f"VERDICT B: {taker_verdict}")
    print(f"NEXT: {next_phase}")
    print("=" * 72)

    report = {
        "phase": 154,
        "description": "WF validation of F168 + taker buy 5th signal test",
        "elapsed_seconds": round(time.time() - _start, 1),
        "track_a_wf_f168": {"results": wf_results, "wins": wf_wins, "avg_delta": wf_avg_delta, "pass": f168_wf_passes},
        "track_b_taker": taker_results,
        "verdict_a": wf_verdict,
        "verdict_b": taker_verdict,
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
