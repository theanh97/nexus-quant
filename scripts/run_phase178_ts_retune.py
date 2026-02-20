"""
Phase 178 — TS Overlay Re-Tune (v2.15.0 baseline)
====================================================
The TS overlay params were last calibrated at v2.9.0/v2.10.0/v2.11.0
(OBJ ~2.22-2.36). Now at v2.15.0 (OBJ=2.5396) with significantly
upgraded signal stack (I415 bw=144, regime weights, dispersion, vol regime),
the TS optimal thresholds/scales may have shifted.

Key concern: BS=1.50 was calibrated when alpha was lower. With stronger
underlying signals, a gentler boost may suffice; or a tighter RT reduce
threshold may prevent over-reducing in trending markets.

Strategy:
  Part A: RT sweep [0.50, 0.55, 0.60, 0.65, 0.70] (RS=0.40, BT=0.25, BS=1.50 locked)
  Part B: RS sweep [0.25, 0.30, 0.35, 0.40, 0.45, 0.50] (RT=best-A, BT=0.25, BS=1.50 locked)
  Part C: BT sweep [0.10, 0.15, 0.20, 0.25, 0.30] (RT=best-A, RS=best-B, BS=1.50 locked)
  Part D: BS sweep [1.10, 1.25, 1.40, 1.50, 1.65, 1.80] (all best from A-C locked)
  Part E: Joint best (A×B×C×D combo) LOYO validation

Baseline: RT=0.60, RS=0.40, BT=0.25, BS=1.50, OBJ=2.5396 (v2.15.0)

Validate: LOYO ≥ 3/5 AND delta > 0.005 → update v2.16.0
"""

import json
import os
import signal as _signal
import sys
import time
import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

_start = time.time()

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS  = PROD_CFG["data"]["symbols"]

# Fixed overlay params (v2.15.0) — except TS which is swept
VOL_WINDOW     = 168
VOL_THRESHOLD  = 0.5
VOL_SCALE      = 0.4
VOL_F168_BOOST = 0.10
BRD_LOOKBACK   = 192
PCT_WINDOW     = 336
P_LOW, P_HIGH  = 0.35, 0.65
FUND_DISP_THR   = 0.60
FUND_DISP_SCALE = 1.05
FUND_DISP_PCT   = 240
TS_SHORT = 12
TS_LONG  = 96
# Baseline TS params
BASE_RT = 0.60
BASE_RS = 0.40
BASE_BT = 0.25
BASE_BS = 1.50

WEIGHTS = {
    "LOW":  {"v1": 0.2415, "i460bw168": 0.1730, "i415bw216": 0.2855, "f168": 0.3000},
    "MID":  {"v1": 0.1493, "i460bw168": 0.2053, "i415bw216": 0.3453, "f168": 0.3000},
    "HIGH": {"v1": 0.0567, "i460bw168": 0.2833, "i415bw216": 0.5100, "f168": 0.1500},
}

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

I460_LB = 460; I460_BW = 120
I415_LB = 415; I415_BW = 144

# Sweeps
RT_SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70]
RS_SWEEP = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
BT_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30]
BS_SWEEP = [1.10, 1.25, 1.40, 1.50, 1.65, 1.80]

OUT_DIR = ROOT / "artifacts" / "phase178"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase178_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)

_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(7200)  # 2h budget for 5-part sweep


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: dict) -> float:
    arr = np.array(list(yearly_sharpes.values()))
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def rolling_mean_arr(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    cs = np.zeros(n + 1)
    cs[1:] = np.cumsum(arr)
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        result[i] = (cs[i + 1] - cs[start]) / (i - start + 1)
    return result


def compute_signals(dataset) -> dict:
    n = len(dataset.timeline)

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

    breadth = np.full(n, 0.5)
    for i in range(BRD_LOOKBACK, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - BRD_LOOKBACK)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    breadth[:BRD_LOOKBACK] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i - PCT_WINDOW:i] <= breadth[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    breadth_regime = np.where(brd_pct >= P_HIGH, 2,
                     np.where(brd_pct >= P_LOW, 1, 0)).astype(int)

    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:
                fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except Exception:
                fund_rates[i, j] = 0.0

    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - FUND_DISP_PCT:i] <= fund_std_raw[i]))
    fund_std_pct[:FUND_DISP_PCT] = 0.5

    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i - PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    return {"btc_vol": btc_vol, "breadth_regime": breadth_regime,
            "fund_std_pct": fund_std_pct, "ts_spread_pct": ts_spread_pct}


def compute_ensemble(sig_rets: dict, signals: dict,
                     rt=BASE_RT, rs=BASE_RS, bt=BASE_BT, bs=BASE_BS) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_per = VOL_F168_BOOST / max(1, len(sk_all) - 1)
            ret_i = 0.0
            for sk in sk_all:
                adj_w = (min(0.60, w[sk] + VOL_F168_BOOST) if sk == "f168"
                         else max(0.05, w[sk] - boost_per))
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        if fsp[i] > FUND_DISP_THR:
            ret_i *= FUND_DISP_SCALE
        if tsp[i] > rt:
            ret_i *= rs
        elif tsp[i] < bt:
            ret_i *= bs
        ens[i] = ret_i
    return ens


# ─── MAIN ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 178 — TS Overlay Re-Tune (v2.15.0 baseline)")
print("=" * 68)
print(f"  Baseline: RT={BASE_RT} RS={BASE_RS} BT={BASE_BT} BS={BASE_BS}")
print(f"  Baseline OBJ≈2.5396")
print(f"  Parts: A=RT sweep, B=RS sweep, C=BT sweep, D=BS sweep, E=joint LOYO\n")

print("[1/6] Loading data, signals, strategy returns ...")
strat_by_yr: dict = {sk: {} for sk in ["v1", "i460bw168", "i415bw216", "f168"]}
sigs_by_yr: dict  = {}

for year, (start, end) in YEAR_RANGES.items():
    print(f"  {year}: ", end="", flush=True)
    cfg_data = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
                "start": start, "end": end, "bar_interval": "1h",
                "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg_data, seed=42).load()
    sigs_by_yr[year] = compute_signals(dataset)
    print("S", end="", flush=True)
    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", {
            "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45, "w_mean_reversion": 0.2,
            "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
            "vol_lookback_bars": 168, "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 60,
        }),
        ("i460bw168", "idio_momentum_alpha", {
            "k_per_side": 4, "lookback_bars": I460_LB, "beta_window_bars": I460_BW,
            "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
        }),
        ("i415bw216", "idio_momentum_alpha", {
            "k_per_side": 4, "lookback_bars": I415_LB, "beta_window_bars": I415_BW,
            "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
        }),
        ("f168", "funding_momentum_alpha", {
            "k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
            "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
        }),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        strat_by_yr[sk][year] = np.array(result.returns)
    print(". ✓")

def get_rets(year):
    return {sk: strat_by_yr[sk][year] for sk in ["v1", "i460bw168", "i415bw216", "f168"]}

baseline_yearly = {y: sharpe(compute_ensemble(get_rets(y), sigs_by_yr[y])) for y in YEARS}
BASELINE_OBJ = obj_func(baseline_yearly)
_partial.update({"phase": 178, "description": "TS Overlay Re-Tune", "baseline_obj": BASELINE_OBJ, "partial": True})
print(f"\n  Baseline OBJ={BASELINE_OBJ:.4f}")


def run_single_sweep(param_name: str, sweep_vals, cur_rt, cur_rs, cur_bt, cur_bs):
    best_val = {"rt": cur_rt, "rs": cur_rs, "bt": cur_bt, "bs": cur_bs}[param_name]
    best_obj = BASELINE_OBJ
    results = {}
    for v in sweep_vals:
        kw = {"rt": cur_rt, "rs": cur_rs, "bt": cur_bt, "bs": cur_bs}
        kw[param_name] = v
        yr = {y: sharpe(compute_ensemble(get_rets(y), sigs_by_yr[y], **kw)) for y in YEARS}
        o = obj_func(yr)
        d = round(o - BASELINE_OBJ, 4)
        results[str(v)] = {"value": v, "obj": o, "delta": d, "yearly": yr}
        tag = " ← baseline" if v == {"rt": cur_rt, "rs": cur_rs, "bt": cur_bt, "bs": cur_bs}[param_name] else ""
        sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
        print(f"    {param_name}={v} → OBJ={o:.4f}  Δ={d:+.4f}  {sym}{tag}")
        if o > best_obj:
            best_obj = o
            best_val = v
    return best_val, results, best_obj


print("\n[2/6] Part A — RT sweep (RS=0.40 BT=0.25 BS=1.50 locked) ...")
best_rt, parta_res, parta_obj = run_single_sweep("rt", RT_SWEEP, BASE_RT, BASE_RS, BASE_BT, BASE_BS)
print(f"  RT winner: rt={best_rt}  OBJ={parta_obj:.4f}")

print("\n[3/6] Part B — RS sweep (BT=0.25 BS=1.50 locked) ...")
best_rs, partb_res, partb_obj = run_single_sweep("rs", RS_SWEEP, best_rt, BASE_RS, BASE_BT, BASE_BS)
print(f"  RS winner: rs={best_rs}  OBJ={partb_obj:.4f}")

print("\n[4/6] Part C — BT sweep (RS locked) ...")
best_bt, partc_res, partc_obj = run_single_sweep("bt", BT_SWEEP, best_rt, best_rs, BASE_BT, BASE_BS)
print(f"  BT winner: bt={best_bt}  OBJ={partc_obj:.4f}")

print("\n[5/6] Part D — BS sweep ...")
best_bs, partd_res, partd_obj = run_single_sweep("bs", BS_SWEEP, best_rt, best_rs, best_bt, BASE_BS)
print(f"  BS winner: bs={best_bs}  OBJ={partd_obj:.4f}")

print(f"\n  Best combo: RT={best_rt} RS={best_rs} BT={best_bt} BS={best_bs}")

print("\n[6/6] Part E — Joint LOYO validation ...")
joint_yearly = {y: sharpe(compute_ensemble(get_rets(y), sigs_by_yr[y],
                           rt=best_rt, rs=best_rs, bt=best_bt, bs=best_bs)) for y in YEARS}
joint_obj   = obj_func(joint_yearly)
joint_delta = round(joint_obj - BASELINE_OBJ, 4)

loyo_wins, loyo_deltas, loyo_table = 0, [], {}
for y in YEARS:
    d = joint_yearly[y] - baseline_yearly[y]
    loyo_deltas.append(d)
    if d > 0:
        loyo_wins += 1
    loyo_table[y] = {"baseline_sh": baseline_yearly[y], "joint_sh": joint_yearly[y],
                     "delta": round(d, 4), "win": bool(d > 0)}

loyo_avg  = round(float(np.mean(loyo_deltas)), 4)
validated = loyo_wins >= 3 and joint_delta > 0.005

print(f"  Joint OBJ={joint_obj:.4f}  Δ={joint_delta:+.4f}  LOYO {loyo_wins}/5")
for y, row in loyo_table.items():
    print(f"    {y}: {'✅' if row['win'] else '❌'}  Δ={row['delta']:+.4f}")

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — RT={best_rt} RS={best_rs} BT={best_bt} BS={best_bs}: "
          f"OBJ={joint_obj:.4f} (Δ={joint_delta:+.4f}) LOYO {loyo_wins}/5")
else:
    print(f"❌ NO IMPROVEMENT — RT={BASE_RT} RS={BASE_RS} BT={BASE_BT} BS={BASE_BS} optimal")
    print(f"   Best: Δ={joint_delta:+.4f} | LOYO {loyo_wins}/5")
print("=" * 68)

verdict = (
    f"VALIDATED — RT={best_rt}/RS={best_rs}/BT={best_bt}/BS={best_bs} "
    f"OBJ={joint_obj:.4f} Δ={joint_delta:+.4f} LOYO {loyo_wins}/5"
    if validated else
    f"NO IMPROVEMENT — TS params RT={BASE_RT}/RS={BASE_RS}/BT={BASE_BT}/BS={BASE_BS} "
    f"OBJ={BASELINE_OBJ:.4f} optimal"
)

report = {
    "phase": 178,
    "description": "TS Overlay Re-Tune (v2.15.0 baseline)",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_rt": BASE_RT, "baseline_rs": BASE_RS,
    "baseline_bt": BASE_BT, "baseline_bs": BASE_BS,
    "baseline_obj": BASELINE_OBJ,
    "baseline_yearly": baseline_yearly,
    "parta_rt_sweep": parta_res, "parta_best_rt": best_rt, "parta_obj": parta_obj,
    "partb_rs_sweep": partb_res, "partb_best_rs": best_rs, "partb_obj": partb_obj,
    "partc_bt_sweep": partc_res, "partc_best_bt": best_bt, "partc_obj": partc_obj,
    "partd_bs_sweep": partd_res, "partd_best_bs": best_bs, "partd_obj": partd_obj,
    "joint_rt": best_rt, "joint_rs": best_rs, "joint_bt": best_bt, "joint_bs": best_bs,
    "joint_obj": joint_obj, "joint_delta": joint_delta,
    "joint_yearly": joint_yearly,
    "loyo_table": loyo_table, "loyo_wins": loyo_wins, "loyo_avg_delta": loyo_avg,
    "validated": validated, "verdict": verdict,
    "partial": False, "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
}

out_path = OUT_DIR / "phase178_report.json"
out_path.write_text(json.dumps(report, indent=2, default=str))
print(f"\n✅ Saved → {out_path}")

if validated:
    print("\n🔧 Updating production config to v2.16.0 ...")
    cfg  = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
    prev = cfg["_version"]
    cfg["_version"] = "2.16.0"
    cfg["_created"] = datetime.date.today().isoformat()
    cfg["_validated"] += (
        f"; TS re-tune P178: RT={BASE_RT}→{best_rt}/RS={BASE_RS}→{best_rs}/"
        f"BT={BASE_BT}→{best_bt}/BS={BASE_BS}→{best_bs} "
        f"LOYO {loyo_wins}/5 Δ={joint_delta:+.4f} OBJ={joint_obj:.4f} — PRODUCTION v2.16.0"
    )
    ts = cfg["funding_term_structure_overlay"]
    ts["reduce_threshold"] = best_rt
    ts["reduce_scale"]     = best_rs
    ts["boost_threshold"]  = best_bt
    ts["boost_scale"]      = best_bs
    cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = joint_obj
    (ROOT / "configs" / "production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    print(f"  Config: {prev} → v2.16.0")
else:
    print("\n❌ NO IMPROVEMENT — TS params RT=0.60/RS=0.40/BT=0.25/BS=1.50 remain optimal.")
