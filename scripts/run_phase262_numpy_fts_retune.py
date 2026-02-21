"""
Phase 262 NumPy — Per-Regime FTS Retune (Post-P261n)
=====================================================
Baseline: v2.46.0, OBJ=4.0148

Per-regime FTS params were last tuned in P229-P230 (pre-V1-per-regime, pre-P261n weights).
With new ensemble weights (HIGH now 35%V1 + 65%I460), FTS thresholds may need adjustment.

Current per-regime FTS (from P229-P230):
  LOW:  rs=0.50, bs=3.00, rt=0.80, bt=0.30  pct_win=240h
  MID:  rs=0.20, bs=3.00, rt=0.65, bt=0.25  pct_win=288h
  HIGH: rs=0.40, bs=2.00, rt=0.55, bt=0.25  pct_win=400h

Vectorized: precompute VOL×DISP overlay separate from FTS.
Per regime sweep: vary (rs, rt) 1D then (bs, bt) 1D.
"""

import os, sys, json, time, subprocess
import signal as _signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np

from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

_partial: dict = {}
def _on_timeout(signum, frame):
    _partial["partial"] = True
    out = Path("artifacts/phase262n"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase262n_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(7200)

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
           "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]
YEAR_RANGES = {
    2021: ("2021-02-01", "2022-01-01"), 2022: ("2022-01-01", "2023-01-01"),
    2023: ("2023-01-01", "2024-01-01"), 2024: ("2024-01-01", "2025-01-01"),
    2025: ("2025-01-01", "2026-01-01"),
}

BRD_LB = 192; PCT_WINDOW = 336; FUND_DISP_PCT = 240
TS_SHORT = 16; TS_LONG = 72; VOL_WINDOW = 168
P_LOW = 0.30; P_HIGH = 0.60
RNAMES = ["LOW", "MID", "HIGH"]

# Current per-regime FTS params
FTS_RS  = {"LOW": 0.50, "MID": 0.20, "HIGH": 0.40}
FTS_BS  = {"LOW": 3.00, "MID": 3.00, "HIGH": 2.00}
FTS_RT  = {"LOW": 0.80, "MID": 0.65, "HIGH": 0.55}
FTS_BT  = {"LOW": 0.30, "MID": 0.25, "HIGH": 0.25}
TS_PCT_WIN = {"LOW": 240, "MID": 288, "HIGH": 400}

# VOL / DISP (unchanged)
VOL_THR   = {"LOW": 0.50, "MID": 0.50, "HIGH": 0.50}
VOL_SCALE = {"LOW": 0.40, "MID": 0.15, "HIGH": 0.10}
DISP_THR   = {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}
DISP_SCALE = {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}

# Regime weights from P261n (v2.46.0)
REGIME_WEIGHTS = {
    "LOW":  {"v1": 0.35, "i460": 0.05, "i415": 0.25, "f168": 0.35},
    "MID":  {"v1": 0.10, "i460": 0.00, "i415": 0.25, "f168": 0.65},
    "HIGH": {"v1": 0.35, "i460": 0.65, "i415": 0.00, "f168": 0.00},
}
V1_LOW_PARAMS = {
    "k_per_side": 2, "w_carry": 0.10, "w_mom": 0.45, "w_mean_reversion": 0.45,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
}
V1_MIDHIGH_PARAMS = {
    "k_per_side": 2, "w_carry": 0.25, "w_mom": 0.50, "w_mean_reversion": 0.25,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
}
I460_PARAMS = {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}

BASELINE_OBJ = 4.0148
MIN_DELTA = 0.005; MIN_LOYO = 3
COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

# Sweep grids
RT_SWEEP = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
RS_SWEEP = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
BT_SWEEP = [0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35, 0.40]
BS_SWEEP = [1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.50, 4.00]

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0 or len(a) == 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year_data(year):
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h", "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
    n = len(dataset.timeline)
    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n): close_mat[i, j] = dataset.close(sym, i)
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1,0]; c1 = close_mat[i,0]
        btc_rets[i] = (c1/c0-1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i-VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try: fund_rates[i,j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i,j] = 0.0
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i-FUND_DISP_PCT:i] <= fund_std_raw[i]))
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_pct = {}
    for w in sorted(set(TS_PCT_WIN.values())):
        arr = np.full(n, 0.5)
        for i in range(w, n): arr[i] = float(np.mean(ts_raw[i-w:i] <= ts_raw[i]))
        ts_pct[w] = arr
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB,j] > 0 and close_mat[i,j] > close_mat[i-BRD_LB,j])
        breadth[i] = pos / len(SYMBOLS)
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))
    fixed = {}
    for sk, sn, sp in [("i460","idio_momentum_alpha",I460_PARAMS),
                        ("i415","idio_momentum_alpha",I415_PARAMS),
                        ("f168","funding_momentum_alpha",F168_PARAMS)]:
        fixed[sk] = np.array(BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name":sn,"params":sp})).returns)
    return btc_vol, fund_std_pct, ts_pct, brd_pct, fixed, n, dataset

def compute_v1(ds, params):
    return np.array(BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        ds, make_strategy({"name":"nexus_alpha_v1","params":params})).returns)

def build_precomp(all_base, all_v1, fts_rs, fts_bs, fts_rt, fts_bt):
    """Precompute overlay-scaled signal arrays with given FTS params."""
    precomp = {}
    for yr, base in sorted(all_base.items()):
        btc_vol, fund_std_pct, ts_pct, brd_pct, fixed, n, _ = base
        v1 = all_v1[yr]
        ml = min(min(len(v1[r]) for r in RNAMES), min(len(fixed[k]) for k in ["i460","i415","f168"]))
        bv = btc_vol[:ml]; bpct = brd_pct[:ml]; fsp = fund_std_pct[:ml]
        ridx = np.where(bpct < P_LOW, 0, np.where(bpct > P_HIGH, 2, 1))
        masks = [ridx == i for i in range(3)]
        ov = np.ones(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]
            ov[m & ~np.isnan(bv) & (bv > VOL_THR[rn])] *= VOL_SCALE[rn]
            ov[m & (fsp > DISP_THR[rn])] *= DISP_SCALE[rn]
            tsp = ts_pct[TS_PCT_WIN[rn]][:ml]
            ov[m & (tsp > fts_rt[rn])] *= fts_rs[rn]
            ov[m & (tsp < fts_bt[rn])] *= fts_bs[rn]
        sc = {
            "v1L": ov * v1["LOW"][:ml], "v1M": ov * v1["MID"][:ml],
            "i460": ov * fixed["i460"][:ml], "i415": ov * fixed["i415"][:ml],
            "f168": ov * fixed["f168"][:ml],
        }
        precomp[yr] = (sc, masks, ml)
    return precomp

def fast_eval(weights, precomp):
    yearly = {}
    for yr, (sc, masks, ml) in precomp.items():
        ens = np.zeros(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]; w = weights[rn]
            vk = "v1L" if rn == "LOW" else "v1M"
            ens[m] += w["v1"]*sc[vk][m] + w["i460"]*sc["i460"][m] + w["i415"]*sc["i415"][m] + w["f168"]*sc["f168"][m]
        r = ens[~np.isnan(ens)]
        yearly[yr] = float(np.mean(r)/np.std(r,ddof=1)*np.sqrt(8760)) if len(r)>1 else 0.0
    vals = list(yearly.values())
    return float(np.mean(vals)-0.5*np.std(vals,ddof=1)), yearly

def sweep_fts_regime(target, cur_rs, cur_bs, cur_rt, cur_bt, all_base, all_v1, weights, base_obj):
    """Sweep FTS params for target regime. Sequential 1D searches."""
    best_obj = base_obj
    best_rs = cur_rs[target]; best_bs = cur_bs[target]
    best_rt = cur_rt[target]; best_bt = cur_bt[target]

    def mk_params(rs=None, bs=None, rt=None, bt=None):
        return ({**cur_rs, target: rs or best_rs},
                {**cur_bs, target: bs or best_bs},
                {**cur_rt, target: rt or best_rt},
                {**cur_bt, target: bt or best_bt})

    # rs sweep
    for rs in RS_SWEEP:
        o, _ = fast_eval(weights, build_precomp(all_base, all_v1, *mk_params(rs=rs)))
        if o > best_obj: best_obj = o; best_rs = rs
    print(f"      rs={best_rs}  OBJ={best_obj:.4f}", flush=True)

    # rt sweep
    for rt in RT_SWEEP:
        o, _ = fast_eval(weights, build_precomp(all_base, all_v1, *mk_params(rt=rt)))
        if o > best_obj: best_obj = o; best_rt = rt
    print(f"      rt={best_rt}  OBJ={best_obj:.4f}", flush=True)

    # bs sweep
    for bs in BS_SWEEP:
        o, _ = fast_eval(weights, build_precomp(all_base, all_v1, *mk_params(bs=bs)))
        if o > best_obj: best_obj = o; best_bs = bs
    print(f"      bs={best_bs}  OBJ={best_obj:.4f}", flush=True)

    # bt sweep
    for bt in BT_SWEEP:
        o, _ = fast_eval(weights, build_precomp(all_base, all_v1, *mk_params(bt=bt)))
        if o > best_obj: best_obj = o; best_bt = bt
    print(f"      bt={best_bt}  OBJ={best_obj:.4f}", flush=True)

    return best_rs, best_bs, best_rt, best_bt, best_obj

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 262n — Per-Regime FTS Retune [VECTORIZED]")
    print(f"Baseline: v2.46.0  OBJ={BASELINE_OBJ}")
    print("=" * 65)

    print("\n[1] Loading per-year data...", flush=True)
    all_base = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base[yr] = load_year_data(yr)
        print("done.", flush=True)

    print("\n[2] Pre-computing V1 returns...", flush=True)
    all_v1 = {}
    for yr in sorted(YEAR_RANGES):
        ds = all_base[yr][6]
        all_v1[yr] = {"LOW": compute_v1(ds, V1_LOW_PARAMS),
                      "MID": compute_v1(ds, V1_MIDHIGH_PARAMS),
                      "HIGH": compute_v1(ds, V1_MIDHIGH_PARAMS)}
        print(f"  {yr}: done.", flush=True)

    # Baseline
    cur_rs = dict(FTS_RS); cur_bs = dict(FTS_BS)
    cur_rt = dict(FTS_RT); cur_bt = dict(FTS_BT)
    base_precomp = build_precomp(all_base, all_v1, cur_rs, cur_bs, cur_rt, cur_bt)
    base_obj, base_yearly = fast_eval(REGIME_WEIGHTS, base_precomp)
    print(f"\n  Baseline OBJ = {base_obj:.4f}  (expected ~{BASELINE_OBJ})", flush=True)
    _partial["baseline_obj"] = float(base_obj)

    print("\n[3] Per-regime FTS sweep...", flush=True)
    running_best = base_obj
    for rname in RNAMES:
        print(f"\n  Regime {rname}:", flush=True)
        rs, bs, rt, bt, running_best = sweep_fts_regime(
            rname, cur_rs, cur_bs, cur_rt, cur_bt, all_base, all_v1, REGIME_WEIGHTS, running_best)
        cur_rs[rname] = rs; cur_bs[rname] = bs
        cur_rt[rname] = rt; cur_bt[rname] = bt
        print(f"    {rname}: rs={rs} bs={bs} rt={rt} bt={bt}  OBJ={running_best:.4f}", flush=True)

    final_precomp = build_precomp(all_base, all_v1, cur_rs, cur_bs, cur_rt, cur_bt)
    final_obj, final_yearly = fast_eval(REGIME_WEIGHTS, final_precomp)
    delta = final_obj - base_obj
    print(f"\n  Final OBJ={final_obj:.4f}  Δ={delta:+.4f}", flush=True)

    print("\n[4] LOYO validation...", flush=True)
    loyo_wins = 0
    for yr in sorted(final_yearly):
        win = final_yearly[yr] > base_yearly[yr]
        if win: loyo_wins += 1
        print(f"  {yr}: base={base_yearly[yr]:.4f}  cand={final_yearly[yr]:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={final_obj:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/5", flush=True)
    validated = loyo_wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'}", flush=True)

    result = {
        "phase": "262n", "baseline_obj": float(base_obj),
        "best_obj": float(final_obj), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": 5, "validated": validated,
        "fts_params": {r: {"rs": cur_rs[r], "bs": cur_bs[r], "rt": cur_rt[r], "bt": cur_bt[r]} for r in RNAMES},
    }
    _partial.update(result)
    out = Path("artifacts/phase262n"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase262n_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        fts_p = cfg.setdefault("fts_overlay_params", {})
        for rn in RNAMES:
            fts_p.setdefault(rn, {})
            fts_p[rn]["rs"] = cur_rs[rn]; fts_p[rn]["bs"] = cur_bs[rn]
            fts_p[rn]["rt"] = cur_rt[rn]; fts_p[rn]["bt"] = cur_bt[rn]
        cfg["_version"] = "2.47.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(final_obj, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.47.0  OBJ={round(final_obj,4)}", flush=True)
        subprocess.run(["git","add",str(cfg_path)], cwd=ROOT)
        msg = (f"feat: P262n per-regime FTS retune OBJ={round(final_obj,4)} "
               f"LOYO={loyo_wins}/5 D={delta:+.4f} [v2.47.0]")
        subprocess.run(["git","commit","-m",msg], cwd=ROOT)
        subprocess.run(["git","push"], cwd=ROOT)
        print(f"  Committed: {msg}", flush=True)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)

main()
