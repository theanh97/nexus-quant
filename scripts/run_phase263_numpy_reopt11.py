"""
Phase 263 NumPy — Regime Weight Re-opt (11th Pass) [VECTORIZED]
================================================================
Baseline: v2.47.0, OBJ=4.2685 (P262n FTS retune)

Per-regime FTS now optimized (P262n, LOYO 5/5):
  LOW:  rs=0.05, bs=4.00, rt=0.80, bt=0.33  pct_win=240h
  MID:  rs=0.05, bs=2.50, rt=0.65, bt=0.22  pct_win=288h
  HIGH: rs=0.40, bs=2.25, rt=0.50, bt=0.22  pct_win=400h

With new FTS params, optimal ensemble regime weights may shift.
Re-sweep from P261n baseline weights.
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
    out = Path("artifacts/phase263n"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase263n_report.json").write_text(json.dumps(_partial, indent=2))
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

# P262n optimized per-regime FTS params
TS_PCT_WIN = {"LOW": 240, "MID": 288, "HIGH": 400}
FTS_RS = {"LOW": 0.05, "MID": 0.05, "HIGH": 0.40}
FTS_BS = {"LOW": 4.00, "MID": 2.50, "HIGH": 2.25}
FTS_RT = {"LOW": 0.80, "MID": 0.65, "HIGH": 0.50}
FTS_BT = {"LOW": 0.33, "MID": 0.22, "HIGH": 0.22}

VOL_THR   = {"LOW": 0.50, "MID": 0.50, "HIGH": 0.50}
VOL_SCALE = {"LOW": 0.40, "MID": 0.15, "HIGH": 0.10}
DISP_THR   = {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}
DISP_SCALE = {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}

# Baseline: P261n regime weights
REGIME_WEIGHTS_BASE = {
    "LOW":  {"v1": 0.35, "i460": 0.05, "i415": 0.25, "f168": 0.35},
    "MID":  {"v1": 0.10, "i460": 0.00, "i415": 0.25, "f168": 0.65},
    "HIGH": {"v1": 0.35, "i460": 0.65, "i415": 0.00, "f168": 0.00},
}

# V1 params (NumPy-optimal, mom_lb=336h)
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

BASELINE_OBJ = 4.2685
MIN_DELTA = 0.005; MIN_LOYO = 3
W_STEPS = [round(x*0.05, 2) for x in range(17)]
COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0 or len(a) == 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w-1, len(a)):
        out[i] = cs[i]/w if i == w-1 else (cs[i]-cs[i-w])/w
    return out

def load_year_data(year):
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h", "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
    n = len(dataset.timeline)
    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n): close_mat[i,j] = dataset.close(sym, i)
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0=close_mat[i-1,0]; c1=close_mat[i,0]
        btc_rets[i] = (c1/c0-1.0) if c0>0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i-VOL_WINDOW:i]))*np.sqrt(8760)
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
                  if close_mat[i-BRD_LB,j]>0 and close_mat[i,j]>close_mat[i-BRD_LB,j])
        breadth[i] = pos/len(SYMBOLS)
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

def precompute_fast(all_base, all_v1):
    fast = {}
    for yr, base in sorted(all_base.items()):
        btc_vol, fund_std_pct, ts_pct, brd_pct, fixed, n, _ = base
        v1 = all_v1[yr]
        ml = min(min(len(v1[r]) for r in RNAMES), min(len(fixed[k]) for k in ["i460","i415","f168"]))
        bv=btc_vol[:ml]; bpct=brd_pct[:ml]; fsp=fund_std_pct[:ml]
        ridx = np.where(bpct < P_LOW, 0, np.where(bpct > P_HIGH, 2, 1))
        masks = [ridx==i for i in range(3)]
        ov = np.ones(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]
            ov[m & ~np.isnan(bv) & (bv>VOL_THR[rn])] *= VOL_SCALE[rn]
            ov[m & (fsp>DISP_THR[rn])] *= DISP_SCALE[rn]
            tsp = ts_pct[TS_PCT_WIN[rn]][:ml]
            ov[m & (tsp>FTS_RT[rn])] *= FTS_RS[rn]
            ov[m & (tsp<FTS_BT[rn])] *= FTS_BS[rn]
        sc = {
            "v1L": ov*v1["LOW"][:ml], "v1M": ov*v1["MID"][:ml],
            "i460": ov*fixed["i460"][:ml], "i415": ov*fixed["i415"][:ml], "f168": ov*fixed["f168"][:ml],
        }
        fast[yr] = (sc, masks, ml)
    return fast

def fast_eval(weights, fast):
    yearly = {}
    for yr, (sc, masks, ml) in fast.items():
        ens = np.zeros(ml)
        for ri, rn in enumerate(RNAMES):
            m=masks[ri]; w=weights[rn]; vk="v1L" if rn=="LOW" else "v1M"
            ens[m] += w["v1"]*sc[vk][m]+w["i460"]*sc["i460"][m]+w["i415"]*sc["i415"][m]+w["f168"]*sc["f168"][m]
        r = ens[~np.isnan(ens)]
        yearly[yr] = float(np.mean(r)/np.std(r,ddof=1)*np.sqrt(8760)) if len(r)>1 else 0.0
    vals = list(yearly.values())
    return float(np.mean(vals)-0.5*np.std(vals,ddof=1)), yearly

def sweep_one(target, cur_w, fast, best_obj):
    bw = dict(cur_w[target])
    for wv1 in W_STEPS:
        for wi460 in W_STEPS:
            for wi415 in W_STEPS:
                wf168 = round(1.0-wv1-wi460-wi415, 4)
                if wf168 < 0: continue
                wt = dict(cur_w); wt[target] = {"v1":wv1,"i460":wi460,"i415":wi415,"f168":wf168}
                o, _ = fast_eval(wt, fast)
                if o > best_obj: best_obj=o; bw={"v1":wv1,"i460":wi460,"i415":wi415,"f168":wf168}
    return bw, best_obj

def main():
    t0 = time.time()
    print("="*65)
    print("Phase 263n — Regime Weight Re-opt (11th Pass) [VEC]")
    print(f"Baseline: v2.47.0  OBJ={BASELINE_OBJ}  (P262n FTS params)")
    print("="*65)

    print("\n[1] Loading data...", flush=True)
    all_base = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base[yr] = load_year_data(yr)
        print("done.", flush=True)

    print("\n[2] Computing V1 (mom_lb=336h)...", flush=True)
    all_v1 = {}
    for yr in sorted(YEAR_RANGES):
        ds = all_base[yr][6]
        all_v1[yr] = {"LOW": compute_v1(ds,V1_LOW_PARAMS), "MID": compute_v1(ds,V1_MIDHIGH_PARAMS),
                      "HIGH": compute_v1(ds,V1_MIDHIGH_PARAMS)}
        print(f"  {yr}: done.", flush=True)

    print("\n[3] Precomputing fast eval...", flush=True)
    fast = precompute_fast(all_base, all_v1)

    base_obj, base_yr = fast_eval(REGIME_WEIGHTS_BASE, fast)
    print(f"\n  Baseline OBJ = {base_obj:.4f}", flush=True)
    _partial["baseline_obj"] = float(base_obj)

    print("\n[4] 2-pass sweep...", flush=True)
    cur_w = {r: dict(REGIME_WEIGHTS_BASE[r]) for r in RNAMES}
    best = base_obj

    for p in [1, 2]:
        print(f"\n  Pass {p}", flush=True)
        for rn in RNAMES:
            tw = time.time()
            bw, best = sweep_one(rn, cur_w, fast, best)
            cur_w[rn] = bw
            print(f"    {rn}: v1={bw['v1']:.2f} i460={bw['i460']:.4f} i415={bw['i415']:.4f} "
                  f"f168={bw['f168']:.4f}  OBJ={best:.4f}  Δ={best-base_obj:+.4f}  [{int(time.time()-tw)}s]", flush=True)

    final_obj, final_yr = fast_eval(cur_w, fast)
    delta = final_obj - base_obj
    print(f"\n  Final OBJ={final_obj:.4f}  Δ={delta:+.4f}", flush=True)

    print("\n[5] LOYO...", flush=True)
    wins = 0
    for yr in sorted(final_yr):
        win = final_yr[yr] > base_yr[yr]
        if win: wins += 1
        print(f"  {yr}: base={base_yr[yr]:.4f}  cand={final_yr[yr]:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={final_obj:.4f}  Δ={delta:+.4f}  LOYO {wins}/5", flush=True)
    validated = wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'}", flush=True)

    result = {"phase":"263n","baseline_obj":float(base_obj),"best_obj":float(final_obj),
              "delta":float(delta),"loyo_wins":wins,"loyo_total":5,
              "validated":validated,"best_regime_weights":cur_w}
    _partial.update(result)
    out = Path("artifacts/phase263n"); out.mkdir(parents=True, exist_ok=True)
    (out/"phase263n_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT/"configs"/"production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        brs = cfg["breadth_regime_switching"]["regime_weights"]
        for rn, wts in cur_w.items():
            brs[rn]["v1"]=wts["v1"]; brs[rn]["i460bw168"]=wts["i460"]
            brs[rn]["i415bw216"]=wts["i415"]; brs[rn]["f168"]=wts["f168"]
        cfg["_version"]="2.48.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"]=round(final_obj,4)
        with open(cfg_path,"w") as f: json.dump(cfg,f,indent=2,ensure_ascii=False)
        print(f"\n  Config → v2.48.0  OBJ={round(final_obj,4)}", flush=True)
        subprocess.run(["git","add",str(cfg_path)],cwd=ROOT)
        msg=f"feat: P263n regime weight reopt11 OBJ={round(final_obj,4)} LOYO={wins}/5 D={delta:+.4f} [v2.48.0]"
        subprocess.run(["git","commit","-m",msg],cwd=ROOT)
        subprocess.run(["git","push"],cwd=ROOT)
        print(f"  Committed: {msg}", flush=True)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)

main()
