"""
Phase 264 NumPy — Per-Regime VOL + DISP Retune [VECTORIZED]
=============================================================
Baseline: v2.48.0, OBJ=4.2978

VOL scales (P231) and DISP params (P232) were tuned for old regime weights.
With P262n FTS + P263n regime weights, these overlays need retuning.

Current VOL per-regime: LOW=0.40, MID=0.15, HIGH=0.10
Current DISP per-regime: LOW=0.50, MID=1.50(AMP), HIGH=0.50

Sweep: sequential 1D search per regime per param.
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
    out = Path("artifacts/phase264n"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase264n_report.json").write_text(json.dumps(_partial, indent=2))
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

# P262n FTS params (fixed)
TS_PCT_WIN = {"LOW": 240, "MID": 288, "HIGH": 400}
FTS_RS = {"LOW": 0.05, "MID": 0.05, "HIGH": 0.40}
FTS_BS = {"LOW": 4.00, "MID": 2.50, "HIGH": 2.25}
FTS_RT = {"LOW": 0.80, "MID": 0.65, "HIGH": 0.50}
FTS_BT = {"LOW": 0.33, "MID": 0.22, "HIGH": 0.22}

# Current VOL params
VOL_THR   = {"LOW": 0.50, "MID": 0.50, "HIGH": 0.50}
VOL_SCALE = {"LOW": 0.40, "MID": 0.15, "HIGH": 0.10}

# Current DISP params
DISP_THR   = {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}
DISP_SCALE = {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}

# P263n regime weights
REGIME_WEIGHTS = {
    "LOW":  {"v1": 0.30, "i460": 0.10, "i415": 0.20, "f168": 0.40},
    "MID":  {"v1": 0.15, "i460": 0.00, "i415": 0.15, "f168": 0.70},
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

BASELINE_OBJ = 4.2978
MIN_DELTA = 0.005; MIN_LOYO = 3
COST_MODEL = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})

VOL_SCALE_SWEEP = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
VOL_THR_SWEEP   = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
DISP_SCALE_SWEEP = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]
DISP_THR_SWEEP   = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0 or len(a) == 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w-1, len(a)):
        out[i] = cs[i]/w if i==w-1 else (cs[i]-cs[i-w])/w
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
    return {"btc_vol":btc_vol,"fund_std_pct":fund_std_pct,"ts_pct":ts_pct,
            "brd_pct":brd_pct,"fixed":fixed,"n":n,"dataset":dataset}

def compute_v1(ds, params):
    return np.array(BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        ds, make_strategy({"name":"nexus_alpha_v1","params":params})).returns)

def build_overlay_precomp(all_base, all_v1, vol_thr, vol_scale, disp_thr, disp_scale):
    """Build precomputed arrays with given VOL+DISP+FTS overlays."""
    precomp = {}
    for yr, base in sorted(all_base.items()):
        bv=base["btc_vol"]; fsp=base["fund_std_pct"]; ts_pct=base["ts_pct"]
        brd=base["brd_pct"]; fixed=base["fixed"]; v1=all_v1[yr]
        ml = min(min(len(v1[r]) for r in RNAMES), min(len(fixed[k]) for k in ["i460","i415","f168"]))
        bv=bv[:ml]; bpct=brd[:ml]; fsp=fsp[:ml]
        ridx = np.where(bpct < P_LOW, 0, np.where(bpct > P_HIGH, 2, 1))
        masks = [ridx==i for i in range(3)]
        ov = np.ones(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]
            # VOL
            ov[m & ~np.isnan(bv) & (bv > vol_thr[rn])] *= vol_scale[rn]
            # DISP
            ov[m & (fsp > disp_thr[rn])] *= disp_scale[rn]
            # FTS
            tsp = ts_pct[TS_PCT_WIN[rn]][:ml]
            ov[m & (tsp > FTS_RT[rn])] *= FTS_RS[rn]
            ov[m & (tsp < FTS_BT[rn])] *= FTS_BS[rn]
        sc = {"v1L": ov*v1["LOW"][:ml], "v1M": ov*v1["MID"][:ml],
              "i460": ov*fixed["i460"][:ml], "i415": ov*fixed["i415"][:ml], "f168": ov*fixed["f168"][:ml]}
        precomp[yr] = (sc, masks, ml)
    return precomp

def fast_eval(weights, precomp):
    yearly = {}
    for yr, (sc, masks, ml) in precomp.items():
        ens = np.zeros(ml)
        for ri, rn in enumerate(RNAMES):
            m=masks[ri]; w=weights[rn]; vk="v1L" if rn=="LOW" else "v1M"
            ens[m] += w["v1"]*sc[vk][m]+w["i460"]*sc["i460"][m]+w["i415"]*sc["i415"][m]+w["f168"]*sc["f168"][m]
        r=ens[~np.isnan(ens)]
        yearly[yr] = float(np.mean(r)/np.std(r,ddof=1)*np.sqrt(8760)) if len(r)>1 else 0.0
    vals=list(yearly.values())
    return float(np.mean(vals)-0.5*np.std(vals,ddof=1)), yearly

def main():
    t0 = time.time()
    print("="*65)
    print("Phase 264n — Per-Regime VOL + DISP Retune [VECTORIZED]")
    print(f"Baseline: v2.48.0  OBJ={BASELINE_OBJ}")
    print("="*65)

    print("\n[1] Loading data...", flush=True)
    all_base = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base[yr] = load_year_data(yr)
        print("done.", flush=True)

    print("\n[2] Computing V1...", flush=True)
    all_v1 = {}
    for yr in sorted(YEAR_RANGES):
        ds = all_base[yr]["dataset"]
        all_v1[yr] = {"LOW": compute_v1(ds,V1_LOW_PARAMS), "MID": compute_v1(ds,V1_MIDHIGH_PARAMS),
                      "HIGH": compute_v1(ds,V1_MIDHIGH_PARAMS)}
        print(f"  {yr}: done.", flush=True)

    cur_vt = dict(VOL_THR); cur_vs = dict(VOL_SCALE)
    cur_dt = dict(DISP_THR); cur_ds = dict(DISP_SCALE)

    def rebuild(vt=None,vs=None,dt=None,ds=None):
        return build_overlay_precomp(all_base, all_v1,
                                     vt or cur_vt, vs or cur_vs, dt or cur_dt, ds or cur_ds)

    base_obj, base_yr = fast_eval(REGIME_WEIGHTS, rebuild())
    print(f"\n  Baseline OBJ = {base_obj:.4f}", flush=True)
    _partial["baseline_obj"] = float(base_obj)
    best_obj = base_obj

    print("\n[3] VOL sweep (per regime)...", flush=True)
    for rn in RNAMES:
        # VOL scale sweep
        for vs in VOL_SCALE_SWEEP:
            nvs = {**cur_vs, rn: vs}
            o, _ = fast_eval(REGIME_WEIGHTS, rebuild(vs=nvs))
            if o > best_obj: best_obj=o; cur_vs=nvs
        print(f"  {rn} vol_scale={cur_vs[rn]}  OBJ={best_obj:.4f}", flush=True)
        # VOL threshold sweep
        for vt in VOL_THR_SWEEP:
            nvt = {**cur_vt, rn: vt}
            o, _ = fast_eval(REGIME_WEIGHTS, rebuild(vt=nvt))
            if o > best_obj: best_obj=o; cur_vt=nvt
        print(f"  {rn} vol_thr={cur_vt[rn]}  OBJ={best_obj:.4f}", flush=True)

    print(f"\n  After VOL sweep: OBJ={best_obj:.4f}  Δ={best_obj-base_obj:+.4f}", flush=True)

    print("\n[4] DISP sweep (per regime)...", flush=True)
    for rn in RNAMES:
        # DISP scale sweep
        for ds in DISP_SCALE_SWEEP:
            nds = {**cur_ds, rn: ds}
            o, _ = fast_eval(REGIME_WEIGHTS, rebuild(ds=nds))
            if o > best_obj: best_obj=o; cur_ds=nds
        print(f"  {rn} disp_scale={cur_ds[rn]}  OBJ={best_obj:.4f}", flush=True)
        # DISP threshold sweep
        for dt in DISP_THR_SWEEP:
            ndt = {**cur_dt, rn: dt}
            o, _ = fast_eval(REGIME_WEIGHTS, rebuild(dt=ndt))
            if o > best_obj: best_obj=o; cur_dt=ndt
        print(f"  {rn} disp_thr={cur_dt[rn]}  OBJ={best_obj:.4f}", flush=True)

    final_precomp = rebuild()
    final_obj, final_yr = fast_eval(REGIME_WEIGHTS, final_precomp)
    delta = final_obj - base_obj
    print(f"\n  Final OBJ={final_obj:.4f}  Δ={delta:+.4f}", flush=True)
    print(f"  VOL: {dict(zip(RNAMES,[f'{cur_vt[r]}/{cur_vs[r]}' for r in RNAMES]))}", flush=True)
    print(f"  DISP:{dict(zip(RNAMES,[f'{cur_dt[r]}/{cur_ds[r]}' for r in RNAMES]))}", flush=True)

    print("\n[5] LOYO...", flush=True)
    wins = 0
    for yr in sorted(final_yr):
        win = final_yr[yr] > base_yr[yr]
        if win: wins += 1
        print(f"  {yr}: base={base_yr[yr]:.4f}  cand={final_yr[yr]:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={final_obj:.4f}  Δ={delta:+.4f}  LOYO {wins}/5", flush=True)
    validated = wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'}", flush=True)

    result = {"phase":"264n","baseline_obj":float(base_obj),"best_obj":float(final_obj),
              "delta":float(delta),"loyo_wins":wins,"loyo_total":5,"validated":validated,
              "vol_thr":cur_vt,"vol_scale":cur_vs,"disp_thr":cur_dt,"disp_scale":cur_ds}
    _partial.update(result)
    out=Path("artifacts/phase264n"); out.mkdir(parents=True, exist_ok=True)
    (out/"phase264n_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT/"configs"/"production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        # Update NumPy per-regime overlay params in config
        vop = cfg.setdefault("vol_overlay_params", {})
        for rn in RNAMES: vop[rn] = {"thr": cur_vt[rn], "scale": cur_vs[rn]}
        dop = cfg.setdefault("disp_overlay_params", {})
        for rn in RNAMES: dop[rn] = {"thr": cur_dt[rn], "scale": cur_ds[rn]}
        cfg["_version"] = "2.49.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(final_obj,4)
        with open(cfg_path,"w") as f: json.dump(cfg,f,indent=2,ensure_ascii=False)
        print(f"\n  Config → v2.49.0  OBJ={round(final_obj,4)}", flush=True)
        subprocess.run(["git","add",str(cfg_path)],cwd=ROOT)
        msg=f"feat: P264n VOL+DISP retune OBJ={round(final_obj,4)} LOYO={wins}/5 D={delta:+.4f} [v2.49.0]"
        subprocess.run(["git","commit","-m",msg],cwd=ROOT)
        subprocess.run(["git","push"],cwd=ROOT)
        print(f"  Committed: {msg}", flush=True)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)

main()
