#!/usr/bin/env python3
"""
Phase 162: Idio Momentum Lookback Fine-Tune on v2.8.0 Stack
============================================================
Baseline: v2.8.0 OBJ=2.2095
  Breadth: lb=192, pct_window=336, p_low=0.35, p_high=0.65
  Vol: scale=0.4, f168_boost=0.15, threshold=0.5

Context: I415bw216 and I460bw168 were found optimal in P84 on p91b stack.
With overlays (breadth regime, vol, funding overlays), the optimal idio lookbacks
may have shifted — the overlays alter WHEN signals are applied, potentially
changing which lookbacks capture the best remaining alpha.

PART A — I415bw216 lookback sweep:
  Test: lb = [390, 400, 410, 415, 420, 430, 445]
  Keep: bw=216, k=4, tgl=0.30, rebal=48

PART B — I460bw168 lookback sweep:
  Test: lb = [430, 445, 460, 475, 490, 510]
  Keep: bw=168, k=4, tgl=0.30, rebal=48

Full v2.8.0 production stack for all variants.
Pass: OBJ > 2.2095 AND LOYO >= 3/5
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
_signal.alarm(3000)

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]

# v2.8.0 overlay params
VOL_WINDOW    = 168;  VOL_THRESHOLD = 0.5
VOL_SCALE     = 0.4;  VOL_F168_BOOST = 0.15
BRD_LOOKBACK  = 192;  PCT_WINDOW = 336
P_LOW, P_HIGH = 0.35, 0.65

FDO_PCT   = PROD_CFG.get("funding_dispersion_overlay", {}).get("boost_threshold_pct", 0.75)
FDO_SCALE = PROD_CFG.get("funding_dispersion_overlay", {}).get("boost_scale", 1.15)

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
REGIME_WEIGHTS = PROD_CFG.get("breadth_regime_switching", {}).get("regime_weights", {})
if REGIME_WEIGHTS:
    for regime, rw in REGIME_WEIGHTS.items():
        key = {"LOW": "prod", "MID": "mid", "HIGH": "p143b"}.get(regime.upper(), regime.lower())
        WEIGHTS[key] = {k: v for k, v in rw.items() if not k.startswith("_")}

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())
BASELINE_OBJ = 2.2095

I415_LBS = [390, 400, 410, 415, 420, 430, 445]
I460_LBS  = [430, 445, 460, 475, 490, 510]

OUT_DIR = ROOT / "artifacts" / "phase162"
OUT_DIR.mkdir(parents=True, exist_ok=True)
COST_MODEL = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})


def sharpe(rets):
    if len(rets) < 100: return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0

def obj_func(ys):
    arr = np.array(ys)
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)

def _save(data, partial=False):
    data["partial"] = partial
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase162_report.json").write_text(json.dumps(data, indent=2, default=str))
    print(f"✅ Saved")

def rolling_pct(sig, window):
    n = len(sig); pct = np.full(n, 0.5)
    for i in range(window, n):
        pct[i] = float(np.mean(sig[i-window:i] <= sig[i]))
    if window < n: pct[:window] = pct[window]
    return pct

def btc_vol(ds):
    n = len(ds.timeline); rets = np.zeros(n)
    for i in range(1, n):
        c0 = ds.close("BTCUSDT", i-1); c1 = ds.close("BTCUSDT", i)
        rets[i] = (c1/c0-1.0) if c0 > 0 else 0.0
    vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        vol[i] = float(np.std(rets[i-VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: vol[:VOL_WINDOW] = vol[VOL_WINDOW]
    return vol

def breadth(ds):
    n = len(ds.timeline); brd = np.full(n, 0.5)
    for i in range(BRD_LOOKBACK, n):
        pos = sum(1 for s in SYMBOLS if (c0:=ds.close(s,i-BRD_LOOKBACK)) > 0 and ds.close(s,i) > c0)
        brd[i] = pos / len(SYMBOLS)
    if BRD_LOOKBACK < n: brd[:BRD_LOOKBACK] = brd[BRD_LOOKBACK]
    return brd

def fund_disp(ds):
    n = len(ds.timeline); fd = np.zeros(n)
    for i in range(n):
        ts = ds.timeline[i]; rates = []
        for s in SYMBOLS:
            try:
                r = ds.last_funding_rate_before(s, ts)
                if r is not None and not np.isnan(float(r)): rates.append(float(r))
            except: pass
        fd[i] = float(np.std(rates)) if len(rates) > 1 else 0.0
    return fd

def fund_ts(ds):
    n = len(ds.timeline); lv = np.zeros(n)
    for i in range(n):
        ts = ds.timeline[i]; rates = []
        for s in SYMBOLS:
            try:
                r = ds.funding_rate_at(s, ts)
                if r is not None and not np.isnan(float(r)): rates.append(float(r))
            except: pass
        lv[i] = float(np.mean(rates)) if rates else 0.0
    sp = np.zeros(n)
    for i in range(FTS_LONG_W, n):
        sp[i] = np.mean(lv[max(0,i-FTS_SHORT_W):i]) - np.mean(lv[i-FTS_LONG_W:i])
    if FTS_LONG_W < n: sp[:FTS_LONG_W] = sp[FTS_LONG_W]
    return sp

def _wkey(sk):
    """Map signal key to weight dict key (f144 signal may be keyed as f168 in regime weights)."""
    if sk == "f144":
        for regime_w in WEIGHTS.values():
            if "f168" in regime_w:
                return "f168"
    return sk

def blend(sig_rets, bv, brd_p, fdo_p, fts_p):
    sks = list(sig_rets.keys())
    ml = min(len(sig_rets[sk]) for sk in sks)
    bv_=bv[:ml]; brd_=brd_p[:ml]; fdo_=fdo_p[:ml]; fts_=fts_p[:ml]
    ens = np.zeros(ml); n_oth = len(sks)-1
    for i in range(ml):
        if brd_[i] >= P_HIGH: w = WEIGHTS["p143b"]
        elif brd_[i] >= P_LOW: w = WEIGHTS["mid"]
        else: w = WEIGHTS["prod"]
        if not np.isnan(bv_[i]) and bv_[i] > VOL_THRESHOLD:
            bo = VOL_F168_BOOST / max(1, n_oth); ri = 0.0
            for sk in sks:
                wk = _wkey(sk)
                aw = min(0.60, w[wk]+VOL_F168_BOOST) if sk=="f144" else max(0.05, w[wk]-bo)
                ri += aw * sig_rets[sk][i]
            ri *= VOL_SCALE
        else:
            ri = sum(w[_wkey(sk)]*sig_rets[sk][i] for sk in sks)
        if fdo_[i] > FDO_PCT: ri *= FDO_SCALE
        sp = fts_[i]
        if sp >= FTS_RT: ri *= FTS_RS
        elif sp <= FTS_BT: ri *= FTS_BS
        ens[i] = ri
    return ens


def main():
    global _partial
    print("="*72)
    print("PHASE 162: Idio Lookback Fine-Tune on v2.8.0 Stack")
    print(f"  Part A: I415bw216 lb = {I415_LBS}")
    print(f"  Part B: I460bw168 lb = {I460_LBS}")
    print("="*72)

    print("\n[1/4] Loading data + base signals...")
    datasets = {}; bv_d={};brd_d={};fdo_d={};fts_d={}
    # Fixed signals: v1, f144 (production params)
    fixed_rets = {"v1": {}, "f144": {}}

    for yr in YEARS:
        s, e = YEAR_RANGES[yr]
        print(f"  {yr}: ", end="", flush=True)
        cfg_d = {"provider":"binance_rest_v1","symbols":SYMBOLS,"start":s,"end":e,"bar_interval":"1h","cache_dir":".cache/binance_rest"}
        ds = make_provider(cfg_d, seed=42).load()
        datasets[yr]=ds; bv_d[yr]=btc_vol(ds); brd_d[yr]=breadth(ds)
        fdo_d[yr]=fund_disp(ds); fts_d[yr]=fund_ts(ds)
        for sk in ["v1", "f144"]:
            sd = SIGNAL_DEFS[sk]
            bt = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": sd["strategy"], "params": dict(sd["params"])})
            fixed_rets[sk][yr] = np.array(BacktestEngine(bt).run(ds, strat).returns, dtype=np.float64)
            print(".", end="", flush=True)
        print("✓")

    results = {}

    def eval_variant(label, i415_lb, i460_lb):
        yr_sharpes = {}
        for yr in YEARS:
            brd_p = rolling_pct(brd_d[yr], PCT_WINDOW)
            fdo_p = rolling_pct(fdo_d[yr], PCT_WINDOW)
            fts_p = rolling_pct(fts_d[yr], FTS_PCT_WIN)
            sr = dict(fixed_rets); sr.update({"i415bw216": {}, "i460bw168": {}})
            for sig, lb, bw in [("i415bw216", i415_lb, 216), ("i460bw168", i460_lb, 168)]:
                params = {"k_per_side":4,"lookback_bars":lb,"beta_window_bars":bw,
                          "target_gross_leverage":0.3,"rebalance_interval_bars":48}
                bt = BacktestConfig(costs=COST_MODEL)
                strat = make_strategy({"name":"idio_momentum_alpha","params":params})
                sr[sig] = np.array(BacktestEngine(bt).run(datasets[yr], strat).returns, dtype=np.float64)
            ens = blend({k: sr[k][yr] if isinstance(sr[k], dict) else sr[k] for k in ["v1","i460bw168","i415bw216","f144"]},
                        bv_d[yr], brd_p, fdo_p, fts_p)
            yr_sharpes[yr] = sharpe(ens)
        obj = obj_func(list(yr_sharpes.values()))
        delta = round(obj - BASELINE_OBJ, 4)
        flag = " ✅" if obj > BASELINE_OBJ else ""
        print(f"  {label:18s} OBJ={obj:.4f} Δ={delta:+.4f}{flag}")
        return {"yearly": yr_sharpes, "obj": obj, "delta": delta, "i415_lb": i415_lb, "i460_lb": i460_lb}

    # Part A: sweep I415 lb, keep I460=460
    print("\n[2/4] PART A — I415bw216 lookback sweep (I460=460 fixed)...")
    for lb in I415_LBS:
        label = f"I415lb{lb}_I460lb460"
        results[label] = eval_variant(label, lb, 460)
        _partial.update({"results": results}); _save(_partial, partial=True)

    best_a = max([(l, d) for l, d in results.items()], key=lambda x: x[1]["obj"])
    print(f"  Part A best: {best_a[0]} OBJ={best_a[1]['obj']:.4f}")

    # Part B: sweep I460 lb, keep best I415
    best_i415 = best_a[1]["i415_lb"]
    print(f"\n[3/4] PART B — I460bw168 lookback sweep (I415={best_i415})...")
    for lb in I460_LBS:
        label = f"I415lb{best_i415}_I460lb{lb}"
        if label not in results:
            results[label] = eval_variant(label, best_i415, lb)
            _partial.update({"results": results}); _save(_partial, partial=True)

    # Overall best
    best_all = max(results.items(), key=lambda x: x[1]["obj"])
    best_label, best_data = best_all
    best_obj = best_data["obj"]

    # LOYO for best if improves
    loyo_wins = 0; loyo_detail = {}
    if best_obj > BASELINE_OBJ:
        print(f"\n[4/4] LOYO validation for {best_label} (OBJ={best_obj:.4f})...")
        bi415, bi460 = best_data["i415_lb"], best_data["i460_lb"]
        for held in YEARS:
            train_yrs = [y for y in YEARS if y != held]
            tr_ens_base=[]; tr_ens_best=[]
            for yr in train_yrs:
                brd_p=rolling_pct(brd_d[yr],PCT_WINDOW); fdo_p=rolling_pct(fdo_d[yr],PCT_WINDOW); fts_p=rolling_pct(fts_d[yr],FTS_PCT_WIN)
                # Baseline (I415=415, I460=460)
                sr_b = dict(fixed_rets)
                for sig, lb, bw in [("i415bw216",415,216),("i460bw168",460,168)]:
                    p={"k_per_side":4,"lookback_bars":lb,"beta_window_bars":bw,"target_gross_leverage":0.3,"rebalance_interval_bars":48}
                    sr_b[sig] = np.array(BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(datasets[yr],make_strategy({"name":"idio_momentum_alpha","params":p})).returns,dtype=np.float64)
                e_b = blend({k: sr_b[k][yr] if isinstance(sr_b[k],dict) else sr_b[k] for k in ["v1","i460bw168","i415bw216","f144"]},bv_d[yr],brd_p,fdo_p,fts_p)
                tr_ens_base.extend(e_b.tolist())
                # Best variant
                sr_x = dict(fixed_rets)
                for sig, lb, bw in [("i415bw216",bi415,216),("i460bw168",bi460,168)]:
                    p={"k_per_side":4,"lookback_bars":lb,"beta_window_bars":bw,"target_gross_leverage":0.3,"rebalance_interval_bars":48}
                    sr_x[sig] = np.array(BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(datasets[yr],make_strategy({"name":"idio_momentum_alpha","params":p})).returns,dtype=np.float64)
                e_x = blend({k: sr_x[k][yr] if isinstance(sr_x[k],dict) else sr_x[k] for k in ["v1","i460bw168","i415bw216","f144"]},bv_d[yr],brd_p,fdo_p,fts_p)
                tr_ens_best.extend(e_x.tolist())
            s_b=sharpe(np.array(tr_ens_base)); s_x=sharpe(np.array(tr_ens_best))
            d=round(s_x-s_b,4); win=d>0; loyo_wins+=int(win)
            loyo_detail[f"loo_{held}"]={"base":s_b,"best":s_x,"delta":d,"win":win}
            print(f"    LOO-{held}: base={s_b:.4f} best={s_x:.4f} Δ={d:+.4f} {'✅' if win else '❌'}")
    else:
        print("\n[4/4] Best variant does not exceed baseline — skipping LOYO.")

    print("\n" + "="*72)
    validated = best_obj > BASELINE_OBJ and loyo_wins >= 3
    if validated:
        verdict = f"VALIDATED — {best_label} OBJ={best_obj:.4f} (+{best_obj-BASELINE_OBJ:.4f}), LOYO {loyo_wins}/5"
        next_phase = f"Phase 163: Update prod config I415lb={best_data['i415_lb']}, I460lb={best_data['i460_lb']}. WF validation."
    elif best_obj > BASELINE_OBJ:
        verdict = f"MARGINAL — {best_label} Δ=+{best_obj-BASELINE_OBJ:.4f} but LOYO {loyo_wins}/5 insufficient"
        next_phase = "Phase 163: Test I415/I460 blended lookbacks (e.g., ensemble of two lb variants)."
    else:
        verdict = f"NO IMPROVEMENT — best {best_label} OBJ={best_obj:.4f} vs baseline {BASELINE_OBJ}. Lookbacks are optimal."
        next_phase = "Phase 163: Accept v2.8.0 as R&D ceiling. Deploy focus: paper trading + live monitoring."
    print(f"VERDICT: {verdict}")
    print(f"NEXT: {next_phase}")
    print("="*72)

    report = {"phase":162,"elapsed_seconds":round(time.time()-_start,1),"baseline_obj":BASELINE_OBJ,
              "results": {l: d for l,d in results.items()},
              "best_label":best_label,"best_obj":best_obj,"loyo_wins":loyo_wins,"loyo_detail":loyo_detail,
              "validated":validated,"verdict":verdict,"next_phase_notes":next_phase}
    _save(report, partial=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}"); traceback.print_exc()
        _partial["error"]=str(e); _partial["traceback"]=traceback.format_exc()
        _save(_partial, partial=True); sys.exit(1)
