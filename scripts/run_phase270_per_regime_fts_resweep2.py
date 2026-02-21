"""
Phase 270 — Per-Regime FTS RS/RT/BS/BT Re-Sweep [VECTORIZED, P262n-style]
==========================================================================
Baseline: v2.48.0, OBJ=4.2978

Rationale:
  Phase 269 had an evaluation bug (inverted VOL direction, wrong overlay split).
  This phase uses P262n's proven approach: rebuild full overlay (VOL+DISP+FTS)
  for each FTS param combo tested. Sweep RS→RT→BS→BT per regime, sequentially.

  P262n (LOYO=5/5, Δ=+0.2537) found from baseline v2.46.0:
    LOW:  rs=0.05 rt=0.80 bs=4.00 bt=0.33
    MID:  rs=0.05 rt=0.65 bs=2.50 bt=0.22
    HIGH: rs=0.40 rt=0.50 bs=2.25 bt=0.22
  vs current per_regime_* keys:
    LOW:  rs=0.05 rt=0.80 bs=4.00 bt=0.30
    MID:  rs=0.05 rt=0.65 bs=2.00 bt=0.40
    HIGH: rs=0.25 rt=0.50 bs=2.00 bt=0.20

  Key differences: HIGH rs 0.25→0.40, MID bs 2.0→2.5, MID bt 0.40→0.22
  All params loaded dynamically from config. P_HIGH=0.68.
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
    _partial["timeout"] = True
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(7200)

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
           "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]

YEAR_RANGES = {
    2021: ("2021-02-01", "2022-01-01"),
    2022: ("2022-01-01", "2023-01-01"),
    2023: ("2023-01-01", "2024-01-01"),
    2024: ("2024-01-01", "2025-01-01"),
    2025: ("2025-01-01", "2026-01-01"),
}

CFG_PATH = ROOT / "configs" / "production_p91b_champion.json"

def load_config_params():
    cfg = json.load(open(CFG_PATH))
    brs = cfg["breadth_regime_switching"]
    fts = cfg.get("fts_overlay_params", {})
    vol = cfg.get("vol_overlay_params", {})
    disp = cfg.get("disp_overlay_params", {})
    v1w  = cfg.get("v1_per_regime_weights", {})
    return {
        "p_low":  brs.get("p_low", 0.30),
        "p_high": brs.get("p_high", 0.68),
        "regime_weights": {
            "LOW":  {"v1": brs["regime_weights"]["LOW"]["v1"],
                     "i460": brs["regime_weights"]["LOW"]["i460bw168"],
                     "i415": brs["regime_weights"]["LOW"]["i415bw216"],
                     "f168": brs["regime_weights"]["LOW"]["f168"]},
            "MID":  {"v1": brs["regime_weights"]["MID"]["v1"],
                     "i460": brs["regime_weights"]["MID"]["i460bw168"],
                     "i415": brs["regime_weights"]["MID"]["i415bw216"],
                     "f168": brs["regime_weights"]["MID"]["f168"]},
            "HIGH": {"v1": brs["regime_weights"]["HIGH"]["v1"],
                     "i460": brs["regime_weights"]["HIGH"]["i460bw168"],
                     "i415": brs["regime_weights"]["HIGH"]["i415bw216"],
                     "f168": brs["regime_weights"]["HIGH"]["f168"]},
        },
        "fts_rs": fts.get("per_regime_rs", {"LOW": 0.05, "MID": 0.05, "HIGH": 0.25}),
        "fts_bs": fts.get("per_regime_bs", {"LOW": 4.0,  "MID": 2.0,  "HIGH": 2.0}),
        "fts_rt": fts.get("per_regime_rt", {"LOW": 0.80, "MID": 0.65, "HIGH": 0.50}),
        "fts_bt": fts.get("per_regime_bt", {"LOW": 0.30, "MID": 0.40, "HIGH": 0.20}),
        "ts_pct_win": fts.get("per_regime_ts_pct_win", {"LOW": 240, "MID": 288, "HIGH": 400}),
        "vol_thr":   vol.get("per_regime_threshold", {"LOW": 0.50, "MID": 0.50, "HIGH": 0.55}),
        "vol_scale": vol.get("per_regime_scale",     {"LOW": 0.40, "MID": 0.15, "HIGH": 0.05}),
        "disp_thr":  disp.get("per_regime_threshold", {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}),
        "disp_scale":disp.get("per_regime_scale",     {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}),
        "v1_weights": {r: v1w.get(r, {"w_carry": 0.25, "w_mom": 0.45, "w_mean_reversion": 0.30})
                       for r in ["LOW", "MID", "HIGH"]},
        "version": cfg["_version"],
        "baseline_obj": cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"],
    }

BRD_LB = 192; PCT_WINDOW = 336; FUND_DISP_PCT = 240
TS_SHORT = 16; TS_LONG = 72; VOL_WINDOW = 168
RNAMES = ["LOW", "MID", "HIGH"]

I460_PARAMS = {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}

MIN_DELTA = 0.005; MIN_LOYO = 3

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

# Sweep grids (centered on P262n's findings + current params)
RS_GRID = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
RT_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
BS_GRID = [1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.50, 4.00, 5.00]
BT_GRID = [0.10, 0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35, 0.40]

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0 or len(a) == 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year_data(year, P):
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h",
           "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
    n = len(dataset.timeline)
    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n): close_mat[i, j] = dataset.close(sym, i)
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1/c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i-VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try: fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i-FUND_DISP_PCT:i] <= fund_std_raw[i]))
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    # Pre-compute ts percentile for all needed windows
    unique_wins = sorted(set(P["ts_pct_win"].values()))
    ts_pct = {}
    for w in unique_wins:
        arr = np.full(n, 0.5)
        for i in range(w, n): arr[i] = float(np.mean(ts_raw[i-w:i] <= ts_raw[i]))
        ts_pct[w] = arr
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))
    fixed_rets = {}
    for sk, sname, params in [
        ("i460", "idio_momentum_alpha", I460_PARAMS),
        ("i415", "idio_momentum_alpha", I415_PARAMS),
        ("f168", "funding_momentum_alpha", F168_PARAMS),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        fixed_rets[sk] = np.array(result.returns)
    return btc_vol, fund_std_pct, ts_pct, brd_pct, fixed_rets, n, dataset

def compute_v1(dataset, v1w_params):
    vp = {"k_per_side": 2,
          "w_carry": v1w_params["w_carry"],
          "w_mom": v1w_params["w_mom"],
          "w_mean_reversion": v1w_params["w_mean_reversion"],
          "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
          "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60}
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": vp}))
    return np.array(result.returns)

def build_precomp(P, all_base_data, all_v1_raw, fts_rs, fts_rt, fts_bs, fts_bt):
    """Build full overlay precomp (VOL+DISP+FTS all included) for given FTS params.

    Follows P262n's proven approach: rebuild for each FTS combo.
    VOL condition: bv > thr (dampen during HIGH volatility). ← P262n-verified direction.
    """
    precomp = {}
    for yr, base_data in sorted(all_base_data.items()):
        btc_vol, fund_std_pct, ts_pct, brd_pct, fixed_rets, n, _ = base_data
        v1 = all_v1_raw[yr]
        ml = min(
            min(len(v1[r]) for r in RNAMES),
            min(len(fixed_rets[k]) for k in ["i460","i415","f168"])
        )
        bv = btc_vol[:ml]
        bpct = brd_pct[:ml]
        fsp = fund_std_pct[:ml]

        # Regime index
        ridx = np.where(bpct < P["p_low"], 0, np.where(bpct > P["p_high"], 2, 1))
        masks = [ridx == i for i in range(3)]

        # Full overlay (VOL + DISP + FTS)
        ov = np.ones(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]
            # VOL: dampen when vol > threshold (high-vol dampening) — P262n direction
            ov[m & ~np.isnan(bv) & (bv > P["vol_thr"][rn])] *= P["vol_scale"][rn]
            # DISP: scale when dispersion > threshold
            ov[m & (fsp > P["disp_thr"][rn])] *= P["disp_scale"][rn]
            # FTS
            tsp = ts_pct[P["ts_pct_win"][rn]][:ml]
            ov[m & (tsp > fts_rt[rn])] *= fts_rs[rn]
            ov[m & (tsp < fts_bt[rn])] *= fts_bs[rn]

        # Pre-scaled signal arrays
        sc = {
            "v1L": ov * v1["LOW"][:ml],
            "v1M": ov * v1["MID"][:ml],
            "i460": ov * fixed_rets["i460"][:ml],
            "i415": ov * fixed_rets["i415"][:ml],
            "f168": ov * fixed_rets["f168"][:ml],
        }
        precomp[yr] = (sc, masks, ml)
    return precomp

def fast_eval(P, precomp):
    """Evaluate OBJ from precomputed data by applying regime weights."""
    yearly_sharpes = []
    for yr, (sc, masks, ml) in sorted(precomp.items()):
        ens = np.zeros(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]
            w = P["regime_weights"][rn]
            vk = "v1L" if rn == "LOW" else "v1M"
            ens[m] += (w["v1"]*sc[vk][m] + w["i460"]*sc["i460"][m]
                       + w["i415"]*sc["i415"][m] + w["f168"]*sc["f168"][m])
        r = ens[~np.isnan(ens)]
        ann = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly_sharpes.append(ann)
    vals = yearly_sharpes
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1))

def sweep_regime(P, rn, cur_rs, cur_bs, cur_rt, cur_bt, all_base_data, all_v1_raw, best_obj):
    """Sweep FTS params for one regime. Sequential 1D searches."""
    rs = {**cur_rs}; rt = {**cur_rt}; bs = {**cur_bs}; bt = {**cur_bt}
    best_rs = rs[rn]; best_rt = rt[rn]; best_bs = bs[rn]; best_bt = bt[rn]

    # Sweep RS
    for v in RS_GRID:
        rs[rn] = v
        obj = fast_eval(P, build_precomp(P, all_base_data, all_v1_raw, rs, rt, bs, bt))
        if obj > best_obj + 1e-6: best_obj = obj; best_rs = v
    rs[rn] = best_rs

    # Sweep RT
    for v in RT_GRID:
        rt[rn] = v
        obj = fast_eval(P, build_precomp(P, all_base_data, all_v1_raw, rs, rt, bs, bt))
        if obj > best_obj + 1e-6: best_obj = obj; best_rt = v
    rt[rn] = best_rt

    # Sweep BS
    for v in BS_GRID:
        bs[rn] = v
        obj = fast_eval(P, build_precomp(P, all_base_data, all_v1_raw, rs, rt, bs, bt))
        if obj > best_obj + 1e-6: best_obj = obj; best_bs = v
    bs[rn] = best_bs

    # Sweep BT
    for v in BT_GRID:
        bt[rn] = v
        obj = fast_eval(P, build_precomp(P, all_base_data, all_v1_raw, rs, rt, bs, bt))
        if obj > best_obj + 1e-6: best_obj = obj; best_bt = v
    bt[rn] = best_bt

    return rs, rt, bs, bt, best_obj

def loyo_obj(P, precomp, exclude_yr):
    """Compute OBJ on all years except exclude_yr using given precomp (FTS baked in)."""
    yearly = []
    for yr, (sc, masks, ml) in sorted(precomp.items()):
        if yr == exclude_yr: continue
        ens = np.zeros(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]; w = P["regime_weights"][rn]
            vk = "v1L" if rn == "LOW" else "v1M"
            ens[m] += (w["v1"]*sc[vk][m] + w["i460"]*sc["i460"][m]
                       + w["i415"]*sc["i415"][m] + w["f168"]*sc["f168"][m])
        r = ens[~np.isnan(ens)]
        ann = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly.append(ann)
    return float(np.mean(yearly) - 0.5 * np.std(yearly, ddof=1))

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 68)
    print("Phase 270 — Per-Regime FTS RS/RT/BS/BT Re-Sweep [VECTORIZED v2]")

    P = load_config_params()
    base_obj = P["baseline_obj"]
    print(f"Baseline: v{P['version']}  OBJ={base_obj:.4f}")
    print(f"Current: RS={P['fts_rs']}  RT={P['fts_rt']}")
    print(f"         BS={P['fts_bs']}  BT={P['fts_bt']}")
    print("=" * 68)

    # [1] Load per-year data
    print("\n[1] Loading per-year data...")
    all_base_data = {}
    for yr in sorted(YEAR_RANGES.keys()):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr, P)
        print("done.")

    # [2] Pre-compute V1 per regime per year
    print("\n[2] Pre-computing V1 per regime × year...")
    all_v1_raw = {}
    for yr, base_data in sorted(all_base_data.items()):
        _, _, _, _, _, _, dataset = base_data
        all_v1_raw[yr] = {}
        for rname in RNAMES:
            all_v1_raw[yr][rname] = compute_v1(dataset, P["v1_weights"][rname])
        print(f"  {yr}: done.")

    # [3] Baseline evaluation
    print("\n[3] Evaluating baseline...")
    cur_rs = {r: P["fts_rs"][r] for r in RNAMES}
    cur_rt = {r: P["fts_rt"][r] for r in RNAMES}
    cur_bs = {r: P["fts_bs"][r] for r in RNAMES}
    cur_bt = {r: P["fts_bt"][r] for r in RNAMES}
    baseline_precomp = build_precomp(P, all_base_data, all_v1_raw, cur_rs, cur_rt, cur_bs, cur_bt)
    obj_measured = fast_eval(P, baseline_precomp)
    print(f"  Measured baseline OBJ = {obj_measured:.4f}  (config says {base_obj:.4f})")

    best_obj = obj_measured

    # [4] Per-regime sweep (2 passes)
    print("\n[4] Per-regime FTS sweep (2 passes)...")
    for pass_num in range(1, 3):
        print(f"\n  === Pass {pass_num} ===")
        for rname in RNAMES:
            cur_rs, cur_rt, cur_bs, cur_bt, best_obj = sweep_regime(
                P, rname, cur_rs, cur_bs, cur_rt, cur_bt, all_base_data, all_v1_raw, best_obj
            )
            print(f"    {rname}: rs={cur_rs[rname]:.3f} rt={cur_rt[rname]:.2f}"
                  f" bs={cur_bs[rname]:.2f} bt={cur_bt[rname]:.2f}  OBJ={best_obj:.4f}")

    delta = best_obj - obj_measured
    print(f"\n  Final OBJ={best_obj:.4f}  Δ={delta:+.4f}")

    # [5] LOYO validation
    print("\n[5] LOYO validation...")
    best_precomp = build_precomp(P, all_base_data, all_v1_raw, cur_rs, cur_rt, cur_bs, cur_bt)
    loyo_wins = 0
    for yr in sorted(YEAR_RANGES.keys()):
        base_loyo = loyo_obj(P, baseline_precomp, yr)
        cand_loyo = loyo_obj(P, best_precomp, yr)
        win = cand_loyo > base_loyo
        loyo_wins += int(win)
        print(f"  {yr}: base={base_loyo:.4f}  cand={cand_loyo:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={best_obj:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/5")

    validated = delta >= MIN_DELTA and loyo_wins >= MIN_LOYO

    if validated:
        print(f"\n✅ VALIDATED")
        cfg = json.load(open(CFG_PATH))
        fts = cfg.setdefault("fts_overlay_params", {})
        fts["per_regime_rs"] = cur_rs
        fts["per_regime_rt"] = cur_rt
        fts["per_regime_bs"] = cur_bs
        fts["per_regime_bt"] = cur_bt
        old_ver = cfg["_version"]
        parts = old_ver.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = ".".join(parts)
        cfg["_version"] = new_ver
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(best_obj, 4)
        json.dump(cfg, open(CFG_PATH, "w"), indent=2)
        print(f"\n  Config → v{new_ver}  OBJ={best_obj:.4f}")
        msg = (f"feat: P270 per-regime FTS resweep v2 OBJ={best_obj:.4f} "
               f"LOYO={loyo_wins}/5 D={delta:+.4f} [v{new_ver}]")
        subprocess.run(["git", "add", str(CFG_PATH)], cwd=ROOT)
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
    else:
        print(f"\n❌ NOT VALIDATED (LOYO {loyo_wins}/5, Δ={delta:+.4f})")
        print(f"  Current params confirmed optimal (or insufficient improvement).")
        print(f"  RS={cur_rs}")
        print(f"  RT={cur_rt}")
        print(f"  BS={cur_bs}")
        print(f"  BT={cur_bt}")

    print(f"\nRuntime: {time.time()-t0:.1f}s")
