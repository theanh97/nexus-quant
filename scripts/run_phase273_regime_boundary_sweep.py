"""
Phase 273 — Regime Boundary (p_low / p_high) Sweep [VECTORIZED]
================================================================
Baseline: v2.49.x (Phase 272 result or current config)

Current: p_low=0.30, p_high=0.68  (set ~Phase244, never re-swept)
Since then regime weights changed dramatically — boundary might be sub-optimal.

Approach:
  1. Load all per-year data + run all backtests ONCE (shared)
  2. For each (p_low, p_high) combo: rebuild masks → overlay → eval
  3. Fast because only numpy ops per combo (no extra backtests needed)
  4. 2-pass coordinate descent: fix p_high → sweep p_low, then fix p_low → sweep p_high
  5. LOYO validation + delta-based OBJ update
"""

import os, sys, json, time, subprocess, signal
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
    out = Path("artifacts/phase273"); out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)
signal.signal(signal.SIGALRM, _on_timeout)
signal.alarm(5400)  # 90 min

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

BRD_LB = 192; PCT_WINDOW = 336; FUND_DISP_PCT = 240
TS_SHORT = 16; TS_LONG = 72; VOL_WINDOW = 168
RNAMES = ["LOW", "MID", "HIGH"]

I460_PARAMS = {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

MIN_DELTA = 0.005; MIN_LOYO = 3

# Sweep grid
P_LOW_GRID  = [round(x * 0.05, 2) for x in range(3, 9)]   # 0.15, 0.20, 0.25, 0.30, 0.35, 0.40
P_HIGH_GRID = [round(x * 0.05, 2) for x in range(11, 18)] # 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85
MIN_MID_SPAN = 0.20  # p_high - p_low must be at least 20 ppt

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
        "disp_thr":  disp.get("per_regime_threshold", {"LOW": 0.50, "MID": 0.70, "HIGH": 0.40}),
        "disp_scale":disp.get("per_regime_scale",     {"LOW": 0.70, "MID": 1.80, "HIGH": 0.50}),
        "v1_weights": {r: v1w.get(r, {"w_carry": 0.25, "w_mom": 0.50, "w_mean_reversion": 0.25})
                       for r in RNAMES},
        "version": cfg.get("_version", cfg.get("version", "?")),
        "baseline_obj": cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"],
    }


def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out


def load_year_data(year, P):
    """Load year data AND compute V1 with current per-regime weights."""
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h",
           "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
    n = len(dataset.timeline)

    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n): close_mat[i, j] = dataset.close(sym, i)

    # BTC vol
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1/c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i-VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]

    # Funding dispersion
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

    # TS raw signal
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)

    # Breadth
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))

    # Fixed signal backtests
    fixed_rets = {}
    for sk, sname, params in [
        ("i460", "idio_momentum_alpha", I460_PARAMS),
        ("i415", "idio_momentum_alpha", I415_PARAMS),
        ("f168", "funding_momentum_alpha", F168_PARAMS),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        fixed_rets[sk] = np.array(result.returns)

    # V1 backtests: one per regime weight set
    # Run 3 separate V1s (LOW, MID, HIGH) using current per-regime weights
    v1_by_regime = {}
    for rn in RNAMES:
        vw = P["v1_weights"][rn]
        vp = {"k_per_side": 2,
              "w_carry": vw["w_carry"], "w_mom": vw["w_mom"],
              "w_mean_reversion": vw["w_mean_reversion"],
              "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
              "vol_lookback_bars": 192, "target_gross_leverage": 0.35,
              "rebalance_interval_bars": 60}
        res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": "nexus_alpha_v1", "params": vp}))
        v1_by_regime[rn] = np.array(res.returns)

    return btc_vol, fund_std_pct, ts_raw, brd_pct, fixed_rets, v1_by_regime, n


def build_fast_data_for_boundaries(P, all_year_data, p_low, p_high):
    """Build precomp for a given (p_low, p_high) boundary pair.
    Re-uses all pre-computed per-year data (no additional backtests).
    """
    fast_data = {}
    for yr, yd in sorted(all_year_data.items()):
        btc_vol, fund_std_pct, ts_raw, brd_pct, fixed_rets, v1_by_regime, n = yd
        ml = min(
            min(len(v1_by_regime[r]) for r in RNAMES),
            min(len(fixed_rets[k]) for k in ["i460", "i415", "f168"])
        )
        bpct = brd_pct[:ml]
        bv   = btc_vol[:ml]
        fsp  = fund_std_pct[:ml]
        tr   = ts_raw[:ml]

        # Regime masks with NEW boundaries
        ridx = np.where(bpct < p_low, 0, np.where(bpct > p_high, 2, 1))
        masks = [ridx == i for i in range(3)]

        # TS pct cache
        unique_wins = sorted(set(P["ts_pct_win"].values()))
        ts_pct_cache = {}
        for w in unique_wins:
            arr = np.full(ml, 0.5)
            for i in range(w, ml): arr[i] = float(np.mean(tr[i-w:i] <= tr[i]))
            ts_pct_cache[w] = arr

        # Overlay multiplier (regime-specific thresholds, scales)
        overlay_mult = np.ones(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]
            # HIGH-vol dampening
            overlay_mult[m & ~np.isnan(bv) & (bv > P["vol_thr"][rn])] *= P["vol_scale"][rn]
            # DISP dampening
            overlay_mult[m & (fsp > P["disp_thr"][rn])] *= P["disp_scale"][rn]
            # FTS
            tsp = ts_pct_cache[P["ts_pct_win"][rn]]
            overlay_mult[m & (tsp > P["fts_rt"][rn])] *= P["fts_rs"][rn]
            overlay_mult[m & (tsp < P["fts_bt"][rn])] *= P["fts_bs"][rn]

        scaled = {
            "v1": {rn: overlay_mult * v1_by_regime[rn][:ml] for rn in RNAMES},
            "i460": overlay_mult * fixed_rets["i460"][:ml],
            "i415": overlay_mult * fixed_rets["i415"][:ml],
            "f168": overlay_mult * fixed_rets["f168"][:ml],
        }
        fast_data[yr] = {"scaled": scaled, "masks": masks, "ml": ml}
    return fast_data


def fast_eval(P, fast_data):
    yearly = []
    for yr, fd in sorted(fast_data.items()):
        sc = fd["scaled"]; masks = fd["masks"]; ml = fd["ml"]
        ens = np.zeros(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]; w = P["regime_weights"][rn]
            ens[m] = (w["v1"] * sc["v1"][rn][m] + w["i460"] * sc["i460"][m]
                     + w["i415"] * sc["i415"][m] + w["f168"] * sc["f168"][m])
        r = ens[~np.isnan(ens)]
        ann = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly.append(ann)
    return float(np.mean(yearly) - 0.5 * np.std(yearly, ddof=1))


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 65)
    print("Phase 273 — Regime Boundary (p_low/p_high) Sweep [VECTORIZED]")

    P = load_config_params()
    base_obj = P["baseline_obj"]
    print(f"Baseline: v{P['version']}  OBJ={base_obj:.4f}")
    print(f"Current boundaries: p_low={P['p_low']:.2f}  p_high={P['p_high']:.2f}")
    print(f"Grid: p_low={P_LOW_GRID}  p_high={P_HIGH_GRID}")
    print("=" * 65)

    # [1] Load per-year data (all backtests done once here)
    print("\n[1] Loading per-year data + running backtests (once each)...")
    all_year_data = {}
    for yr in sorted(YEAR_RANGES.keys()):
        print(f"  {yr}: ", end="", flush=True)
        all_year_data[yr] = load_year_data(yr, P)
        print(f"done.  ({time.time()-t0:.0f}s)")

    # [2] Baseline fast data
    print("\n[2] Building baseline fast data...")
    baseline_fd = build_fast_data_for_boundaries(P, all_year_data, P["p_low"], P["p_high"])
    obj_base = fast_eval(P, baseline_fd)
    print(f"  Measured baseline OBJ = {obj_base:.4f}  (config says {base_obj:.4f})")

    # [3] 2-pass coordinate descent on (p_low, p_high)
    print("\n[3] 2-pass boundary sweep...")
    best_p_low = P["p_low"]
    best_p_high = P["p_high"]
    best_obj = obj_base

    for pass_num in range(1, 3):
        print(f"\n  === Pass {pass_num} ===")

        # Sweep p_low (fixed p_high)
        print(f"  Sweeping p_low (p_high={best_p_high:.2f}):")
        for pl in P_LOW_GRID:
            if best_p_high - pl < MIN_MID_SPAN: continue
            fd = build_fast_data_for_boundaries(P, all_year_data, pl, best_p_high)
            obj = fast_eval(P, fd)
            marker = " <--" if obj > best_obj + 1e-6 else ""
            print(f"    p_low={pl:.2f}  OBJ={obj:.4f}{marker}")
            if obj > best_obj + 1e-6:
                best_obj = obj; best_p_low = pl

        # Sweep p_high (fixed p_low)
        print(f"  Sweeping p_high (p_low={best_p_low:.2f}):")
        for ph in P_HIGH_GRID:
            if ph - best_p_low < MIN_MID_SPAN: continue
            fd = build_fast_data_for_boundaries(P, all_year_data, best_p_low, ph)
            obj = fast_eval(P, fd)
            marker = " <--" if obj > best_obj + 1e-6 else ""
            print(f"    p_high={ph:.2f}  OBJ={obj:.4f}{marker}")
            if obj > best_obj + 1e-6:
                best_obj = obj; best_p_high = ph

        print(f"  Pass {pass_num} best: p_low={best_p_low:.2f}  p_high={best_p_high:.2f}  OBJ={best_obj:.4f}")

    delta = best_obj - obj_base
    print(f"\n  Final: p_low={best_p_low:.2f}  p_high={best_p_high:.2f}  OBJ={best_obj:.4f}  Δ={delta:+.4f}")

    # [4] LOYO validation
    print("\n[4] LOYO validation...")
    best_fd = build_fast_data_for_boundaries(P, all_year_data, best_p_low, best_p_high)
    loyo_wins = 0
    for yr in sorted(YEAR_RANGES.keys()):
        others = [y for y in YEAR_RANGES if y != yr]
        def subset_obj(fd, yrs):
            sh = []
            for y in yrs:
                d = fd[y]; sc = d["scaled"]; masks = d["masks"]; ml = d["ml"]
                ens = np.zeros(ml)
                for ri, rn in enumerate(RNAMES):
                    m = masks[ri]; w = P["regime_weights"][rn]
                    ens[m] = (w["v1"]*sc["v1"][rn][m] + w["i460"]*sc["i460"][m]
                             + w["i415"]*sc["i415"][m] + w["f168"]*sc["f168"][m])
                r = ens[~np.isnan(ens)]
                ann = float(np.mean(r)/np.std(r,ddof=1)*np.sqrt(8760)) if len(r) > 1 else 0.0
                sh.append(ann)
            return float(np.mean(sh) - 0.5*np.std(sh, ddof=1))
        base_loyo = subset_obj(baseline_fd, others)
        cand_loyo = subset_obj(best_fd, others)
        win = cand_loyo > base_loyo
        loyo_wins += int(win)
        print(f"  {yr}: base={base_loyo:.4f}  cand={cand_loyo:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={best_obj:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/5")
    validated = delta >= MIN_DELTA and loyo_wins >= MIN_LOYO

    if validated:
        print(f"\n✅ VALIDATED  (p_low={best_p_low:.2f}  p_high={best_p_high:.2f})")
        cfg = json.load(open(CFG_PATH))
        # Update boundaries
        cfg["breadth_regime_switching"]["p_low"] = best_p_low
        cfg["breadth_regime_switching"]["p_high"] = best_p_high
        # Version bump
        old_ver = cfg.get("_version", "2.49.2")
        parts = old_ver.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = ".".join(parts)
        cfg["_version"] = new_ver
        # Also update the "version" field
        old_vver = cfg.get("version", "v2.49.2")
        vparts = old_vver.lstrip("v").split(".")
        vparts[-1] = str(int(vparts[-1]) + 1)
        cfg["version"] = "v" + ".".join(vparts)
        # Delta-based OBJ update
        stored_obj = cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"]
        reported_obj = round(stored_obj + delta, 4)
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = reported_obj
        json.dump(cfg, open(CFG_PATH, "w"), indent=2)
        print(f"\n  Config → {cfg['version']}  OBJ={reported_obj:.4f} (stored {stored_obj:.4f} + Δ{delta:+.4f})")
        print(f"  p_low: {P['p_low']:.2f} → {best_p_low:.2f}")
        print(f"  p_high: {P['p_high']:.2f} → {best_p_high:.2f}")
        msg = (f"feat: P273 regime boundary sweep OBJ={reported_obj:.4f} "
               f"LOYO={loyo_wins}/5 D={delta:+.4f} [p_low={best_p_low:.2f}/p_high={best_p_high:.2f}]")
        subprocess.run(["git", "add", str(CFG_PATH)], cwd=ROOT)
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
    else:
        print(f"\n❌ NOT VALIDATED (LOYO {loyo_wins}/5, Δ={delta:+.4f})")
        print(f"  Current boundaries confirmed: p_low={P['p_low']:.2f}  p_high={P['p_high']:.2f}")

    print(f"\nRuntime: {time.time()-t0:.1f}s")
    print("[DONE] Phase 273 complete.")
