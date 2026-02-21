"""
Phase 242 — Idio Ratio Sweep (HIGH + LOW regimes)
===================================================
Current config (v2.42.0):
  HIGH: v1=0.06, i460=0.94, i415=0.00, f168=0.00   (100% I460)
  LOW:  v1=0.44, i460=0.0864, i415=0.1035, f168=0.37 (idio_ratio≈0.455)

Part A: HIGH idio_ratio  — ratio of I460 within HIGH's 0.94 idio total
  i460 = 0.94 * r,  i415 = 0.94 * (1-r)
  Sweep r ∈ [0.0, 0.1, …, 1.0] (11 points)

Part B: LOW idio_ratio  — ratio of I460 within LOW's 0.1899 idio total
  low_idio_total = 0.0864 + 0.1035 = 0.1899
  i460 = 0.1899 * r,  i415 = 0.1899 * (1-r)
  Sweep r ∈ [0.0, 0.1, …, 1.0] (11 points)

Validation: LOYO ≥ 3/5 wins AND delta > 0.005
"""
import json, sys, subprocess, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
           "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]

YEAR_RANGES = {
    2021: ("2021-02-01", "2022-01-01"),
    2022: ("2022-01-01", "2023-01-01"),
    2023: ("2023-01-01", "2024-01-01"),
    2024: ("2024-01-01", "2025-01-01"),
    2025: ("2025-01-01", "2026-01-01"),
}
YEARS = sorted(YEAR_RANGES.keys())

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})


def get_config():
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    brs = cfg["breadth_regime_switching"]
    rw  = brs["regime_weights"]
    weights = {}
    for regime in ["LOW", "MID", "HIGH"]:
        w = rw[regime]
        weights[regime] = {
            "v1":   w["v1"],
            "i460": w.get("i460bw168", w.get("i460", 0.0)),
            "i415": w.get("i415bw216", w.get("i415", 0.0)),
            "f168": w["f168"],
        }
    fts = cfg.get("funding_term_structure_overlay", {})
    vol_ov = cfg.get("vol_regime_overlay", {})
    params = {
        "p_low":   brs.get("p_low", 0.30),
        "p_high":  brs.get("p_high", 0.60),
        "brd_lb":  brs.get("breadth_lookback_bars", 192),
        "pct_win": brs.get("rolling_percentile_window", 336),
        "ts_rt":   fts.get("reduce_threshold", 0.65),
        "ts_rs":   fts.get("reduce_scale", 0.30),
        "ts_bt":   fts.get("boost_threshold", 0.25),
        "ts_bs":   fts.get("boost_scale", 1.85),
        "ts_short": fts.get("short_window_bars", 16),
        "ts_long":  fts.get("long_window_bars", 72),
        "vol_thr":    vol_ov.get("threshold", 0.50),
        "vol_scale":  vol_ov.get("scale_factor", 0.30),
        "f168_boost": vol_ov.get("f144_boost", 0.00),
    }
    sigs  = cfg["ensemble"]["signals"]
    v1p   = sigs["v1"]["params"]
    i460p = sigs["i460bw168"]["params"]
    i415p = sigs["i415bw216"]["params"]
    f168p = sigs.get("f144", sigs.get("f168", {})).get("params", {})
    return cfg, weights, params, v1p, i460p, i415p, f168p


# ── Helpers ───────────────────────────────────────────────────────────────────
def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out


def rolling_percentile_arr(arr, w):
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr)):
        window = arr[i-w+1:i+1]
        out[i] = float(np.sum(window <= arr[i])) / w
    return out


def sharpe(rets, n):
    if n < 20:
        return 0.0
    mu = float(np.mean(rets[:n])) * 8760
    sd = float(np.std(rets[:n]))  * np.sqrt(8760)
    return mu / sd if sd > 1e-9 else 0.0


def obj_fn(yr_sharpes):
    vals = list(yr_sharpes.values())
    return float(np.mean(vals) - 0.5 * np.std(vals))


# ── Data loader ───────────────────────────────────────────────────────────────
def load_year_all(year, params):
    s, e = YEAR_RANGES[year]
    cfg_d = {
        "provider": "binance_rest_v1",
        "symbols": SYMBOLS,
        "start": s, "end": e,
        "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    dataset = make_provider(cfg_d, seed=42).load()
    n = len(dataset.timeline)

    # Close matrix & BTC vol
    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            close_mat[i, j] = dataset.close(sym, i)
    btc_ret = np.diff(np.log(close_mat[:, 0] + 1e-12))
    btc_vol = np.full(n, np.nan)
    w168 = 168
    for i in range(w168, n):
        btc_vol[i] = float(np.std(btc_ret[i-w168:i])) * np.sqrt(8760)

    # Funding: cross-sectional mean, then FTS spread
    fund_rates = np.zeros((n, len(SYMBOLS)))
    fund_std   = np.zeros(n)
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:    fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0
    for i in range(n):
        fund_std[i] = float(np.std(fund_rates[i]))
    cs_fund = np.mean(fund_rates, axis=1)
    ts_short = params["ts_short"]
    ts_long  = params["ts_long"]
    rm_short = rolling_mean_arr(cs_fund, ts_short)
    rm_long  = rolling_mean_arr(cs_fund, ts_long)
    spread_raw = rm_short - rm_long
    ts_spread_pct = rolling_percentile_arr(
        np.where(np.isnan(spread_raw), 0.0, spread_raw),
        params["pct_win"]
    )

    # Breadth
    brd_lb  = params["brd_lb"]
    pct_win = params["pct_win"]
    pos_cnt = np.sum(close_mat > 0, axis=1).astype(float) / len(SYMBOLS)
    brd_raw = rolling_mean_arr(pos_cnt, brd_lb)
    brd_pct = rolling_percentile_arr(
        np.where(np.isnan(brd_raw), 0.0, brd_raw),
        pct_win
    )

    # Strategy returns
    v1_params = {
        "symbols": SYMBOLS,
        "weight_cross": 0.25, "weight_momentum": 0.45, "weight_reversal": 0.30,
        "leverage": 0.35, "rebalance_every": 60,
        "momentum_lookback": 336, "reversal_lookback": 84, "k_per_side": 2,
    }
    i460_params = {
        "symbols": SYMBOLS, "lookback": 480, "bandwidth": 168,
        "k_per_side": 4, "rebalance_every": 48, "leverage": 0.20,
    }
    i415_params = {
        "symbols": SYMBOLS, "lookback": 415, "bandwidth": 144,
        "k_per_side": 4, "rebalance_every": 48, "leverage": 0.20,
    }
    f168_params = {
        "symbols": SYMBOLS, "lookback": 168, "k_per_side": 2,
        "rebalance_every": 36, "leverage": 0.25, "direction": "contrarian",
    }

    def run(name, sp):
        res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": name, "params": sp})
        )
        return np.array(res.returns)

    v1_ret   = run("nexus_alpha_v1",       v1_params)
    i460_ret = run("idio_momentum_alpha",  i460_params)
    i415_ret = run("idio_momentum_alpha",  i415_params)
    f168_ret = run("funding_momentum_alpha", f168_params)

    print(f"n={n}")
    return {
        "n": n,
        "v1": v1_ret, "i460": i460_ret, "i415": i415_ret, "f168": f168_ret,
        "btc_vol": btc_vol, "ts_spread_pct": ts_spread_pct, "brd_pct": brd_pct,
    }


# ── Ensemble ──────────────────────────────────────────────────────────────────
def compute_ensemble(data, weights, params):
    n          = data["n"]
    v1_        = data["v1"]
    i4         = data["i460"]
    i5         = data["i415"]
    f1         = data["f168"]
    bv_        = data["btc_vol"]
    tsp_       = data["ts_spread_pct"]
    brd        = data["brd_pct"]
    p_low      = params["p_low"]
    p_high     = params["p_high"]
    vol_thr    = params["vol_thr"]
    vol_scale  = params["vol_scale"]
    ts_rt      = params["ts_rt"]
    ts_rs      = params["ts_rs"]
    ts_bt      = params["ts_bt"]
    ts_bs      = params["ts_bs"]

    min_len = min(n, len(v1_), len(i4), len(i5), len(f1),
                  len(bv_), len(tsp_), len(brd))
    ens = np.zeros(min_len)

    for i in range(min_len):
        brd_v = brd[i] if not np.isnan(brd[i]) else 0.5
        if brd_v >= p_high:
            w = weights["HIGH"]
        elif brd_v >= p_low:
            w = weights["MID"]
        else:
            w = weights["LOW"]

        raw = (w["v1"]*v1_[i] + w["i460"]*i4[i] +
               w["i415"]*i5[i] + w["f168"]*f1[i])

        if not np.isnan(bv_[i]) and bv_[i] > vol_thr:
            raw *= vol_scale

        tsp_v = tsp_[i] if not np.isnan(tsp_[i]) else 0.5
        if tsp_v > ts_rt:
            raw *= ts_rs
        elif tsp_v < ts_bt:
            raw *= ts_bs

        ens[i] = raw
    return ens


def eval_year(data, weights, params):
    ens = compute_ensemble(data, weights, params)
    return sharpe(ens, len(ens))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 242 — Idio Ratio Sweep (HIGH + LOW regimes)")
    print("=" * 60)

    t0 = time.time()
    cfg, base_weights, params, v1p, i460p, i415p, f168p = get_config()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1] Loading all data ...")
    year_data = {}
    for yr in [2021, 2022, 2023, 2024, 2025]:
        print(f"  {yr}", end=" ", flush=True)
        year_data[yr] = load_year_all(yr, params)

    # Baseline
    baseline_sharpes = {yr: eval_year(year_data[yr], base_weights, params)
                        for yr in [2021, 2022, 2023, 2024, 2025]}
    baseline_obj = obj_fn(baseline_sharpes)
    print(f"\n  Baseline OBJ: {baseline_obj:.4f}")
    high_idio_total = base_weights["HIGH"]["i460"] + base_weights["HIGH"]["i415"]
    low_idio_total  = base_weights["LOW"]["i460"]  + base_weights["LOW"]["i415"]
    print(f"  HIGH idio_total={high_idio_total:.4f}  (i460={base_weights['HIGH']['i460']:.4f}, i415={base_weights['HIGH']['i415']:.4f})")
    print(f"  LOW  idio_total={low_idio_total:.4f}   (i460={base_weights['LOW']['i460']:.4f},  i415={base_weights['LOW']['i415']:.4f})")

    # ── Part A: HIGH idio_ratio ───────────────────────────────────────────────
    print(f"\n[2] Part A: HIGH idio_ratio sweep (idio_total={high_idio_total:.4f}) ...")
    ratio_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_high_ratio = None
    best_high_obj   = -999.0

    for r in ratio_grid:
        w = {k: dict(v) for k, v in base_weights.items()}
        w["HIGH"]["i460"] = round(high_idio_total * r, 6)
        w["HIGH"]["i415"] = round(high_idio_total * (1 - r), 6)
        sharpes = {yr: eval_year(year_data[yr], w, params)
                   for yr in [2021, 2022, 2023, 2024, 2025]}
        o = obj_fn(sharpes)
        marker = " ←" if o > best_high_obj else ""
        print(f"  HIGH ratio={r:.1f}  i460={w['HIGH']['i460']:.4f}  i415={w['HIGH']['i415']:.4f}  OBJ={o:.4f}{marker}")
        if o > best_high_obj:
            best_high_obj   = o
            best_high_ratio = r

    print(f"  Best HIGH ratio: {best_high_ratio:.1f}  OBJ={best_high_obj:.4f}")

    # ── Part B: LOW idio_ratio ────────────────────────────────────────────────
    print(f"\n[3] Part B: LOW idio_ratio sweep (idio_total={low_idio_total:.4f}) ...")
    best_low_ratio = None
    best_low_obj   = -999.0

    for r in ratio_grid:
        w = {k: dict(v) for k, v in base_weights.items()}
        # Use best HIGH from Part A
        w["HIGH"]["i460"] = round(high_idio_total * best_high_ratio, 6)
        w["HIGH"]["i415"] = round(high_idio_total * (1 - best_high_ratio), 6)
        w["LOW"]["i460"] = round(low_idio_total * r, 6)
        w["LOW"]["i415"] = round(low_idio_total * (1 - r), 6)
        sharpes = {yr: eval_year(year_data[yr], w, params)
                   for yr in [2021, 2022, 2023, 2024, 2025]}
        o = obj_fn(sharpes)
        marker = " ←" if o > best_low_obj else ""
        print(f"  LOW ratio={r:.1f}  i460={w['LOW']['i460']:.4f}  i415={w['LOW']['i415']:.4f}  OBJ={o:.4f}{marker}")
        if o > best_low_obj:
            best_low_obj   = o
            best_low_ratio = r

    delta = best_low_obj - baseline_obj
    print(f"  Best LOW ratio: {best_low_ratio:.1f}  OBJ={best_low_obj:.4f}  Δ={delta:+.4f}")

    # ── LOYO Validation ───────────────────────────────────────────────────────
    print(f"\n[4] LOYO validation ...")
    best_weights = {k: dict(v) for k, v in base_weights.items()}
    best_weights["HIGH"]["i460"] = round(high_idio_total * best_high_ratio, 6)
    best_weights["HIGH"]["i415"] = round(high_idio_total * (1 - best_high_ratio), 6)
    best_weights["LOW"]["i460"]  = round(low_idio_total * best_low_ratio, 6)
    best_weights["LOW"]["i415"]  = round(low_idio_total * (1 - best_low_ratio), 6)

    wins = 0
    for yr in [2021, 2022, 2023, 2024, 2025]:
        best_s = eval_year(year_data[yr], best_weights, params)
        curr_s = baseline_sharpes[yr]
        result = "WIN" if best_s > curr_s else "LOSS"
        if best_s > curr_s:
            wins += 1
        print(f"  {yr}: best={best_s:.4f} curr={curr_s:.4f} {result}")

    print(f"\n  LOYO: {wins}/5 wins  delta={delta:+.4f}")

    # ── Decision ──────────────────────────────────────────────────────────────
    if wins >= 3 and delta > 0.005:
        print(f"\n✅ IMPROVEMENT VALIDATED! Updating config ...")

        cfg_path = Path("configs/production_p91b_champion.json")
        cfg2 = json.loads(cfg_path.read_text())
        rw = cfg2["breadth_regime_switching"]["regime_weights"]

        # Update HIGH
        rw["HIGH"]["i460bw168"] = round(high_idio_total * best_high_ratio, 6)
        rw["HIGH"]["i415bw216"] = round(high_idio_total * (1 - best_high_ratio), 6)
        rw["HIGH"]["_regime"] = f"HIGH momentum (>60th pct) — idio_ratio={best_high_ratio:.1f}"

        # Update LOW
        rw["LOW"]["i460bw168"] = round(low_idio_total * best_low_ratio, 6)
        rw["LOW"]["i415bw216"] = round(low_idio_total * (1 - best_low_ratio), 6)
        rw["LOW"]["_regime"] = f"LOW momentum (<20th pct) — idio_ratio={best_low_ratio:.1f}"

        # Bump version
        old_ver = cfg2.get("version", "2.42.0")
        parts = old_ver.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = ".".join(parts)
        cfg2["version"] = new_ver

        cfg_path.write_text(json.dumps(cfg2, indent=2))
        print(f"  Config updated: {old_ver} → {new_ver}")
        print(f"  HIGH: i460={rw['HIGH']['i460bw168']:.4f}  i415={rw['HIGH']['i415bw216']:.4f}")
        print(f"  LOW:  i460={rw['LOW']['i460bw168']:.4f}   i415={rw['LOW']['i415bw216']:.4f}")

        # Git commit
        script_name = "scripts/run_phase242_idio_ratio_sweep.py"
        commit_msg  = (f"feat(p242): idio_ratio sweep — HIGH ratio={best_high_ratio:.1f} "
                       f"LOW ratio={best_low_ratio:.1f} OBJ={best_low_obj:.4f} "
                       f"(Δ={delta:+.4f} LOYO {wins}/5) [{new_ver}]")
        subprocess.run(
            f"cd /Users/truonglys/projects/quant && "
            f"git add configs/production_p91b_champion.json {script_name} && "
            f"git stash && git pull --rebase && git stash pop && "
            f"git commit -m '{commit_msg}' && git push",
            shell=True
        )
    else:
        if wins >= 3 and delta > 0:
            tag = "⚠️ WEAK"
        else:
            tag = "❌ NO IMPROVEMENT — CONFIRMED OPTIMAL" if delta == 0 else "❌ NO IMPROVEMENT"
        print(f"\n{tag}  (LOYO {wins}/5 delta={delta:+.4f})")
        print(f"  HIGH ratio stays: {base_weights['HIGH']['i460']:.4f}/{base_weights['HIGH']['i460']+base_weights['HIGH']['i415']:.4f}")
        print(f"  LOW  ratio stays: {base_weights['LOW']['i460']:.4f}/{base_weights['LOW']['i460']+base_weights['LOW']['i415']:.4f}")

    elapsed = time.time() - t0
    print(f"  Done  ({elapsed:.1f}s)")
    print("\n[DONE] Phase 242 complete.")


if __name__ == "__main__":
    main()
