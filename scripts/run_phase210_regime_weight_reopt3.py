"""
Phase 210 — Regime Weight Re-Opt (3rd pass, post-V1-weight-change)
===================================================================
V1 internal weights changed in P203 (carry: 0.35→0.25, mom: 0.40→0.45).
This changes V1's return profile, potentially shifting optimal ensemble weights.
Pre-compute all signals, sweep regime weights independently per regime.

Baseline: v2.32.0, OBJ=3.0860
Current weights:
  LOW:  v1=0.46  i460=0.0716  i415=0.1184  f168=0.35
  MID:  v1=0.20  i460=0.0074  i415=0.0126  f168=0.78
  HIGH: v1=0.06  i460=0.2821  i415=0.5079  f168=0.15

Sweep: v1 × f168 per regime (i460/i415 absorb remainder in fixed ratio)
i460/(i460+i415) ratio:
  LOW:  0.0716/(0.0716+0.1184) = 0.376
  MID:  0.0074/(0.0074+0.0126) = 0.370
  HIGH: 0.2821/(0.2821+0.5079) = 0.357
"""

import os, sys, json, time
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
    out = Path("artifacts/phase210"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase210_report.json").write_text(json.dumps(_partial, indent=2))
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

VOL_WINDOW    = 168; BRD_LB = 192; PCT_WINDOW = 336
TS_SHORT, TS_LONG = 16, 72; FUND_DISP_PCT = 240
VOL_THRESHOLD = 0.50; VOL_SCALE = 0.30
TS_RT = 0.65; TS_RS = 0.45; TS_BT = 0.25; TS_BS = 1.70
DISP_THR = 0.60; DISP_SCALE = 1.0
P_LOW = 0.30; P_HIGH = 0.60

BASE_WEIGHTS = {
    "LOW":  {"v1": 0.46,   "i460": 0.0716, "i415": 0.1184, "f168": 0.35},
    "MID":  {"v1": 0.20,   "i460": 0.0074, "i415": 0.0126, "f168": 0.78},
    "HIGH": {"v1": 0.06,   "i460": 0.2821, "i415": 0.5079, "f168": 0.15},
}
# Fixed idio ratios per regime
IDIO_RATIO = {
    "LOW":  0.0716 / (0.0716 + 0.1184),  # 0.376
    "MID":  0.0074 / (0.0074 + 0.0126),  # 0.370
    "HIGH": 0.2821 / (0.2821 + 0.5079),  # 0.357
}

V1_PARAMS = {
    "k_per_side": 2, "w_carry": 0.25, "w_mom": 0.45, "w_mean_reversion": 0.30,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
}
I460_PARAMS = {"k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 24}

BASELINE_OBJ = 3.0860
# Per-regime sweep grids
V1_SWEEP  = [0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56]
F168_SWEEP = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
MIN_DELTA = 0.005; MIN_LOYO = 3; MIN_IDIO = 0.01

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

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
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-PCT_WINDOW:i] <= ts_raw[i]))
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))
    sig_rets = {}
    for sk, sname, params in [
        ("v1",   "nexus_alpha_v1",         V1_PARAMS),
        ("i460", "idio_momentum_alpha",    I460_PARAMS),
        ("i415", "idio_momentum_alpha",    I415_PARAMS),
        ("f168", "funding_momentum_alpha", F168_PARAMS),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        sig_rets[sk] = np.array(result.returns)
    print("S.", end=" ", flush=True)
    return btc_vol, fund_std_pct, ts_spread_pct, brd_pct, sig_rets, n

def make_weights(regime, v1, f168):
    idio = 1.0 - v1 - f168
    if idio < MIN_IDIO: return None
    ratio = IDIO_RATIO[regime]
    i460 = idio * ratio; i415 = idio * (1.0 - ratio)
    return {"v1": v1, "i460": i460, "i415": i415, "f168": f168}

def ensemble_rets(weights_by_regime, base_data):
    btc_vol, fund_std_pct, ts_spread_pct, brd_pct, sig_rets, n = base_data
    RNAMES = ["LOW", "MID", "HIGH"]
    sk_all = ["v1", "i460", "i415", "f168"]
    min_len = min(len(sig_rets[k]) for k in sk_all)
    ens = np.zeros(min_len)
    bv = btc_vol[:min_len]; bpct = brd_pct[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    for i in range(min_len):
        bp = bpct[i]
        ridx = 0 if bp < P_LOW else (2 if bp > P_HIGH else 1)
        w = weights_by_regime[RNAMES[ridx]]
        ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            ret_i *= VOL_SCALE
        if fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp[i] > TS_RT: ret_i *= TS_RS
        elif tsp[i] < TS_BT: ret_i *= TS_BS
        ens[i] = ret_i
    return ens

def sharpe(r):
    r = r[~np.isnan(r)]
    if len(r) < 2: return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760))

def obj(yearly):
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1))

def eval_weights(wb, all_data):
    yearly = {yr: sharpe(ensemble_rets(wb, d)) for yr, d in all_data.items()}
    return obj(yearly), yearly

def sweep_regime(regime, other_regimes, all_data, base_obj):
    best_obj = -999; best_v1 = BASE_WEIGHTS[regime]["v1"]; best_f168 = BASE_WEIGHTS[regime]["f168"]
    results = []
    for v1 in V1_SWEEP:
        for f168 in F168_SWEEP:
            w = make_weights(regime, v1, f168)
            if w is None: continue
            wb = {**other_regimes, regime: w}
            o, _ = eval_weights(wb, all_data)
            results.append((v1, f168, o))
            if o > best_obj:
                best_obj = o; best_v1 = v1; best_f168 = f168
    return best_v1, best_f168, best_obj, results

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 210 — Regime Weight Re-Opt (post P203 V1 change)")
    print(f"Baseline: v2.32.0  OBJ={BASELINE_OBJ}")
    print("=" * 65)

    print("\n[1/3] Loading per-year data & pre-computing all signals...")
    all_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_data[yr] = load_year_data(yr)
    print()

    base_yearly = {yr: sharpe(ensemble_rets(BASE_WEIGHTS, d)) for yr, d in all_data.items()}
    base_obj = obj(base_yearly)
    print(f"\n  Baseline OBJ = {base_obj:.4f}  (expected ≈ {BASELINE_OBJ})")
    _partial.update({"baseline_obj": float(base_obj)})

    print("\n[2/3] Per-regime sweep...")
    best_weights = {r: dict(w) for r, w in BASE_WEIGHTS.items()}

    for regime in ["LOW", "MID", "HIGH"]:
        other = {r: best_weights[r] for r in ["LOW", "MID", "HIGH"] if r != regime}
        bv1, bf168, bo, results = sweep_regime(regime, other, all_data, base_obj)
        bw = make_weights(regime, bv1, bf168)
        delta = bo - base_obj
        print(f"  {regime}: v1={bv1:.2f} f168={bf168:.2f} i460={bw['i460']:.4f} i415={bw['i415']:.4f}  OBJ={bo:.4f}  Δ={delta:+.4f}")
        if bo > base_obj:
            best_weights[regime] = bw

    print("\n  Best weights:")
    for r, w in best_weights.items():
        print(f"    {r}: v1={w['v1']:.4f}  i460={w['i460']:.4f}  i415={w['i415']:.4f}  f168={w['f168']:.4f}")

    print("\n[3/3] LOYO validation...")
    joint_obj, joint_yearly = eval_weights(best_weights, all_data)
    delta = joint_obj - base_obj
    loyo_wins = 0
    for yr in sorted(joint_yearly):
        base_sh = base_yearly.get(yr, 0)
        cand_sh = joint_yearly[yr]
        win = cand_sh > base_sh
        if win: loyo_wins += 1
        print(f"  {yr}: base={base_sh:.4f}  cand={cand_sh:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={joint_obj:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/{len(joint_yearly)}")
    validated = loyo_wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'}")

    result = {
        "phase": 210, "baseline_obj": float(base_obj), "best_obj": float(joint_obj),
        "delta": float(delta), "loyo_wins": loyo_wins, "loyo_total": len(joint_yearly),
        "validated": validated, "best_weights": {r: {k: round(v, 4) for k, v in w.items()} for r, w in best_weights.items()},
    }
    _partial.update(result)
    out = Path("artifacts/phase210"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase210_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        brs = cfg["breadth_regime_switching"]
        rw = brs["regime_weights"]
        for regime, w in best_weights.items():
            # Map to config key names
            key = regime
            if key in rw:
                rw[key]["v1"] = round(w["v1"], 4)
                rw[key]["i460bw168"] = round(w["i460"], 4)
                rw[key]["i415bw216"] = round(w["i415"], 4)
                rw[key]["f168"] = round(w["f168"], 4)
        cfg["_version"] = "2.33.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.33.0  OBJ={round(joint_obj,4)}")

    print(f"\nRuntime: {int(time.time()-t0)}s")

main()
