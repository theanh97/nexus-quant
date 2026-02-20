#!/usr/bin/env python3
"""
Phase 89: I415_bw216 + I600=5% Fine Grid + Dual-bw216 + Ultra-Fine
====================================================================
Phase 88 discoveries:
  - I410_bw216 + I600=5%: AVG=2.015, MIN=1.529 ← P88 CHAMP
  - I415_bw216 + I600=7.5%: AVG=2.002, MIN=1.526 — UNTESTED with I600=5%!
  - I415_bw216: [1.967, 1.737, 1.143, 2.015, 1.338] MIN=1.143 (ALL years > 1.0)
  - I460_bw216 + I415_bw216 + I600=7.5%: AVG=2.004, MIN=1.502
  - I600 trend: 5% > 7.5% > 10% for balanced MIN

Phase 89 agenda:
  A) Run reference signals + I415_bw216 + I410_bw216 + I460 variants
  B) I460_bw168 + I415_bw216 + I600=5% (P88's E2 + better I600) ← KEY!
  C) I460_bw216 + I415_bw216 + I600=5% (P88's E3 + better I600)
  D) I460_bw168 + I410_bw216 + I415_bw216 (triple idio, all bw=216) + I600=5%
  E) Fine 0.625% grid around best champion from B/C/D
  F) Ultra-fine 0.3125% grid around absolute best
  G) AVG-max push: replace I460_bw168 with I474_bw216 for high-AVG variant
  H) Pareto summary, champion save
"""

import json, math, statistics, subprocess, sys, copy
from pathlib import Path
import numpy as np

OUT_DIR = "artifacts/phase89"
YEARS = ["2021", "2022", "2023", "2024", "2025"]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
           "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

YEAR_RANGES = {
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}

BASE_CONFIG = {
    "seed": 42,
    "venue": {"name": "binance_usdm", "kind": "crypto_perp", "vip_tier": 0},
    "data": {
        "provider": "binance_rest_v1",
        "symbols": SYMBOLS,
        "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    },
    "execution": {"style": "taker", "slippage_bps": 3.0},
    "costs": {"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0},
    "benchmark": {"version": "v1", "walk_forward": {"enabled": True}},
    "risk": {
        "max_drawdown": 0.30,
        "max_turnover_per_rebalance": 0.8,
        "max_gross_leverage": 0.7,
        "max_position_per_symbol": 0.3,
    },
    "self_learn": {"enabled": False},
}

V1_PARAMS = {
    "k_per_side": 2,
    "w_carry": 0.35, "w_mom": 0.45, "w_confirm": 0.0,
    "w_mean_reversion": 0.20, "w_vol_momentum": 0.0, "w_funding_trend": 0.0,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
    "vol_lookback_bars": 168,
    "target_portfolio_vol": 0.0, "use_min_variance": False,
    "target_gross_leverage": 0.35, "min_gross_leverage": 0.05, "max_gross_leverage": 0.65,
    "rebalance_interval_bars": 60, "strict_agreement": False,
}

def make_fund_params(lb: int, k: int = 2) -> dict:
    return {
        "k_per_side": k, "funding_lookback_bars": lb, "direction": "contrarian",
        "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
    }

def make_idio_params(lookback: int, beta_window: int, k: int = 2) -> dict:
    return {
        "k_per_side": k, "lookback_bars": lookback, "beta_window_bars": beta_window,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }

def compute_metrics(result_path: str) -> dict:
    d = json.load(open(result_path))
    rets = d.get("returns", [])
    if not rets or len(rets) < 100:
        return {"sharpe": 0.0, "error": "insufficient data", "returns_np": None}
    arr = np.array(rets, dtype=np.float64)
    sharpe = float(arr.mean() / arr.std() * np.sqrt(8760)) if arr.std() > 0 else 0.0
    return {"sharpe": round(sharpe, 3), "returns_np": arr}

def make_config(run_name: str, year: str, strategy_name: str, params: dict) -> str:
    cfg = copy.deepcopy(BASE_CONFIG)
    start, end = YEAR_RANGES[year]
    cfg["run_name"] = run_name
    cfg["data"]["start"] = start
    cfg["data"]["end"] = end
    cfg["strategy"] = {"name": strategy_name, "params": params}
    path = f"/tmp/phase89_{run_name}_{year}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path

def run_strategy(label: str, strategy_name: str, params: dict) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Running: {label}", flush=True)
    year_results = {}
    for year in YEARS:
        run_name = f"{label}_{year}"
        config_path = make_config(run_name, year, strategy_name, params)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "nexus_quant", "run",
                 "--config", config_path, "--out", OUT_DIR],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                year_results[year] = {"error": "run failed", "sharpe": 0.0, "returns_np": None}
                print(f"    {year}: ERROR", flush=True)
                continue
        except subprocess.TimeoutExpired:
            year_results[year] = {"error": "timeout", "sharpe": 0.0, "returns_np": None}
            print(f"    {year}: TIMEOUT", flush=True)
            continue

        runs_dir = Path(OUT_DIR) / "runs"
        if not runs_dir.exists():
            year_results[year] = {"error": "no runs dir", "sharpe": 0.0, "returns_np": None}
            continue
        matching = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(run_name)],
            key=lambda d: d.stat().st_mtime,
        )
        if matching:
            rp = matching[-1] / "result.json"
            if rp.exists():
                m = compute_metrics(str(rp))
                year_results[year] = m
                print(f"    {year}: Sharpe={m['sharpe']}", flush=True)
                continue
        year_results[year] = {"error": "no result", "sharpe": 0.0, "returns_np": None}
        print(f"    {year}: no result", flush=True)

    sharpes = [year_results[y].get("sharpe", 0.0) for y in YEARS
               if isinstance(year_results.get(y, {}).get("sharpe"), (int, float))]
    avg = round(sum(sharpes) / len(sharpes), 3) if sharpes else 0.0
    mn = round(min(sharpes), 3) if sharpes else 0.0
    yby = [round(year_results.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    year_results["_avg"] = avg
    year_results["_min"] = mn
    year_results["_pos"] = sum(1 for s in sharpes if s > 0)
    print(f"  → AVG={avg}, MIN={mn}", flush=True)
    print(f"  → YbY: {yby}", flush=True)
    return year_results

def build_year_matrices(base: dict, strat_names: list) -> dict:
    year_matrices = {}
    for year in YEARS:
        arrays = []
        for name in strat_names:
            arr = base[name][year].get("returns_np")
            arrays.append(arr if arr is not None else np.zeros(1))
        min_len = min(len(a) for a in arrays)
        year_matrices[year] = np.stack([a[:min_len] for a in arrays])
    return year_matrices

def sweep_configs_numpy(weight_configs: list, strat_names: list,
                        year_matrices: dict) -> list:
    if not weight_configs:
        return []
    W = np.array([[wc.get(n, 0.0) for n in strat_names]
                  for wc in weight_configs], dtype=np.float64)
    year_sharpes = {}
    for year in YEARS:
        R = year_matrices[year]
        B = W @ R
        mu = B.mean(axis=1)
        sd = B.std(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            year_sharpes[year] = np.where(sd > 0, mu / sd * np.sqrt(8760), 0.0)
    yby_matrix = np.stack([year_sharpes[y] for y in YEARS], axis=1)
    avgs = yby_matrix.mean(axis=1)
    mins = yby_matrix.min(axis=1)
    return [(float(avgs[i]), float(mins[i]), wc) for i, wc in enumerate(weight_configs)]

def pareto_filter(results: list) -> list:
    if not results:
        return []
    avgs = np.array([r[0] for r in results])
    mins = np.array([r[1] for r in results])
    dominated = np.zeros(len(results), dtype=bool)
    for i in range(len(results)):
        mask = (avgs >= avgs[i]) & (mins >= mins[i]) & ((avgs > avgs[i]) | (mins > mins[i]))
        mask[i] = False
        if mask.any():
            dominated[i] = True
    return [results[i] for i in range(len(results)) if not dominated[i]]

def yby_for_weights(w: dict, strat_names: list, year_matrices: dict) -> list:
    W = np.array([[w.get(n, 0.0) for n in strat_names]])
    yby = []
    for year in YEARS:
        R = year_matrices[year]
        B = (W @ R)[0]
        s = float(B.mean() / B.std() * np.sqrt(8760)) if B.std() > 0 else 0.0
        yby.append(round(s, 3))
    return yby

def pct_range(lo: int, hi: int, step: int):
    return [x / 10000 for x in range(lo, hi + 1, step)]

def blend_and_report(label: str, strat_names: list, year_matrices: dict,
                     i600_levels: list, v1_range=(1250, 3750, 125),
                     id1_range=(500, 2000, 125), id2_range=(1000, 3000, 125),
                     id1_key="idio1", id2_key="idio2",
                     p88_min: float = 1.5287) -> dict:
    """Sweep multiple I600 levels, return best balanced result per level + overall best."""
    best_per_level = {}
    overall_best = None

    for wi600 in i600_levels:
        configs = []
        for wv1 in pct_range(*v1_range):
            for wi1 in pct_range(*id1_range):
                for wi2 in pct_range(*id2_range):
                    wf144 = 1.0 - wv1 - wi1 - wi2 - wi600
                    if 0.15 <= wf144 <= 0.55:
                        configs.append({"v1": wv1, id1_key: wi1, id2_key: wi2,
                                        "i600": wi600, "f144": round(wf144, 8)})

        if not configs:
            print(f"  [{label} I600={wi600*100:.1f}%] No valid configs", flush=True)
            continue

        results = sweep_configs_numpy(configs, strat_names, year_matrices)
        bal_cands = [(a, m, w) for a, m, w in results if a >= 2.000]

        if bal_cands:
            best = max(bal_cands, key=lambda x: (x[1], x[0]))
            a, m, w = best
            yby = yby_for_weights(w, strat_names, year_matrices)
            beats = "★★ DOMINATES P88!" if m > p88_min else (">" if m > 1.4928 else "")
            print(f"  [{label} I600={wi600*100:.1f}%] AVG={a:.4f}, MIN={m:.4f} {beats}", flush=True)
            print(f"    YbY={yby}", flush=True)
            wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w.items()) if v > 0.001)
            print(f"    Weights: {wfmt}", flush=True)
            best_per_level[wi600] = (a, m, w)
            if overall_best is None or m > overall_best[1]:
                overall_best = (a, m, w, wi600)
        else:
            print(f"  [{label} I600={wi600*100:.1f}%] No config reached AVG>=2.000 ({len(configs)} tested)", flush=True)

    return {"per_level": best_per_level, "overall": overall_best}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: Run reference + new signals
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 89: I415_bw216 + I600=5% Fine Grid + Dual-bw216 + Ultra-Fine")
print("=" * 70, flush=True)
print("\nSECTION A: Running reference + new signals", flush=True)

# Reference strategies
v1_data    = run_strategy("p89_v1",           "nexus_alpha_v1",       V1_PARAMS)
i460_bw168 = run_strategy("p89_i460_bw168_k4","idio_momentum_alpha",  make_idio_params(460, 168, k=4))
i460_bw216 = run_strategy("p89_i460_bw216_k4","idio_momentum_alpha",  make_idio_params(460, 216, k=4))
i410_bw216 = run_strategy("p89_i410_bw216_k4","idio_momentum_alpha",  make_idio_params(410, 216, k=4))
i415_bw216 = run_strategy("p89_i415_bw216_k4","idio_momentum_alpha",  make_idio_params(415, 216, k=4))
i474_bw216 = run_strategy("p89_i474_bw216_k4","idio_momentum_alpha",  make_idio_params(474, 216, k=4))
i600_k2    = run_strategy("p89_i600_k2",      "idio_momentum_alpha",  make_idio_params(600, 168, k=2))
fund_144   = run_strategy("p89_fund144",       "funding_momentum_alpha",make_fund_params(144, k=2))

print("\n" + "─"*60)
print("SECTION A SUMMARY:", flush=True)
print(f"  {'Signal':20s}  {'2021':>6}  {'2022':>6}  {'2023':>6}  {'2024':>6}  {'2025':>6}  {'AVG':>6}  {'MIN':>6}", flush=True)
for label, data in [
    ("V1", v1_data),
    ("I460_bw168_k4", i460_bw168), ("I460_bw216_k4", i460_bw216),
    ("I410_bw216_k4", i410_bw216), ("I415_bw216_k4", i415_bw216),
    ("I474_bw216_k4", i474_bw216),
    ("I600_k2", i600_k2), ("F144_k2", fund_144),
]:
    yby = [round(data.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    print(f"  {label:20s}  {yby[0]:>6.3f}  {yby[1]:>6.3f}  {yby[2]:>6.3f}  {yby[3]:>6.3f}  {yby[4]:>6.3f}  {data['_avg']:>6.3f}  {data['_min']:>6.3f}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: I460_bw168 + I415_bw216 + variable I600 (KEY UNTESTED!)
# P88 E2 used I600=7.5%: AVG=2.002, MIN=1.526 — what about I600=5%?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION B: I460_bw168 + I415_bw216 + variable I600 (KEY UNTESTED!)")
print("  P88 E2 at I600=7.5%: AVG=2.002, MIN=1.526")
print("  P88 F (I410_bw216) at I600=5%: AVG=2.015, MIN=1.529 (current champion)")
print("  HYPOTHESIS: I415_bw216 + I600=5% should beat MIN=1.529!")
print("═"*70, flush=True)

STRATS_B = ["v1", "i460bw168", "i415bw216", "i600", "f144"]
base_b = {"v1": v1_data, "i460bw168": i460_bw168, "i415bw216": i415_bw216,
          "i600": i600_k2, "f144": fund_144}
ym_b = build_year_matrices(base_b, STRATS_B)

res_b = blend_and_report(
    "B:I415bw216", STRATS_B, ym_b,
    i600_levels=[0.05, 0.075, 0.10],
    v1_range=(1250, 3750, 125), id1_range=(500, 2000, 125), id2_range=(1000, 3000, 125),
    id1_key="i460bw168", id2_key="i415bw216",
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: I460_bw216 + I415_bw216 + variable I600
# P88 E3 at I600=7.5%: AVG=2.004, MIN=1.502 — test I600=5%
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION C: I460_bw216 + I415_bw216 + variable I600")
print("  P88 E3 at I600=7.5%: AVG=2.004, MIN=1.502")
print("  Both I460 and I415 upgraded to bw=216")
print("═"*70, flush=True)

STRATS_C = ["v1", "i460bw216", "i415bw216", "i600", "f144"]
base_c = {"v1": v1_data, "i460bw216": i460_bw216, "i415bw216": i415_bw216,
          "i600": i600_k2, "f144": fund_144}
ym_c = build_year_matrices(base_c, STRATS_C)

res_c = blend_and_report(
    "C:DualBW216", STRATS_C, ym_c,
    i600_levels=[0.05, 0.075, 0.10],
    v1_range=(1250, 3750, 125), id1_range=(500, 2000, 125), id2_range=(1000, 3000, 125),
    id1_key="i460bw216", id2_key="i415bw216",
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: Triple idio — I460_bw168 + I415_bw216 + I410_bw216
# Both I415 and I410 bw=216 for strong 2022 coverage
# P88 shows: I415_bw216 has 2023=1.143 (bridge), I410_bw216 has 2022=1.928
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION D: Triple idio I460_bw168 + I415_bw216 + I410_bw216")
print("  I415_bw216 covers 2023 (1.143), I410_bw216 covers 2022 (1.928)")
print("  Testing if combining both bw=216 signals beats single-idio configs")
print("═"*70, flush=True)

STRATS_D = ["v1", "i460bw168", "i415bw216", "i410bw216", "i600", "f144"]
base_d = {"v1": v1_data, "i460bw168": i460_bw168, "i415bw216": i415_bw216,
          "i410bw216": i410_bw216, "i600": i600_k2, "f144": fund_144}
ym_d = build_year_matrices(base_d, STRATS_D)

for wi600 in [0.05, 0.075]:
    configs_d = []
    for wv1 in pct_range(1250, 3500, 125):
        for wi460 in pct_range(500, 1750, 125):
            for wi415 in pct_range(750, 2250, 125):
                for wi410 in pct_range(750, 2250, 125):
                    wf144 = 1.0 - wv1 - wi460 - wi415 - wi410 - wi600
                    if 0.15 <= wf144 <= 0.50 and wi415 + wi410 + wi460 <= 0.45:
                        configs_d.append({
                            "v1": wv1, "i460bw168": wi460, "i415bw216": wi415,
                            "i410bw216": wi410, "i600": wi600, "f144": round(wf144, 8)
                        })

    print(f"\n  D: Triple-idio I600={wi600*100:.1f}% ({len(configs_d)} configs)", flush=True)
    results_d = sweep_configs_numpy(configs_d, STRATS_D, ym_d)
    bal_d = [(a, m, w) for a, m, w in results_d if a >= 2.000]
    if bal_d:
        best_d = max(bal_d, key=lambda x: (x[1], x[0]))
        a, m, w = best_d
        yby = yby_for_weights(w, STRATS_D, ym_d)
        beats = "★★ DOMINATES P88!" if m > 1.5287 else "> P87" if m > 1.4928 else ""
        print(f"  Best D [{wi600*100:.1f}%]: AVG={a:.4f}, MIN={m:.4f} {beats}", flush=True)
        print(f"    YbY={yby}", flush=True)
        wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w.items()) if v > 0.001)
        print(f"    Weights: {wfmt}", flush=True)
    else:
        print(f"  No triple-idio config reached AVG>=2.000", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION E: Fine 0.625% grid around best champion from B/C
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION E: Fine 0.625% Grid around Best Champion (Sections B/C)")
print("═"*70, flush=True)

P88_MIN = 1.5287
all_section_bests = []

# Find best across B and C
for sec_label, res_dict, strats, ym in [
    ("B:I415bw216", res_b, STRATS_B, ym_b),
    ("C:DualBW216", res_c, STRATS_C, ym_c),
]:
    if res_dict["overall"] is not None:
        a, m, w, wi600 = res_dict["overall"]
        all_section_bests.append((a, m, w, wi600, sec_label, strats, ym))

if not all_section_bests:
    print("  No balanced result to refine from Sections B/C.", flush=True)
    fine_champion = None
else:
    best_to_refine = max(all_section_bests, key=lambda x: (x[1], x[0]))
    a_c, m_c, w_c, wi600_c, lbl_c, strats_c, ym_c_fine = best_to_refine
    yby_c = yby_for_weights(w_c, strats_c, ym_c_fine)
    print(f"\n  Refining: {lbl_c} AVG={a_c:.4f}/MIN={m_c:.4f}", flush=True)
    print(f"  YbY={yby_c}", flush=True)

    # Build fine grid ±3% around center weights (0.625% step)
    def fine_3weight_configs(center_w, strats, i600_fixed, step=63):
        """0.625% step grid centered on best weights for 3-weight case."""
        moveable = [(k, v) for k, v in center_w.items()
                    if k not in ("i600", "f144")]
        configs = []
        if len(moveable) == 3:
            n1, w1c = moveable[0]
            n2, w2c = moveable[1]
            n3, w3c = moveable[2]
            lo1, hi1 = max(int(w1c*10000)-300, 500), min(int(w1c*10000)+300, 4000)
            lo2, hi2 = max(int(w2c*10000)-300, 250), min(int(w2c*10000)+300, 2500)
            lo3, hi3 = max(int(w3c*10000)-300, 750), min(int(w3c*10000)+300, 3500)
            for wv in pct_range(lo1, hi1, step):
                for wa in pct_range(lo2, hi2, step):
                    for wb in pct_range(lo3, hi3, step):
                        wf = 1.0 - wv - wa - wb - i600_fixed
                        if 0.15 <= wf <= 0.55:
                            configs.append({n1: wv, n2: wa, n3: wb,
                                            "i600": i600_fixed, "f144": round(wf, 8)})
        return configs

    fine_cfgs = fine_3weight_configs(w_c, strats_c, wi600_c, step=63)
    if fine_cfgs:
        print(f"  Fine grid: {len(fine_cfgs)} configs (0.625% step)", flush=True)
        fine_results = sweep_configs_numpy(fine_cfgs, strats_c, ym_c_fine)
        bal_fine = [(a, m, w) for a, m, w in fine_results if a >= 2.000]
        if bal_fine:
            best_fine = max(bal_fine, key=lambda x: (x[1], x[0]))
            a_f, m_f, w_f = best_fine
            yby_f = yby_for_weights(w_f, strats_c, ym_c_fine)
            improved = "IMPROVED!" if m_f > m_c else "no improvement"
            print(f"  Fine result: AVG={a_f:.4f}, MIN={m_f:.4f} ({improved})", flush=True)
            print(f"  YbY={yby_f}", flush=True)
            if m_f > m_c:
                a_c, m_c, w_c = a_f, m_f, w_f
                lbl_c = lbl_c + "_fine"

    fine_champion = (a_c, m_c, w_c, wi600_c, lbl_c, strats_c, ym_c_fine)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION F: Ultra-fine 0.3125% grid around absolute best
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION F: Ultra-fine 0.3125% Grid around Absolute Best")
print("═"*70, flush=True)

ultra_champion = None
if fine_champion is not None:
    a_fc, m_fc, w_fc, wi600_fc, lbl_fc, strats_fc, ym_fc = fine_champion

    def ultra_fine_configs(center_w, i600_fixed, step=31):
        """0.3125% step grid."""
        moveable = [(k, v) for k, v in center_w.items()
                    if k not in ("i600", "f144")]
        configs = []
        if len(moveable) == 3:
            n1, w1c = moveable[0]
            n2, w2c = moveable[1]
            n3, w3c = moveable[2]
            lo1, hi1 = max(int(w1c*10000)-200, 500), min(int(w1c*10000)+200, 4000)
            lo2, hi2 = max(int(w2c*10000)-200, 250), min(int(w2c*10000)+200, 2500)
            lo3, hi3 = max(int(w3c*10000)-200, 750), min(int(w3c*10000)+200, 3500)
            for wv in pct_range(lo1, hi1, step):
                for wa in pct_range(lo2, hi2, step):
                    for wb in pct_range(lo3, hi3, step):
                        wf = 1.0 - wv - wa - wb - i600_fixed
                        if 0.15 <= wf <= 0.55:
                            configs.append({n1: wv, n2: wa, n3: wb,
                                            "i600": i600_fixed, "f144": round(wf, 8)})
        return configs

    ultra_cfgs = ultra_fine_configs(w_fc, wi600_fc, step=31)
    if ultra_cfgs:
        print(f"\n  Ultra-fine grid: {len(ultra_cfgs)} configs (0.3125% step)", flush=True)
        ultra_results = sweep_configs_numpy(ultra_cfgs, strats_fc, ym_fc)
        bal_ultra = [(a, m, w) for a, m, w in ultra_results if a >= 2.000]
        if bal_ultra:
            best_ultra = max(bal_ultra, key=lambda x: (x[1], x[0]))
            a_u, m_u, w_u = best_ultra
            yby_u = yby_for_weights(w_u, strats_fc, ym_fc)
            improved = "IMPROVED!" if m_u > m_fc else "converged"
            print(f"  Ultra result: AVG={a_u:.4f}, MIN={m_u:.4f} ({improved})", flush=True)
            print(f"  YbY={yby_u}", flush=True)
            if m_u > m_fc:
                a_fc, m_fc, w_fc = a_u, m_u, w_u
                lbl_fc = lbl_fc + "_ultrafine"
            ultra_champion = (a_fc, m_fc, w_fc, wi600_fc, lbl_fc, strats_fc, ym_fc)
        else:
            ultra_champion = fine_champion
    else:
        ultra_champion = fine_champion
else:
    ultra_champion = None

# ─────────────────────────────────────────────────────────────────────────────
# SECTION G: AVG-max push — I474_bw216 + I415_bw216 combination
# I474_bw216 might replace I437 in the AVG-max config (higher return years)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION G: AVG-max Push — I474_bw216 + I415_bw216 or I460_bw216")
print("  I474 standalone had AVG=1.880. At bw=216, could be even higher.")
print("  Testing if I474_bw216 improves on P86 AVG-max (2.268/1.125)")
print("═"*70, flush=True)

# G1: I474_bw216 as high-AVG signal
i474_avg = i474_bw216.get("_avg", 0)
i474_yby = [round(i474_bw216.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
print(f"\n  I474_bw216 standalone: AVG={i474_avg:.3f}, YbY={i474_yby}", flush=True)

# Compare with known P86 AVG-max combo: I437_bw168 + I474_bw168
# Test: V1 + I474_bw216 + I415_bw216 (replacing I437 + I474 from P86 AVG-max)
STRATS_G1 = ["v1", "i474bw216", "i415bw216", "i600", "f144"]
base_g1 = {"v1": v1_data, "i474bw216": i474_bw216, "i415bw216": i415_bw216,
            "i600": i600_k2, "f144": fund_144}
ym_g1 = build_year_matrices(base_g1, STRATS_G1)

configs_g1 = []
for wv1 in pct_range(0, 2500, 125):        # 0 to 25%
    for wi474 in pct_range(1000, 3500, 125):  # 10 to 35%
        for wi415 in pct_range(1000, 3500, 125):  # 10 to 35%
            wi600 = 0.10
            wf144 = 1.0 - wv1 - wi474 - wi415 - wi600
            if 0.25 <= wf144 <= 0.55:
                configs_g1.append({"v1": wv1, "i474bw216": wi474, "i415bw216": wi415,
                                    "i600": wi600, "f144": round(wf144, 8)})

print(f"\n  G1: I474_bw216 + I415_bw216 AVG-max grid ({len(configs_g1)} configs)", flush=True)
results_g1 = sweep_configs_numpy(configs_g1, STRATS_G1, ym_g1)

# Find AVG-max (highest AVG regardless of MIN)
if results_g1:
    best_g1_avg = max(results_g1, key=lambda x: x[0])
    a_g1, m_g1, w_g1 = best_g1_avg
    yby_g1 = yby_for_weights(w_g1, STRATS_G1, ym_g1)
    print(f"  G1 AVG-max: AVG={a_g1:.4f}, MIN={m_g1:.4f}", flush=True)
    print(f"  YbY={yby_g1}", flush=True)
    wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_g1.items()) if v > 0.001)
    print(f"  Weights: {wfmt}", flush=True)
    if a_g1 > 2.268:
        print(f"  ★★ NEW AVG-MAX RECORD! (AVG {a_g1:.4f} > P86's 2.268)", flush=True)
    # Also find balanced (AVG>=2.0) in G1
    bal_g1 = [(a, m, w) for a, m, w in results_g1 if a >= 2.000]
    if bal_g1:
        best_g1_bal = max(bal_g1, key=lambda x: (x[1], x[0]))
        a_g1b, m_g1b, w_g1b = best_g1_bal
        yby_g1b = yby_for_weights(w_g1b, STRATS_G1, ym_g1)
        print(f"  G1 balanced: AVG={a_g1b:.4f}, MIN={m_g1b:.4f}, YbY={yby_g1b}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION H: Pareto summary + champion save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION H: Pareto Summary + Champion Save")
print("═"*70, flush=True)

# Determine overall champion
champion_candidates = []
if ultra_champion is not None:
    a, m, w, wi600, lbl, strats, ym = ultra_champion
    if a >= 2.000:
        champion_candidates.append((a, m, w, wi600, lbl, strats, ym))

# Also check Section B/C raw results at I600=5%
for sec_label, res_dict, strats, ym in [
    ("B:I415bw216", res_b, STRATS_B, ym_b),
    ("C:DualBW216", res_c, STRATS_C, ym_c),
]:
    if res_dict["overall"] is not None:
        a, m, w, wi600 = res_dict["overall"]
        if a >= 2.000:
            champion_candidates.append((a, m, w, wi600, sec_label, strats, ym))

if champion_candidates:
    overall_best = max(champion_candidates, key=lambda x: (x[1], x[0]))
    a_best, m_best, w_best, wi600_best, lbl_best, strats_best, ym_best = overall_best
    yby_best = yby_for_weights(w_best, strats_best, ym_best)

    print(f"\n★★★ PHASE 89 BALANCED CHAMPION [{lbl_best}]:", flush=True)
    print(f"    AVG={a_best:.4f}, MIN={m_best:.4f}", flush=True)
    print(f"    YbY={yby_best}", flush=True)
    wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_best.items()) if v > 0.001)
    print(f"    Weights: {wfmt}", flush=True)

    if m_best > P88_MIN:
        print(f"    ★★ STRICTLY DOMINATES P88 balanced (MIN {m_best:.4f} > {P88_MIN:.4f})!", flush=True)

    # Save champion config
    champion_config = {
        "phase": 89,
        "section": lbl_best,
        "avg_sharpe": round(a_best, 4),
        "min_sharpe": round(m_best, 4),
        "yby_sharpes": {y: round(v, 3) for y, v in zip(YEARS, yby_best)},
        "weights": {k: round(v, 6) for k, v in w_best.items() if v > 0.001},
        "signal_config": {
            "strat_names": strats_best,
            "notes": {
                "I460_bw168_k4": "idio_momentum_alpha lb=460 bw=168 k=4",
                "I460_bw216_k4": "idio_momentum_alpha lb=460 bw=216 k=4",
                "I410_bw216_k4": "idio_momentum_alpha lb=410 bw=216 k=4",
                "I415_bw216_k4": "idio_momentum_alpha lb=415 bw=216 k=4",
                "I474_bw216_k4": "idio_momentum_alpha lb=474 bw=216 k=4",
                "I600_k2":       "idio_momentum_alpha lb=600 bw=168 k=2",
                "F144_k2":       "funding_momentum_alpha lb=144 k=2",
                "V1":            "nexus_alpha_v1 k=2",
            }
        }
    }
    out_path = "configs/ensemble_p89_balanced.json"
    with open(out_path, "w") as f:
        json.dump(champion_config, f, indent=2)
    print(f"\n    Saved: {out_path}", flush=True)

    # Pareto summary
    all_pareto = [
        (2.268, 1.125, "P86-AVG-MAX"),
        (2.015, 1.529, "P88-CHAMP"),
        (a_best, m_best, f"P89-{lbl_best}"),
    ]
    for sec_label, res_dict, strats, ym in [
        ("P89-B-I415bw216", res_b, STRATS_B, ym_b),
        ("P89-C-DualBW216", res_c, STRATS_C, ym_c),
    ]:
        if res_dict["overall"] is not None:
            a, m, w, _ = res_dict["overall"]
            all_pareto.append((a, m, sec_label))

    # Simple Pareto filter
    pareto = []
    for i, (a, m, lbl) in enumerate(all_pareto):
        dominated = any(
            a2 >= a and m2 >= m and (a2 > a or m2 > m)
            for j, (a2, m2, lbl2) in enumerate(all_pareto) if j != i
        )
        if not dominated:
            pareto.append((a, m, lbl))

    pareto.sort(key=lambda x: -x[0])
    print("\nFINAL PARETO FRONTIER:", flush=True)
    print(f"  {'AVG':>6}  {'MIN':>6}  Config", flush=True)
    for a, m, lbl in pareto:
        print(f"  {a:>6.4f}  {m:>6.4f}  {lbl}", flush=True)

else:
    print("  No valid balanced champion found in Phase 89.", flush=True)
    print("  P88 champion (2.015/1.529) remains current best.", flush=True)

print("\nPHASE 89 COMPLETE", flush=True)
