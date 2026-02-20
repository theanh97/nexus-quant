#!/usr/bin/env python3
"""
Phase 63: Extended Lookback Discovery + Corrected Parameter Search

Phase 62 revealed: optimal lookback is NOT 336h — it's at 437h+ for SR Alpha.
The SR Alpha sensitivity showed monotonically increasing performance with longer lookbacks
up to 437h (the edge of what was tested).

This phase:
  A. Extended lookback sweep: 336h → 720h (3h increments around candidates)
  B. Test true best parameters with proper search (avoid re-testing noise)
  C. Re-evaluate ensemble with corrected optima

First-principles basis:
  - Momentum lookback = information diffusion horizon
  - Crypto: slower institutional adoption → longer diffusion than equities?
  - Traditional equity momentum: 12 months (252 trading days) is canonical
  - 336h = 2 weeks, 504h = 3 weeks, 720h = 1 month
  - Academic consensus: skip 1 month, use 12-month window → avoid reversal
  - For 10-coin crypto universe: smaller universe → need longer signal for significance

Hypothesis: optimal is ~3-4 weeks (504-720h) based on:
  1. Crypto market structure: retail-dominated → slower price discovery
  2. Institutional rebalancing: quarterly/monthly cycles create 3-4 week signals
  3. Funding rate cycles: 8h funding creates ~1 week autocorrelation structure
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase63"
YEARS = ["2021", "2022", "2023", "2024", "2025"]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

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
    "data": {"provider": "binance_rest_v1", "symbols": SYMBOLS, "bar_interval": "1h", "cache_dir": ".cache/binance_rest"},
    "execution": {"style": "taker", "slippage_bps": 3.0},
    "costs": {"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0},
    "benchmark": {"version": "v1", "walk_forward": {"enabled": True}},
    "risk": {"max_drawdown": 0.30, "max_turnover_per_rebalance": 0.8, "max_gross_leverage": 0.7, "max_position_per_symbol": 0.3},
    "self_learn": {"enabled": False},
}

SR_BASE = {
    "k_per_side": 2,
    "vol_lookback_bars": 168,
    "target_gross_leverage": 0.35,
    "rebalance_interval_bars": 48,
}

IDIO_BASE = {
    "k_per_side": 2,
    "beta_window_bars": 72,
    "vol_lookback_bars": 168,
    "target_gross_leverage": 0.30,
    "rebalance_interval_bars": 48,
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


def compute_metrics(result_path: str) -> dict:
    d = json.load(open(result_path))
    eq = d.get("equity_curve", [])
    rets = d.get("returns", [])
    if not eq or not rets or len(rets) < 100:
        return {"sharpe": 0, "error": "insufficient data"}
    mu = statistics.mean(rets)
    sd = statistics.pstdev(rets)
    sharpe = (mu / sd) * math.sqrt(8760) if sd > 0 else 0
    return {"sharpe": round(sharpe, 3)}


def make_config(run_name: str, year: str, strategy_name: str, params: dict) -> str:
    cfg = copy.deepcopy(BASE_CONFIG)
    start, end = YEAR_RANGES[year]
    cfg["run_name"] = run_name
    cfg["data"]["start"] = start
    cfg["data"]["end"] = end
    cfg["strategy"] = {"name": strategy_name, "params": params}
    path = f"/tmp/phase63_{run_name}_{year}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def run_variant(name: str, strategy_name: str, params: dict) -> dict:
    year_results = {}
    for year in YEARS:
        run_name = f"{name}_{year}"
        config_path = make_config(run_name, year, strategy_name, params)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "nexus_quant", "run", "--config", config_path, "--out", OUT_DIR],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                year_results[year] = {"error": "run failed"}
                print(f"    {year}: ERROR", flush=True)
                continue
        except subprocess.TimeoutExpired:
            year_results[year] = {"error": "timeout"}
            continue

        runs_dir = Path(OUT_DIR) / "runs"
        if not runs_dir.exists():
            year_results[year] = {"error": "no runs dir"}
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
                print(f"    {year}: Sharpe={m.get('sharpe', '?')}", flush=True)
                continue
        year_results[year] = {"error": "no result"}

    sharpes = [y.get("sharpe", 0) for y in year_results.values() if isinstance(y.get("sharpe"), (int, float))]
    avg = round(sum(sharpes) / len(sharpes), 3) if sharpes else 0
    mn = round(min(sharpes), 3) if sharpes else 0
    pos = sum(1 for s in sharpes if s > 0)
    year_results["_avg_sharpe"] = avg
    year_results["_min_sharpe"] = mn
    year_results["_pos"] = pos
    return year_results


def main():
    print("=" * 80)
    print("  PHASE 63: EXTENDED LOOKBACK DISCOVERY")
    print("=" * 80)
    print("  Phase 62 revealed: optimal SR lookback is beyond 336h (performance")
    print("  monotonically increases up to 437h — at the edge of search space)")
    print("")
    print("  First-principles hypothesis (crypto-specific):")
    print("  - Traditional equity: 12-month momentum canonical (information diffusion)")
    print("  - Crypto: retail-dominated, slower discovery → may need longer lookbacks")
    print("  - 720h = 1 month ≈ 30 trading days (equities: canonical momentum skip period)")
    print("  - Funding rate structure (8h) creates weekly autocorrelation → 504-720h")
    print("=" * 80, flush=True)

    all_results = {}
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # ─── Part A: SR Alpha extended lookback ───
    print("\n" + "=" * 70)
    print("  A: SHARPE RATIO ALPHA — EXTENDED LOOKBACK (336h → 1008h)")
    print("=" * 70, flush=True)

    sr_lookbacks = [
        # The Phase 62 sensitivity tested up to 437h and it was still rising
        # Now extend the search into the full "monthly momentum" territory
        ("sr_336",  336),   # Previous champion
        ("sr_403",  403),   # +20%
        ("sr_437",  437),   # +30% (best in Phase 62)
        ("sr_504",  504),   # 3 weeks — crypto-canonical?
        ("sr_600",  600),   # ~3.5 weeks
        ("sr_672",  672),   # 4 weeks
        ("sr_720",  720),   # 1 month
        ("sr_840",  840),   # 5 weeks
        ("sr_1008", 1008),  # 6 weeks (42 days)
    ]

    sr_results = {}
    for name, lb in sr_lookbacks:
        params = {**SR_BASE, "lookback_bars": lb}
        print(f"\n  >> {name} (lookback={lb}h = {lb/24:.1f}d)", flush=True)
        r = run_variant(name, "sharpe_ratio_alpha", params)
        sr_results[name] = r
        yby = [r.get(y, {}).get("sharpe", "?") for y in YEARS]
        print(f"     AVG={r['_avg_sharpe']}, MIN={r['_min_sharpe']}, pos={r['_pos']}/5 | {yby}", flush=True)
    all_results["sr_alpha"] = sr_results

    # ─── Part B: Idio Momentum extended lookback ───
    print("\n" + "=" * 70)
    print("  B: IDIO MOMENTUM ALPHA — EXTENDED LOOKBACK (336h → 1008h)")
    print("=" * 70, flush=True)

    # Phase 62 showed idio_lookback is also a SPIKE — needs full extended search
    idio_lookbacks = [
        ("idio_336",  336),
        ("idio_403",  403),
        ("idio_437",  437),
        ("idio_504",  504),
        ("idio_600",  600),
        ("idio_672",  672),
        ("idio_720",  720),
        ("idio_840",  840),
        ("idio_1008", 1008),
    ]

    idio_results = {}
    for name, lb in idio_lookbacks:
        params = {**IDIO_BASE, "lookback_bars": lb}
        print(f"\n  >> {name} (lookback={lb}h = {lb/24:.1f}d)", flush=True)
        r = run_variant(name, "idio_momentum_alpha", params)
        idio_results[name] = r
        yby = [r.get(y, {}).get("sharpe", "?") for y in YEARS]
        print(f"     AVG={r['_avg_sharpe']}, MIN={r['_min_sharpe']}, pos={r['_pos']}/5 | {yby}", flush=True)
    all_results["idio_momentum"] = idio_results

    # ─── Summary ───
    print("\n" + "=" * 80)
    print("  SR ALPHA LOOKBACK SWEEP — RANKED BY AVG SHARPE")
    print("=" * 80)
    sr_ranked = sorted(sr_results.items(), key=lambda x: x[1]["_avg_sharpe"], reverse=True)
    for name, r in sr_ranked:
        lb = int(name.split("_")[1])
        yby = [r.get(y, {}).get("sharpe", "?") for y in YEARS]
        print(f"  {name:<12} (lb={lb:>4}h={lb/24:>4.1f}d): AVG={r['_avg_sharpe']:.3f} MIN={r['_min_sharpe']:.3f} pos={r['_pos']}/5 | {yby}")

    print("\n" + "=" * 80)
    print("  IDIO MOMENTUM LOOKBACK SWEEP — RANKED BY AVG SHARPE")
    print("=" * 80)
    idio_ranked = sorted(idio_results.items(), key=lambda x: x[1]["_avg_sharpe"], reverse=True)
    for name, r in idio_ranked:
        lb = int(name.split("_")[1])
        yby = [r.get(y, {}).get("sharpe", "?") for y in YEARS]
        print(f"  {name:<12} (lb={lb:>4}h={lb/24:>4.1f}d): AVG={r['_avg_sharpe']:.3f} MIN={r['_min_sharpe']:.3f} pos={r['_pos']}/5 | {yby}")

    # ─── Find new optima and run ensemble ───
    best_sr = sr_ranked[0]
    best_idio = idio_ranked[0]
    print(f"\n  New best SR: {best_sr[0]} AVG={best_sr[1]['_avg_sharpe']}")
    print(f"  New best Idio: {best_idio[0]} AVG={best_idio[1]['_avg_sharpe']}")

    # ─── Ensemble with new optima ───
    print("\n" + "=" * 80)
    print("  C: ENSEMBLE ANALYSIS WITH NEW OPTIMA")
    print("=" * 80, flush=True)

    # Extract lookbacks from best variant names
    sr_best_lb = int(best_sr[0].split("_")[1])
    idio_best_lb = int(best_idio[0].split("_")[1])

    sr_best_params = {**SR_BASE, "lookback_bars": sr_best_lb}
    idio_best_params = {**IDIO_BASE, "lookback_bars": idio_best_lb}

    # Collect per-year returns for V1, SR_best, Idio_best
    v1_sharpes = {}
    sr_sharpes = {y: best_sr[1].get(y, {}).get("sharpe") for y in YEARS}
    idio_sharpes = {y: best_idio[1].get(y, {}).get("sharpe") for y in YEARS}

    # Run V1 if needed (use cached results from Phase 62)
    print("\n  Running V1-Long (for fresh ensemble baseline):", flush=True)
    v1_results = run_variant("v1long_p63", "nexus_alpha_v1", V1_PARAMS)
    v1_sharpes = {y: v1_results.get(y, {}).get("sharpe") for y in YEARS}
    all_results["v1_base"] = v1_results

    # Simulate ensemble combinations (returns-level blend approximation)
    print("\n  Ensemble weights sweep:", flush=True)

    def blend_sharpe(yr_v1, yr_sr, yr_idio, w_v1, w_sr, w_idio):
        """Approximate ensemble Sharpe by blending year-by-year Sharpe ratios
        (valid approximation when correlations are moderate)."""
        blended_sharpes = []
        for y in YEARS:
            s_v1 = yr_v1.get(y)
            s_sr = yr_sr.get(y)
            s_idio = yr_idio.get(y)
            if s_v1 is None or s_sr is None or s_idio is None:
                continue
            blended = w_v1 * s_v1 + w_sr * s_sr + w_idio * s_idio
            blended_sharpes.append(blended)
        if not blended_sharpes:
            return 0, 0, 0
        avg = sum(blended_sharpes) / len(blended_sharpes)
        mn = min(blended_sharpes)
        pos = sum(1 for s in blended_sharpes if s > 0)
        return round(avg, 3), round(mn, 3), pos

    best_ens = {"key": None, "avg": 0, "mn": 0}
    ens_results = []
    for w1 in [1.0, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
        for w2 in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]:
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0:
                continue
            avg, mn, pos = blend_sharpe(v1_sharpes, sr_sharpes, idio_sharpes, w1, w2, w3)
            ens_results.append((w1, w2, w3, avg, mn, pos))
            if avg > best_ens["avg"] and pos == 5:
                best_ens = {"key": (w1, w2, w3), "avg": avg, "mn": mn, "pos": pos}

    # Show top 12 by avg
    ens_results.sort(key=lambda x: x[3], reverse=True)
    print(f"\n  V1/{best_sr[0]}/{best_idio[0]} ensemble sweep:")
    print(f"  {'V1':>6} {'SR':>6} {'Idio':>6} {'AVG':>7} {'MIN':>7} {'Pos':>4}")
    for w1, w2, w3, avg, mn, pos in ens_results[:15]:
        mark = " ★" if pos == 5 and avg > 1.2 else ""
        print(f"  {w1:>6.2f} {w2:>6.2f} {w3:>6.2f} {avg:>7.3f} {mn:>7.3f} {pos:>4}{mark}")

    # ─── Final Summary ───
    print("\n" + "=" * 80)
    print("  PHASE 63 FINAL SUMMARY")
    print("=" * 80)
    print(f"\n  PHASE 62 OPTIMA (for reference):")
    print(f"    SR Alpha:  lb=336h  AVG=0.846")
    print(f"    IdioMom:   lb=336h  AVG=1.039")
    print(f"\n  PHASE 63 NEW OPTIMA:")
    print(f"    SR Alpha:  {best_sr[0]}  AVG={best_sr[1]['_avg_sharpe']}")
    print(f"    IdioMom:   {best_idio[0]}  AVG={best_idio[1]['_avg_sharpe']}")

    if best_ens["key"]:
        w1, w2, w3 = best_ens["key"]
        print(f"\n  NEW CHAMPION ENSEMBLE (V1={w1} / SR={w2} / Idio={w3}):")
        print(f"    AVG={best_ens['avg']}, MIN={best_ens['mn']}, Pos={best_ens['pos']}/5")
    else:
        print(f"\n  Best ensemble: V1={best_ens}")

    # Save
    out_path = Path(OUT_DIR) / "phase63_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
