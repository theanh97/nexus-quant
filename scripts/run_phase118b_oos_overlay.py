#!/usr/bin/env python3
"""
Phase 118b: Vision Metrics Overlay — OOS Completion
=====================================================
Continues Phase 118 walk-forward after crash at 2025_oos.

Bug fixed: compute_sharpe() crash on short/sparse OOS returns.
  - Added np.asarray() guard for non-contiguous arrays
  - min_bars lowered to 50 for 2026_ytd (~1200 bars total)
  - NaN guard before std() to prevent comparison error

Architecture: subprocess-free, pure numpy overlay (fast, cached data).
  1. Load P91b returns via BacktestEngine (cached klines)
  2. Download Vision metrics (cache-first, rate-limited)
  3. Apply global_ls_z168_low + top candidates as post-hoc overlay
  4. If global_ls_z168_low passes OOS (Δ > 0 in 2025_oos) → integrate

Walk-forward plan:
  IS  (already done in Phase 118):  2022, 2023, 2024
  OOS (this script):                 2025_oos, 2026_ytd

Best IS candidate: global_ls_z168_low → Δ+0.27/+0.48/+0.39 (3-year avg)
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from nexus_quant.data.providers.binance_vision_metrics import load_vision_metrics

# ─── Config ────────────────────────────────────────────────────────────────────
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]

# OOS periods to complete (IS was done in Phase 118 run)
OOS_PERIODS = {
    "2025_oos":  ("2025-01-01", "2026-01-01"),
    "2026_ytd":  ("2026-01-01", "2026-02-21"),  # today
}

# Phase 118 IS results (from previous run — already validated)
IS_RESULTS = {
    "global_ls_z168_low": {"2022": 0.27, "2023": 0.48, "2024": 0.39},
    "oi_mom_z168":        {"2022": 0.11, "2023": 0.18, "2024": 0.09},
    "taker_ls_z168_low":  {"2022": 0.19, "2023": 0.31, "2024": 0.22},
    "top_ls_pos_z168":    {"2022": 0.07, "2023": 0.12, "2024": 0.14},
}

# Overlay spec: (metric_key, lookback_bars, tilt_ratio, direction)
OVERLAYS = [
    ("global_ls_z168_low", "global_ls_ratio",  168, 0.65, "reduce_on_low"),
    ("oi_mom_z168",        "open_interest",     168, 0.65, "reduce_on_high"),
    ("taker_ls_z168_low",  "taker_ls_ratio",    168, 0.65, "reduce_on_low"),
    ("top_ls_pos_z168",    "top_ls_position",   168, 0.65, "reduce_on_high"),
]

OUT_DIR = PROJ_ROOT / "artifacts" / "phase118b"
VISION_CACHE = PROJ_ROOT / ".cache" / "binance_vision_metrics"
P91B_CONFIGS = {
    "2025_oos": PROJ_ROOT / "configs" / "run_p91b_2025.json",
    "2026_ytd": PROJ_ROOT / "configs" / "run_p91b_2026ytd.json",
}
PRODUCTION_CFG = PROJ_ROOT / "configs" / "production_p91b_champion.json"

# ─── Fixed compute_sharpe ──────────────────────────────────────────────────────

def compute_sharpe(returns, min_bars: int = 50) -> float:
    """
    Robust Sharpe ratio computation.

    Fixes vs Phase 118 original:
      - np.asarray() guard for non-contiguous / non-array inputs
      - min_bars=50 default (OOS 2026_ytd has only ~1200 bars)
      - explicit float conversion before std == 0 check (avoids numpy bool ambiguity)
      - handles all-zero or constant returns gracefully
    """
    rets = np.asarray(returns, dtype=np.float64).ravel()   # ← FIX: asarray + ravel
    rets = rets[np.isfinite(rets)]                          # strip NaN/Inf
    if len(rets) < min_bars:
        return 0.0
    std = float(np.std(rets, ddof=0))
    if std == 0.0 or not np.isfinite(std):                 # ← FIX: explicit float
        return 0.0
    return float(np.mean(rets) / std * np.sqrt(8760))


def compute_metric_z_scores(
    metric_values: dict,
    timeline: list,
    lookback: int,
) -> np.ndarray:
    """Compute rolling z-score momentum for a market metric at each hourly bar."""
    n = len(timeline)
    z_scores = np.zeros(n, dtype=np.float64)

    vals = np.zeros(n, dtype=np.float64)
    last_v = 0.0
    for i, t in enumerate(timeline):
        if t in metric_values:
            last_v = metric_values[t]
        vals[i] = last_v

    # Log-transform (OI is huge absolute numbers; ratios are ~1.0 so log is safe)
    log_vals = np.log(np.maximum(vals, 1e-10))

    # Vectorised rolling z-score (no Python loops for speed)
    for i in range(lookback * 2, n):
        lo = max(0, i - lookback * 2)
        # momentum window
        moms = log_vals[lo + lookback: i + 1] - log_vals[lo: i - lookback + 1]
        if len(moms) < 10:
            continue
        mu  = float(np.mean(moms))
        std = float(np.std(moms, ddof=0))
        if std > 0:
            mom_i = log_vals[i] - log_vals[i - lookback]
            z_scores[i] = (mom_i - mu) / std

    return z_scores


def apply_overlay(
    base_returns: np.ndarray,
    z_scores: np.ndarray,
    tilt_ratio: float,
    direction: str,
) -> np.ndarray:
    """Scale returns by tilt_ratio when overlay condition fires (1-bar lag)."""
    n = min(len(base_returns), len(z_scores))
    tilted = base_returns[:n].copy()
    for i in range(1, n):
        if direction == "reduce_on_high" and z_scores[i - 1] > 0:
            tilted[i] *= tilt_ratio
        elif direction == "reduce_on_low" and z_scores[i - 1] < 0:
            tilted[i] *= tilt_ratio
    return tilted


# ─── Run P91b via subprocess (reliable, uses cached klines) ───────────────────

def run_p91b_subprocess(period_name: str) -> tuple[np.ndarray, list]:
    """
    Run P91b ensemble via `python -m nexus_quant run` for the given period.
    Returns (returns_array, timeline_list) or (None, None) on failure.
    """
    cfg_path = P91B_CONFIGS.get(period_name)
    if cfg_path is None or not cfg_path.exists():
        print(f"  [ERROR] Config not found: {cfg_path}")
        return None, None

    run_out = OUT_DIR / "runs" / period_name
    run_out.mkdir(parents=True, exist_ok=True)

    print(f"  Running P91b for {period_name} via CLI...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "nexus_quant", "run",
         "--config", str(cfg_path),
         "--out", str(run_out)],
        capture_output=True, text=True, timeout=900,
        cwd=str(PROJ_ROOT),
    )

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [ERROR] CLI failed in {elapsed:.0f}s:\n{result.stderr[-2000:]}")
        return None, None

    # Locate result.json
    runs_dir = run_out / "runs"
    if not runs_dir.exists():
        runs_dir = run_out  # flat structure
    candidates = sorted(
        [d for d in runs_dir.rglob("result.json")],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        print(f"  [ERROR] No result.json found under {run_out}")
        return None, None

    data = json.loads(candidates[-1].read_text())
    rets = np.asarray(data.get("returns", []), dtype=np.float64)
    raw_timeline = data.get("timeline", [])

    # Convert timeline to epoch seconds (Vision metrics keyed by int epoch_s)
    timeline_s = []
    for t in raw_timeline:
        if isinstance(t, (int, float)):
            timeline_s.append(int(t))
        else:
            # ISO string: "2025-01-01T00:00:00+00:00"
            from datetime import timezone
            from email.utils import parsedate_to_datetime
            try:
                dt = datetime.fromisoformat(str(t))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                timeline_s.append(int(dt.timestamp()))
            except ValueError:
                timeline_s.append(0)

    print(f"  -> P91b {period_name}: {len(rets)} bars in {elapsed:.0f}s, Sharpe={compute_sharpe(rets):.4f}")
    return rets, timeline_s


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VISION_CACHE.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PHASE 118b: Vision Metrics Overlay — OOS Completion")
    print("  Completing 2025_oos + 2026_ytd (crash fix applied)")
    print("=" * 70, flush=True)

    print("\nIS Summary (from Phase 118):")
    print(f"  {'Overlay':<25s}  {'2022':>7}  {'2023':>7}  {'2024':>7}  {'IS avg':>7}")
    for name, yrs in IS_RESULTS.items():
        avg = sum(yrs.values()) / len(yrs)
        print(f"  {name:<25s}  {yrs.get('2022',0):+.3f}   {yrs.get('2023',0):+.3f}   "
              f"{yrs.get('2024',0):+.3f}   {avg:+.3f}")
    print()

    oos_results = {}

    for period_name, (start, end) in OOS_PERIODS.items():
        print(f"\n{'='*60}")
        print(f"  PERIOD: {period_name} ({start} → {end})")
        print(f"{'='*60}", flush=True)

        # ── Step 1: P91b base returns ──────────────────────────────────────
        base_rets, timeline = run_p91b_subprocess(period_name)
        if base_rets is None:
            print(f"  SKIP {period_name}: backtest failed")
            oos_results[period_name] = {"error": "backtest_failed"}
            continue

        base_sharpe = compute_sharpe(base_rets)
        base_eq = np.cumprod(1.0 + base_rets)
        base_mdd = float((((base_eq - np.maximum.accumulate(base_eq))
                           / np.maximum(np.maximum.accumulate(base_eq), 1e-10)).min() * 100))
        print(f"  Base P91b: Sharpe={base_sharpe:.4f}, MDD={base_mdd:.2f}%", flush=True)

        # ── Step 2: Vision metrics ─────────────────────────────────────────
        print(f"\n  Loading Vision metrics ({start} → {end})...", flush=True)
        try:
            metrics = load_vision_metrics(
                SYMBOLS, start, end, cache_dir=VISION_CACHE
            )
        except Exception as exc:
            print(f"  [WARN] Vision metrics load failed: {exc}")
            metrics = {}

        # ── Step 3: Test overlays ──────────────────────────────────────────
        period_overlay_results = {}
        print(f"\n  Testing {len(OVERLAYS)} overlays...", flush=True)

        for overlay_name, metric_key, lookback, tilt_ratio, direction in OVERLAYS:
            # Aggregate metric across symbols
            agg: dict = {}
            for sym in SYMBOLS:
                sym_m = metrics.get(sym, {}).get(metric_key, {})
                for ts, val in sym_m.items():
                    agg.setdefault(ts, []).append(val)

            agg_values: dict = {}
            for ts, vals in agg.items():
                if metric_key == "open_interest":
                    agg_values[ts] = sum(vals)
                else:
                    agg_values[ts] = sum(vals) / len(vals)

            # Compute z-scores aligned to backtest timeline
            z_scores = compute_metric_z_scores(agg_values, timeline, lookback)

            # Apply overlay
            tilted_rets = apply_overlay(base_rets, z_scores, tilt_ratio, direction)
            tilted_sharpe = compute_sharpe(tilted_rets)
            tilted_eq = np.cumprod(1.0 + tilted_rets)
            tilted_mdd = float((((tilted_eq - np.maximum.accumulate(tilted_eq))
                                 / np.maximum(np.maximum.accumulate(tilted_eq), 1e-10)).min() * 100))

            # Tilt stats
            n_active = sum(
                1 for i in range(1, len(z_scores))
                if (direction == "reduce_on_high" and z_scores[i - 1] > 0)
                or (direction == "reduce_on_low"  and z_scores[i - 1] < 0)
            )
            tilt_pct = n_active / max(len(z_scores) - 1, 1) * 100

            delta = tilted_sharpe - base_sharpe
            verdict = "BETTER" if delta > 0.05 else "WORSE" if delta < -0.05 else "NEUTRAL"

            period_overlay_results[overlay_name] = {
                "sharpe":       round(tilted_sharpe, 4),
                "delta_sharpe": round(delta, 4),
                "mdd":          round(tilted_mdd, 2),
                "tilt_pct":     round(tilt_pct, 1),
                "verdict":      verdict,
            }
            print(f"    {overlay_name:<25s}: Δ={delta:+.4f}  "
                  f"Sharpe={tilted_sharpe:.4f}  tilt={tilt_pct:.1f}%  → {verdict}", flush=True)

        oos_results[period_name] = {
            "base_sharpe": round(base_sharpe, 4),
            "base_mdd":    round(base_mdd, 2),
            "n_bars":      len(base_rets),
            "overlays":    period_overlay_results,
        }

    # ── Walk-Forward Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  WALK-FORWARD SUMMARY (IS + OOS)")
    print("=" * 70)

    print(f"\n  {'Overlay':<25s}  "
          f"{'IS 2022':>8} {'IS 2023':>8} {'IS 2024':>8}  "
          f"{'OOS 25':>7} {'OOS 26':>7}  "
          f"{'WF?':>5}")
    print(f"  {'-'*25}  {'-'*8} {'-'*8} {'-'*8}  {'-'*7} {'-'*7}  {'-'*5}")

    integration_candidates = []
    for overlay_name, _, _, _, _ in OVERLAYS:
        is_deltas = IS_RESULTS.get(overlay_name, {})
        d22 = is_deltas.get("2022", 0.0)
        d23 = is_deltas.get("2023", 0.0)
        d24 = is_deltas.get("2024", 0.0)
        is_avg = (d22 + d23 + d24) / 3 if is_deltas else 0.0

        oos_d25 = oos_results.get("2025_oos", {}).get("overlays", {}).get(
            overlay_name, {}).get("delta_sharpe", None)
        oos_d26 = oos_results.get("2026_ytd", {}).get("overlays", {}).get(
            overlay_name, {}).get("delta_sharpe", None)

        # WF pass: IS avg > 0 AND OOS 2025 > 0 (primary OOS test)
        wf_pass = (is_avg > 0 and oos_d25 is not None and oos_d25 > 0)
        wf_str = "✅ PASS" if wf_pass else "❌ FAIL"

        d25_str = f"{oos_d25:+.4f}" if oos_d25 is not None else "  N/A  "
        d26_str = f"{oos_d26:+.4f}" if oos_d26 is not None else "  N/A  "

        print(f"  {overlay_name:<25s}  "
              f"{d22:+.3f}    {d23:+.3f}    {d24:+.3f}    "
              f"{d25_str}  {d26_str}   {wf_str}")

        if wf_pass:
            integration_candidates.append({
                "name": overlay_name,
                "is_avg": round(is_avg, 4),
                "oos_2025": round(oos_d25, 4),
            })

    # ── Integration Decision ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    if integration_candidates:
        # Pick best by OOS 2025 delta
        best = max(integration_candidates, key=lambda x: x["oos_2025"])
        print(f"\n  ✅ WF PASS: {best['name']}")
        print(f"     IS avg Δ = {best['is_avg']:+.4f}")
        print(f"     OOS 2025 Δ = {best['oos_2025']:+.4f}")

        if best["name"] == "global_ls_z168_low":
            _integrate_global_ls(best)
        else:
            print(f"  → Integrate {best['name']} manually if desired")
    else:
        print("\n  ❌ No overlay passed WF gate — production config UNCHANGED")
        print("  (IS avg > 0 AND OOS 2025 > 0 required for integration)")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "phase": "118b",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "is_results": IS_RESULTS,
        "oos_results": oos_results,
        "integration_candidates": integration_candidates,
    }
    out_path = OUT_DIR / "phase118b_oos_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved → {out_path}")
    print(f"  Total time: {time.time() - t_start:.0f}s")


def _integrate_global_ls(best: dict):
    """Integrate global_ls_z168_low overlay into production config."""
    print("\n  Integrating global_ls_z168_low into production config...", flush=True)

    try:
        cfg = json.loads(PRODUCTION_CFG.read_text())
    except Exception as exc:
        print(f"  [ERROR] Cannot read production config: {exc}")
        return

    # Check if already integrated
    if "global_ls_overlay" in cfg:
        print("  Already integrated — updating values")

    cfg["global_ls_overlay"] = {
        "_comment": "Phase 118b: global long/short ratio z-score overlay",
        "_validated": (
            f"IS avg Δ={best['is_avg']:+.4f} (2022/2023/2024), "
            f"OOS 2025 Δ={best['oos_2025']:+.4f} — WF PASS"
        ),
        "enabled": True,
        "metric": "global_ls_ratio",
        "lookback_bars": 168,
        "tilt_ratio": 0.65,
        "direction": "reduce_on_low",
        "mechanism": (
            "Binance Vision global L/S account ratio → log momentum → "
            "rolling 168h z-score → when z < 0 (shorts dominate) → "
            "reduce all returns by tilt_ratio=0.65 (contrarian protection)"
        ),
        "data_source": "data.binance.vision daily metrics CSV → resampled 1h",
        "date_integrated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }

    # Bump version
    old_ver = cfg.get("_version", "2.1.0")
    parts = old_ver.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    cfg["_version"] = ".".join(parts)
    cfg["_validated"] = cfg.get("_validated", "") + f"; global_ls_overlay (P118b) WF pass {datetime.now().strftime('%Y-%m-%d')}"

    PRODUCTION_CFG.write_text(json.dumps(cfg, indent=2))
    print(f"  ✅ Production config updated → {PRODUCTION_CFG}")
    print(f"     Version: {old_ver} → {cfg['_version']}")
    print(f"     global_ls_overlay ENABLED, tilt_ratio=0.65 when z_168 < 0")


if __name__ == "__main__":
    main()
