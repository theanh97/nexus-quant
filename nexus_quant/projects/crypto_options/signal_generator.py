"""
Crypto Options Signal Generator
================================

Production signal generator for the validated VRP + Skew MR ensemble.
Reads live Deribit data collected by the hourly cron job and generates
daily trading signals.

Ensemble: Mixed-frequency (hourly VRP + daily Skew MR)
Weights: VRP 40% + Skew MR 60% (wisdom v6.0.0)

Signals:
    VRP: Always short vol unless VRP z-score < -3.0 (extreme vol spike).
         Monitored hourly, rebalanced weekly.
    Skew MR: Fade extreme 25d skew when z > 2.0 (sell) or z < -2.0 (buy).
             Monitored daily, rebalanced weekly.

Output: Target weights per symbol + ensemble signal + metadata.
Logged to: artifacts/crypto_options/signals_log.jsonl

Usage:
    from nexus_quant.projects.crypto_options.signal_generator import OptionsSignalGenerator
    gen = OptionsSignalGenerator()
    signal = gen.generate()
    print(signal)  # dict with target_weights, rationale, confidence, etc.

    # CLI usage:
    python3 -m nexus_quant.projects.crypto_options.signal_generator
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.options_signal")

# ── Constants ─────────────────────────────────────────────────────────────────

LIVE_DATA_DIR = Path("data/cache/deribit/live")
SIGNALS_LOG = Path("artifacts/crypto_options/signals_log.jsonl")

SYMBOLS = ["BTC", "ETH"]

# Validated ensemble weights (wisdom v10.0.0)
ENSEMBLE_WEIGHTS = {"VRP": 0.40, "Skew_MR": 0.60}

# VRP strategy params (hourly-validated, adapted for daily monitoring)
VRP_PARAMS = {
    "base_leverage": 1.5,          # total short vol leverage
    "exit_z_threshold": -3.0,      # exit when z < this (never in practice)
    "vrp_lookback_days": 30,       # 30 days of data for z-score
    "rebalance_days": 7,           # weekly rebalance
}

# Skew MR params (daily-validated champion)
SKEW_PARAMS = {
    "lookback_days": 60,           # 60-day z-score window
    "z_entry": 2.0,                # enter when |z| > 2.0
    "z_exit": 0.0,                 # exit when z crosses 0
    "target_leverage": 1.0,
    "rebalance_days": 5,           # weekly rebalance
}

# Minimum data requirements
MIN_DATA_DAYS_VRP = 21             # need at least 21 days for VRP
MIN_DATA_DAYS_SKEW = 45            # need at least 45 days for Skew MR
MIN_DATA_DAYS_FULL_CONFIDENCE = 60 # full confidence after 60 days

# IV-percentile position sizing (R25+R28 optimized, validated on synthetic + delta hedging)
# Step function: <25th pct → 0.5x, 25-75th → 1.0x, >75th → 1.7x
# R28 sweep: 55 configs, best=0.3/2.0 (avg=2.677), production-safe=0.5/1.7 (avg=2.593)
IV_SIZING_ENABLED = True
IV_SIZING_LOOKBACK = 180           # 180-day lookback for percentile calculation
IV_SIZING_PCT_LOW = 0.25           # below this → scale down
IV_SIZING_PCT_HIGH = 0.75          # above this → scale up
IV_SIZING_SCALE_LOW = 0.50         # scale factor when IV is low
IV_SIZING_SCALE_HIGH = 1.70        # scale factor when IV is high (R28: 1.5→1.7)


class OptionsSignalGenerator:
    """Generate ensemble signals from live Deribit data."""

    def __init__(
        self,
        data_dir: Path = LIVE_DATA_DIR,
        signals_log: Path = SIGNALS_LOG,
    ) -> None:
        self.data_dir = data_dir
        self.signals_log = signals_log
        self._skew_positions: Dict[str, float] = {}  # current skew positions

    def generate(self) -> Dict[str, Any]:
        """Generate current signal from accumulated live data.

        Returns dict with:
            timestamp: ISO timestamp
            target_weights: {sym: weight} — net target per symbol
            vrp_signal: {sym: weight} — VRP component
            skew_signal: {sym: weight} — Skew MR component
            confidence: "high" | "medium" | "low" | "insufficient_data"
            data_days: number of days of data available
            rationale: human-readable explanation
            market_snapshot: current market data
        """
        now = datetime.now(tz=timezone.utc)

        # 1. Load live data
        live_data = self._load_live_data()
        if not live_data:
            signal = self._make_signal(
                now, {s: 0.0 for s in SYMBOLS},
                vrp={s: 0.0 for s in SYMBOLS},
                skew={s: 0.0 for s in SYMBOLS},
                confidence="insufficient_data",
                data_days=0,
                rationale="No live data available. Collector may not be running.",
            )
            self._log_signal(signal)
            return signal

        # 2. Compute daily aggregates
        daily = self._aggregate_to_daily(live_data)
        data_days = len(daily)

        # 3. Market snapshot (latest data point)
        snapshot = self._latest_snapshot(live_data)

        # 4. Determine confidence level
        if data_days >= MIN_DATA_DAYS_FULL_CONFIDENCE:
            confidence = "high"
        elif data_days >= MIN_DATA_DAYS_SKEW:
            confidence = "medium"
        elif data_days >= MIN_DATA_DAYS_VRP:
            confidence = "low"
        else:
            confidence = "insufficient_data"

        # 5. Generate VRP signal
        vrp_weights = self._vrp_signal(daily, data_days)

        # 6. Generate Skew MR signal
        skew_weights = self._skew_signal(daily, data_days)

        # 6.5 Apply IV-percentile position sizing (R25)
        iv_scales = {}
        if IV_SIZING_ENABLED and data_days >= IV_SIZING_LOOKBACK:
            for sym in SYMBOLS:
                pct = self._iv_percentile(daily, sym)
                if pct is not None:
                    scale = self._iv_sizing_scale(pct)
                    iv_scales[sym] = scale
                    vrp_weights[sym] = vrp_weights.get(sym, 0.0) * scale
                    skew_weights[sym] = skew_weights.get(sym, 0.0) * scale

        # 7. Combine ensemble
        target_weights = {}
        rationale_parts = []
        for sym in SYMBOLS:
            w_vrp = vrp_weights.get(sym, 0.0)
            w_skew = skew_weights.get(sym, 0.0)
            combined = (
                ENSEMBLE_WEIGHTS["VRP"] * w_vrp
                + ENSEMBLE_WEIGHTS["Skew_MR"] * w_skew
            )
            target_weights[sym] = round(combined, 4)

        # 8. Build rationale
        for sym in SYMBOLS:
            vrp_z = self._compute_vrp_zscore(daily, sym)
            skew_z = self._compute_skew_zscore(daily, sym)
            parts = []
            if vrp_z is not None:
                parts.append(f"VRP_z={vrp_z:+.2f}")
            if skew_z is not None:
                parts.append(f"Skew_z={skew_z:+.2f}")
            if sym in iv_scales:
                parts.append(f"IV_scale={iv_scales[sym]:.1f}")
            parts.append(f"VRP_w={vrp_weights.get(sym, 0):.2f}")
            parts.append(f"Skew_w={skew_weights.get(sym, 0):.2f}")
            parts.append(f"net={target_weights[sym]:.4f}")
            rationale_parts.append(f"{sym}: {' '.join(parts)}")

        rationale = f"{data_days}d data, {confidence} confidence. " + " | ".join(rationale_parts)

        signal = self._make_signal(
            now, target_weights,
            vrp=vrp_weights,
            skew=skew_weights,
            confidence=confidence,
            data_days=data_days,
            rationale=rationale,
            snapshot=snapshot,
        )
        self._log_signal(signal)
        return signal

    # ── VRP Signal ────────────────────────────────────────────────────────────

    def _vrp_signal(self, daily: List[Dict], data_days: int) -> Dict[str, float]:
        """VRP signal: always short vol unless VRP z-score < exit threshold."""
        if data_days < MIN_DATA_DAYS_VRP:
            return {s: 0.0 for s in SYMBOLS}

        weights = {}
        per_sym = VRP_PARAMS["base_leverage"] / len(SYMBOLS)

        for sym in SYMBOLS:
            z = self._compute_vrp_zscore(daily, sym)
            if z is None:
                weights[sym] = 0.0
            elif z < VRP_PARAMS["exit_z_threshold"]:
                weights[sym] = 0.0  # vol spike → exit
            else:
                weights[sym] = -per_sym  # short vol

        return weights

    def _compute_vrp_zscore(self, daily: List[Dict], sym: str) -> Optional[float]:
        """Compute VRP z-score from daily data."""
        lookback = VRP_PARAMS["vrp_lookback_days"]
        recent = daily[-lookback:] if len(daily) >= lookback else daily

        vrp_vals = []
        for d in recent:
            iv = d.get(f"{sym}_iv_atm")
            rv = d.get(f"{sym}_rv_24h")
            if iv is not None and rv is not None:
                vrp_vals.append(iv - rv)

        if len(vrp_vals) < 10:
            return None

        current = vrp_vals[-1]
        mean = sum(vrp_vals) / len(vrp_vals)
        std = statistics.pstdev(vrp_vals) if len(vrp_vals) > 1 else 0.0
        if std < 1e-6:
            return 0.0
        return (current - mean) / std

    # ── Skew MR Signal ────────────────────────────────────────────────────────

    def _skew_signal(self, daily: List[Dict], data_days: int) -> Dict[str, float]:
        """Skew MR signal: fade extreme skew z-scores."""
        if data_days < MIN_DATA_DAYS_SKEW:
            return {s: 0.0 for s in SYMBOLS}

        weights = {}
        per_sym = SKEW_PARAMS["target_leverage"] / len(SYMBOLS)

        for sym in SYMBOLS:
            z = self._compute_skew_zscore(daily, sym)
            current_pos = self._skew_positions.get(sym, 0.0)

            if z is None:
                weights[sym] = 0.0
                self._skew_positions[sym] = 0.0
                continue

            # Entry/exit logic (same as SkewTradeV2Strategy)
            if current_pos == 0.0:
                if z >= SKEW_PARAMS["z_entry"]:
                    weights[sym] = -per_sym  # sell skew
                elif z <= -SKEW_PARAMS["z_entry"]:
                    weights[sym] = per_sym   # buy skew
                else:
                    weights[sym] = 0.0
            else:
                if abs(z) <= SKEW_PARAMS["z_exit"]:
                    weights[sym] = 0.0       # exit
                elif current_pos < 0 and z <= 0:
                    weights[sym] = 0.0       # exit short skew
                elif current_pos > 0 and z >= 0:
                    weights[sym] = 0.0       # exit long skew
                else:
                    weights[sym] = current_pos  # hold

            self._skew_positions[sym] = weights[sym]

        return weights

    def _compute_skew_zscore(self, daily: List[Dict], sym: str) -> Optional[float]:
        """Compute skew z-score from daily data."""
        lookback = SKEW_PARAMS["lookback_days"]
        recent = daily[-lookback:] if len(daily) >= lookback else daily

        skew_vals = []
        for d in recent:
            s = d.get(f"{sym}_skew_25d")
            if s is not None:
                skew_vals.append(s)

        if len(skew_vals) < 10:
            return None

        current = skew_vals[-1]
        mean = sum(skew_vals) / len(skew_vals)
        std = statistics.pstdev(skew_vals) if len(skew_vals) > 1 else 0.0
        if std < 1e-6:
            return 0.0
        return (current - mean) / std

    # ── IV Percentile Sizing (R25) ───────────────────────────────────────────

    def _iv_percentile(self, daily: List[Dict], sym: str) -> Optional[float]:
        """Compute IV percentile rank over lookback window."""
        lookback = IV_SIZING_LOOKBACK
        if len(daily) < lookback:
            return None

        window = []
        for d in daily[-lookback:]:
            iv = d.get(f"{sym}_iv_atm")
            if iv is not None:
                window.append(iv)

        if len(window) < 30:
            return None

        current = window[-1]
        below = sum(1 for v in window[:-1] if v < current)
        return below / (len(window) - 1)

    @staticmethod
    def _iv_sizing_scale(pct: float) -> float:
        """Step function sizing: <25th→0.5x, 25-75th→1.0x, >75th→1.5x."""
        if pct < IV_SIZING_PCT_LOW:
            return IV_SIZING_SCALE_LOW
        elif pct > IV_SIZING_PCT_HIGH:
            return IV_SIZING_SCALE_HIGH
        return 1.0

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_live_data(self) -> Dict[str, List[Dict]]:
        """Load all live data from CSV files, per symbol."""
        result: Dict[str, List[Dict]] = {}

        for sym in SYMBOLS:
            rows = []
            csv_files = sorted(self.data_dir.glob(f"**/{sym}_*.csv"))
            for f in csv_files:
                try:
                    with open(f) as fp:
                        reader = csv.DictReader(fp)
                        for row in reader:
                            rows.append(row)
                except Exception as e:
                    logger.warning("Failed to read %s: %s", f, e)
            if rows:
                result[sym] = rows

        return result

    def _aggregate_to_daily(self, live_data: Dict[str, List[Dict]]) -> List[Dict]:
        """Aggregate hourly data to daily closing values.

        Returns list of daily dicts with keys like:
            BTC_price, BTC_iv_atm, BTC_skew_25d, BTC_rv_24h, ...
            ETH_price, ETH_iv_atm, ...
            date (YYYY-MM-DD)
        """
        # Group by date
        date_data: Dict[str, Dict[str, Any]] = {}

        for sym, rows in live_data.items():
            for row in rows:
                ts = row.get("ts_utc", "")
                date = ts[:10]  # YYYY-MM-DD
                if not date:
                    continue

                if date not in date_data:
                    date_data[date] = {"date": date}

                # Take last value for each day (closing snapshot)
                d = date_data[date]
                for field in ["price", "iv_atm", "iv_25d_put", "iv_25d_call",
                              "skew_25d", "term_spread", "rv_1h", "rv_24h"]:
                    val = row.get(field, "")
                    if val:
                        try:
                            d[f"{sym}_{field}"] = float(val)
                        except (ValueError, TypeError):
                            pass

        # Sort by date and return
        return [date_data[d] for d in sorted(date_data.keys())]

    def _latest_snapshot(self, live_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Get the most recent data point per symbol."""
        snapshot = {}
        for sym, rows in live_data.items():
            if rows:
                last = rows[-1]
                snapshot[sym] = {
                    "ts": last.get("ts_utc"),
                    "price": float(last["price"]) if last.get("price") else None,
                    "iv_atm": float(last["iv_atm"]) if last.get("iv_atm") else None,
                    "skew_25d": float(last["skew_25d"]) if last.get("skew_25d") else None,
                    "rv_24h": float(last["rv_24h"]) if last.get("rv_24h") else None,
                }
        return snapshot

    # ── Output ────────────────────────────────────────────────────────────────

    def _make_signal(
        self, now: datetime, target_weights: Dict[str, float],
        vrp: Dict[str, float], skew: Dict[str, float],
        confidence: str, data_days: int, rationale: str,
        snapshot: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        return {
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "epoch": int(now.timestamp()),
            "target_weights": target_weights,
            "vrp_signal": vrp,
            "skew_signal": skew,
            "ensemble_weights": ENSEMBLE_WEIGHTS,
            "confidence": confidence,
            "data_days": data_days,
            "rationale": rationale,
            "market_snapshot": snapshot or {},
            "gross_leverage": sum(abs(w) for w in target_weights.values()),
        }

    def _log_signal(self, signal: Dict[str, Any]) -> None:
        """Append signal to JSONL log."""
        self.signals_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.signals_log, "a") as f:
            f.write(json.dumps(signal, default=str) + "\n")
        logger.info("Signal logged to %s", self.signals_log)


def status_report() -> str:
    """Show signal generator status and latest signal."""
    lines = ["=== Crypto Options Signal Generator Status ===", ""]

    gen = OptionsSignalGenerator()
    live_data = gen._load_live_data()
    daily = gen._aggregate_to_daily(live_data) if live_data else []

    lines.append(f"Data directory: {LIVE_DATA_DIR}")
    lines.append(f"Data days: {len(daily)}")

    for sym in SYMBOLS:
        rows = live_data.get(sym, [])
        lines.append(f"  {sym}: {len(rows)} hourly snapshots")

    if len(daily) < MIN_DATA_DAYS_VRP:
        lines.append(f"\nStatus: COLLECTING DATA ({len(daily)}/{MIN_DATA_DAYS_VRP} days for VRP)")
        lines.append(f"  VRP needs {MIN_DATA_DAYS_VRP} days → ready ~{(datetime.now() + timedelta(days=MIN_DATA_DAYS_VRP - len(daily))).strftime('%Y-%m-%d')}")
        lines.append(f"  Skew needs {MIN_DATA_DAYS_SKEW} days → ready ~{(datetime.now() + timedelta(days=MIN_DATA_DAYS_SKEW - len(daily))).strftime('%Y-%m-%d')}")
    else:
        lines.append(f"\nStatus: GENERATING SIGNALS")
        signal = gen.generate()
        lines.append(f"  Confidence: {signal['confidence']}")
        lines.append(f"  Target weights: {signal['target_weights']}")
        lines.append(f"  VRP signal: {signal['vrp_signal']}")
        lines.append(f"  Skew signal: {signal['skew_signal']}")
        lines.append(f"  Gross leverage: {signal['gross_leverage']:.4f}")

    # Latest signal from log
    if gen.signals_log.exists():
        try:
            with open(gen.signals_log) as f:
                last_line = None
                for line in f:
                    last_line = line
            if last_line:
                last = json.loads(last_line)
                lines.append(f"\nLatest logged signal: {last['timestamp']}")
                lines.append(f"  Weights: {last['target_weights']}")
                lines.append(f"  Confidence: {last['confidence']}")
        except Exception:
            pass

    lines.append("")
    lines.append("Ensemble: VRP 40% + Skew MR 60% (mixed-frequency)")
    lines.append("  VRP: hourly-validated, avg Sharpe 3.649")
    lines.append("  Skew MR: daily-validated, avg Sharpe 1.744")
    lines.append("  Ensemble: avg Sharpe 2.723 (walk-forward 2021-2025)")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Crypto Options Signal Generator")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--generate", action="store_true", help="Generate signal now")
    args = parser.parse_args()

    if args.status:
        print(status_report())
    elif args.generate:
        gen = OptionsSignalGenerator()
        signal = gen.generate()
        print(json.dumps(signal, indent=2, default=str))
    else:
        print(status_report())
