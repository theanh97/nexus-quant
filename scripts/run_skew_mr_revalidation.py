"""
Crypto Options Skew MR — Revalidation with Real Deribit Data
=============================================================
Run this after 60+ days of live Deribit data collection (~2026-04-20).

Purpose:
  - Load real IV/skew/term data from live collector CSVs
  - Re-run skew mean-reversion strategy with actual Deribit vol surface
  - Walk-forward validate: Train 2026-02-20 to 2026-04-01, OOS 2026-04-01+
  - If avg Sharpe > 0.5 and min Sharpe > 0.3 → PASS → add to portfolio as 3rd slot

Usage:
    python3 scripts/run_skew_mr_revalidation.py
    python3 scripts/run_skew_mr_revalidation.py --status   # check data availability
"""
from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("skew_mr_reval")

LIVE_DATA_DIR = ROOT / "data" / "cache" / "deribit" / "live"
MIN_DAYS_REQUIRED = 60
COLLECTION_START = "2026-02-20"


def check_data_status() -> dict:
    """Check how many days of live Deribit data are available."""
    if not LIVE_DATA_DIR.exists():
        return {"available": False, "days": 0, "message": "No live data directory found."}

    # Count unique dates across all CSVs
    dates = set()
    for csv_path in LIVE_DATA_DIR.rglob("*.csv"):
        # Format: BTC_YYYY-MM-DD.csv or ETH_YYYY-MM-DD.csv
        name = csv_path.stem
        parts = name.split("_", 1)
        if len(parts) == 2:
            dates.add(parts[1])

    n_days = len(dates)
    ready = n_days >= MIN_DAYS_REQUIRED

    sorted_dates = sorted(dates) if dates else []
    first = sorted_dates[0] if sorted_dates else "N/A"
    last = sorted_dates[-1] if sorted_dates else "N/A"

    return {
        "available": ready,
        "days": n_days,
        "min_required": MIN_DAYS_REQUIRED,
        "first_date": first,
        "last_date": last,
        "remaining": max(0, MIN_DAYS_REQUIRED - n_days),
        "message": f"{'READY' if ready else 'COLLECTING'}: {n_days}/{MIN_DAYS_REQUIRED} days ({first} to {last})",
    }


def load_live_data():
    """Load live Deribit IV/skew data from collector CSVs."""
    import csv

    records = {"BTC": [], "ETH": []}

    for csv_path in sorted(LIVE_DATA_DIR.rglob("*.csv")):
        name = csv_path.stem
        parts = name.split("_", 1)
        if len(parts) != 2:
            continue
        sym = parts[0]  # BTC or ETH
        if sym not in records:
            continue

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    records[sym].append({
                        "timestamp": row.get("timestamp", ""),
                        "price": float(row.get("price", 0)),
                        "iv_atm": float(row.get("iv_atm", 0)),
                        "iv_25d_put": float(row.get("iv_25d_put", 0)),
                        "iv_25d_call": float(row.get("iv_25d_call", 0)),
                        "skew_25d": float(row.get("skew_25d", 0)),
                        "rv_realized": float(row.get("rv_realized", 0)),
                        "term_spread": float(row.get("term_spread", 0)),
                    })
                except (ValueError, KeyError):
                    continue

    logger.info(f"Loaded live data: BTC={len(records['BTC'])} rows, ETH={len(records['ETH'])} rows")
    return records


def run_revalidation():
    """Run walk-forward revalidation of Skew MR with real data."""
    status = check_data_status()
    logger.info(f"Data status: {status['message']}")

    if not status["available"]:
        logger.warning(
            f"Not enough data yet. Need {status['remaining']} more days. "
            f"ETA: ~{status['remaining']} days from now."
        )
        logger.info("Run the live collector to keep gathering data:")
        logger.info("  python3 -m nexus_quant.projects.crypto_options.collectors --loop")
        return

    records = load_live_data()
    if not records["BTC"] or not records["ETH"]:
        logger.error("Live data files exist but are empty or malformed.")
        return

    # Build dataset from live data
    logger.info("Building dataset from live Deribit data...")

    # TODO: When we have 60+ days of real data, implement:
    # 1. Convert records to MarketDataset with real skew features
    # 2. Instantiate SkewTradeStrategy with default params
    # 3. Run backtest on first 40 days (IS), validate on remaining 20+ days (OOS)
    # 4. Compute Sharpe, check pass criteria
    # 5. If PASS: update portfolio config to add as 3rd slot

    logger.info("=" * 60)
    logger.info("Revalidation pipeline ready. Waiting for sufficient data.")
    logger.info(f"  Current: {status['days']} days")
    logger.info(f"  Required: {MIN_DAYS_REQUIRED} days")
    logger.info(f"  ETA: ~{status['remaining']} days")
    logger.info("=" * 60)

    # Save status
    output = {
        "script": "run_skew_mr_revalidation.py",
        "run_date": datetime.now().isoformat(),
        "data_status": status,
        "verdict": "WAITING_FOR_DATA",
        "pass_criteria": {
            "min_sharpe_oos": 0.3,
            "avg_sharpe_wf": 0.5,
            "max_correlation_with_perps": 0.3,
        },
    }
    out_path = ROOT / "artifacts" / "skew_mr_revalidation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Status saved: {out_path}")


def main():
    if "--status" in sys.argv:
        status = check_data_status()
        print(json.dumps(status, indent=2))
        return

    logger.info("=" * 60)
    logger.info("NEXUS — Skew MR Revalidation with Real Deribit Data")
    logger.info("=" * 60)
    run_revalidation()


if __name__ == "__main__":
    main()
