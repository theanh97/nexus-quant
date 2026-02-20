"""
NEXUS Track Record Generator
==============================
Generates the platform-wide performance report and saves to:
  - artifacts/track_record.json   (machine-readable)
  - Console output                (human-readable table)

Usage:
    cd "/Users/qtmobile/Desktop/Nexus - Quant Trading "
    python3 scripts/generate_track_record.py
    python3 scripts/generate_track_record.py --json   # JSON only
    python3 scripts/generate_track_record.py --update-live  # pull latest backtest results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEXUS Track Record Generator")
    p.add_argument("--json", action="store_true", help="Output JSON only")
    p.add_argument("--update-live", action="store_true",
                   help="Pull latest backtest results from artifacts/")
    p.add_argument("--out", default="artifacts/track_record.json", help="Output file")
    return p.parse_args()


def update_from_latest_artifacts(tr: "NexusTrackRecord") -> None:
    """
    Scan artifacts/metrics/ for latest backtest results and update the track record.
    Looks for: yearly_sharpe, cagr, max_drawdown in metrics JSON files.
    """
    artifacts_dir = ROOT / "artifacts" / "metrics"
    if not artifacts_dir.exists():
        return

    for metrics_file in sorted(artifacts_dir.glob("*.json")):
        try:
            with open(metrics_file) as f:
                m = json.load(f)
        except Exception:
            continue

        run_name = m.get("run_name", "")
        if not run_name:
            continue

        # Match to project
        proj_name = None
        if "cta" in run_name.lower() or "commodity" in run_name.lower():
            proj_name = "commodity_cta"
        elif "nexus" in run_name.lower() or "p91b" in run_name.lower():
            proj_name = "crypto_perps"

        if proj_name and proj_name in tr.projects:
            proj = tr.projects[proj_name]
            summary = m.get("summary", {})
            yearly = m.get("yearly", {})

            for yr_str, yr_data in yearly.items():
                try:
                    yr = int(yr_str)
                    sharpe = yr_data.get("sharpe")
                    mdd = yr_data.get("max_drawdown")
                    cagr = yr_data.get("cagr")
                    if sharpe is not None:
                        from nexus_quant.reporting.track_record import YearMetrics
                        proj.years[yr] = YearMetrics(
                            year=yr,
                            sharpe=round(sharpe, 3),
                            cagr_pct=round(cagr * 100, 2) if cagr else None,
                            max_drawdown_pct=round(mdd * 100, 2) if mdd else None,
                            status="oos",
                        )
                except (ValueError, TypeError, KeyError):
                    continue

            proj.compute_aggregates()


def main() -> None:
    args = parse_args()

    from nexus_quant.reporting.track_record import NexusTrackRecord

    tr = NexusTrackRecord.build_from_memory()

    if args.update_live:
        update_from_latest_artifacts(tr)

    tr.save(args.out)

    if args.json:
        print(json.dumps(tr.to_dict(), indent=2, default=str))
    else:
        print(tr.summary())
        print(f"\n[Saved to {args.out}]")


if __name__ == "__main__":
    main()
