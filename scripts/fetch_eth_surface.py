#!/usr/bin/env python3
"""Fetch ETH real IV surface only (BTC already done in R50)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fetch_real_iv_surface import (
    fetch_spot_prices, fetch_weekly_surfaces, interpolate_to_daily,
    analyze_real_vs_synthetic, OUTPUT_DIR
)
import json

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("R50 (continued): ETH Real IV Surface")
print("=" * 70)

prices = fetch_spot_prices("ETH", "2019-06-01", "2026-02-21")
if not prices:
    print("No ETH prices!")
    sys.exit(1)

weekly = fetch_weekly_surfaces("ETH", prices, "2019-06-01", "2026-02-21")
if not weekly:
    print("No ETH surface data!")
    sys.exit(1)

# Save weekly
with open(OUTPUT_DIR / "ETH_weekly_surface.json", "w") as f:
    json.dump(weekly, f, indent=2)
print(f"\nSaved {len(weekly)} weekly points")

# Interpolate to daily
daily = interpolate_to_daily(weekly)
with open(OUTPUT_DIR / "ETH_daily_surface.json", "w") as f:
    json.dump(daily, f, indent=2)

# Save CSV
with open(OUTPUT_DIR / "ETH_daily_surface.csv", "w") as f:
    fields = ["date", "currency", "iv_atm", "skew_25d", "butterfly_25d",
              "term_spread", "iv_25d_put", "iv_25d_call", "spot", "interpolated"]
    f.write(",".join(fields) + "\n")
    for d in daily:
        vals = [str(d.get(field, "")) for field in fields]
        f.write(",".join(vals) + "\n")
print(f"Saved {len(daily)} daily points")

# Analyze
analyze_real_vs_synthetic(daily, "ETH")
