"""
NEXUS Project: Commodity Futures CTA (Trend + Carry)
======================================================
Target: Sharpe > 0.8, beat SG CTA Index (Sharpe ~0.5–0.7)

Strategies:
- cta_trend      : Multi-timeframe trend following (20/60/120-day)
- cta_carry      : Roll-yield / carry proxy (backwardation vs contango)
- cta_mom_value  : Momentum + long-run value combo
- cta_ensemble   : Static-weight ensemble of all 3 + vol tilt overlay

Data: Yahoo Finance free continuous front-month futures (daily bars)
Universe: 13 commodities across Energy, Metals, Grains, Softs, Livestock
"""
from __future__ import annotations

__version__ = "1.0.0"
__project__ = "commodity_cta"
