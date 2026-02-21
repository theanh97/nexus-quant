"""
SPX PCS Strategy Presets
=========================

Strategies được implement trong algoxpert engine (src/strategies/presets.py).
NEXUS reference only.

Available presets:
  baseline           — No technical filters, pure delta/VIX gate
  mr_bounce_dynamic  — Mean-reversion bounce with dynamic threshold
  ma_crossover       — MA(20/50) regime filter
  rsi_oversold       — RSI < 30 entry filter
  bb_oversold        — Bollinger lower band filter
  combo_2of3_mr_rsi_bb — MR + RSI + BB, 2/3 majority gate (RECOMMENDED)
  combo_3of3_mr_ma_bb  — MR + MA + BB, all 3 required
  trend_retake       — Trend continuation filter

Best candidate from IS optimization (4-day):
  delta=0.20, tp=0.5, sl=2.0, vix_gate=30 → Sharpe 19.47 (OVERFIT, 4 days only)
  Need: full 2021-2024 WF validation.
"""
