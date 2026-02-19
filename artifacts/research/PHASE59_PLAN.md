# Phase 59+ Research Plan — New Alpha Signals

## Current State (Phase 58 complete)
- **Champion**: V1-Long k=2 (AVG Sharpe 1.558, MIN 0.948 across 5 years)
- **Params**: carry=0.35, mom=0.45, MR=0.20, mom_lb=336h, mr_lb=72h, rebal=60h
- **Weakness**: 2021 MDD=25.1%, no bear-market hedging
- **All optimization exhausted**: 200+ backtests confirm V1-Long at robustness frontier
- **Next step**: Find ORTHOGONAL alpha signals (uncorrelated with V1-Long)

## Available Data Fields (MarketDataset)
- `perp_close` — perpetual futures close prices
- `perp_volume` — trading volume per bar
- `taker_buy_volume` — aggressive buyer volume
- `funding` — event-based funding rates (symbol → epoch → rate)
- `spot_close` — spot prices (optional)
- `open_interest` — OI per bar (optional)
- `long_short_ratio_global/top` — positioning ratios (optional)
- `perp_mark_close`, `perp_index_close` — mark/index prices
- `basis(symbol, idx)` — perp/spot spread

## Strategy Ideas to Implement

### 1. Lead-Lag Alpha (BTC → Altcoins)
**Hypothesis**: BTC price moves lead altcoin moves by ~4-24 hours.
**Signal**:
- Compute BTC return over last N bars (e.g., 4h, 8h, 12h)
- If BTC up → go long altcoins with highest beta to BTC
- If BTC down → go short altcoins with highest beta
- Exclude BTC from traded universe (it's the signal, not the trade)
**Why orthogonal**: This is a CROSS-ASSET signal, not self-referential momentum/MR
**Lookbacks to test**: 4h, 8h, 12h, 24h
**Expected**: Low Sharpe individually (~0.3-0.7) but potentially uncorrelated with V1

### 2. Volume Reversal Alpha (Capitulation/Climax Detection)
**Hypothesis**: Extreme volume + negative return = capitulation = buy. Extreme volume + positive return = distribution = sell.
**Signal**:
- Compute volume z-score (rolling 168h window)
- Compute return sign over last 24h
- Capitulation score = volume_zscore × (-return_sign)
- High score → capitulation → long; Low score → climax → short
- Cross-sectional ranking, top-k long / bottom-k short
**Why orthogonal**: Volume-based, not price-momentum or MR
**Lookbacks**: Volume window 72h/168h, return window 12h/24h/48h
**Expected**: Sharpe 0.3-0.8 if real

### 3. Basis Momentum Alpha (Funding Rate Changes)
**Hypothesis**: CHANGE in funding rate predicts future returns (not level).
- Rising funding → increasing speculation → trend continues short-term
- Falling funding → decreasing speculation → reversal coming
**Signal**:
- Compute ΔFunding = funding_rate(t) - funding_rate(t-N)
- Cross-sectional ranking by ΔFunding
- Long top-k (rising funding = trend continuation)
- Short bottom-k (falling funding)
**Why orthogonal**: Funding CHANGE is different from funding LEVEL (carry)
**Lookbacks**: ΔFunding over 24h, 72h, 168h
**Expected**: Sharpe 0.2-0.5 (weak but potentially decorrelated)

### 4. Volatility Breakout Alpha
**Hypothesis**: After vol compression, breakout direction predicts trend.
**Signal**:
- Compute vol ratio = vol_short(24h) / vol_long(168h)
- When ratio < 0.5 (vol compression) → prepare for breakout
- Direction = sign of last 12h return during breakout
- Long breakout-up assets, short breakout-down assets
**Why orthogonal**: Regime-timing signal, not level-based
**Lookbacks**: Short vol 12h/24h, long vol 72h/168h
**Expected**: Sharpe 0.3-0.6

### 5. Relative Strength Acceleration
**Hypothesis**: Assets accelerating in relative strength outperform.
**Signal**:
- RS_short = rank(return_72h) among universe
- RS_long = rank(return_336h) among universe
- Acceleration = RS_short - RS_long
- Long top-k accelerating, short bottom-k decelerating
**Why orthogonal**: Measures CHANGE in momentum, not momentum itself
**Expected**: Sharpe 0.3-0.7

## Execution Plan

### Phase 59A: Implement & Test Individual Signals
1. Implement all 5 strategies as separate files
2. Test each across 5 years (2021, 2022, 2023, 2025, 2026)
3. Filter: keep only strategies with AVG Sharpe > 0 and no catastrophic year (< -1.0)

### Phase 59B: Correlation Analysis
1. For surviving strategies, compute correlation of daily returns with V1-Long
2. Target: correlation < 0.3 with V1-Long
3. Rank by: Sharpe × (1 - correlation_with_V1Long)

### Phase 59C: Ensemble Construction
1. Best uncorrelated signal + V1-Long in ensemble
2. Test weight combinations: 80/20, 70/30, 60/40
3. Validate: ensemble must beat V1-Long standalone on BOTH avg AND min Sharpe

### Phase 59D: Risk Overlay (if MDD still > 20%)
1. Portfolio-level drawdown protection
2. Dynamic leverage based on realized vol regime
3. Target: reduce 2021 MDD from 25% to <15% without killing Sharpe

## Priority Order
1. **Lead-Lag** (highest academic backing)
2. **Volume Reversal** (most orthogonal to price-based signals)
3. **Basis Momentum** (uses funding data differently)
4. **Volatility Breakout** (regime-based)
5. **RS Acceleration** (most similar to existing momentum, lowest priority)

## Success Criteria
- At least 1 new signal with AVG Sharpe > 0.3 and correlation < 0.3 with V1-Long
- Ensemble with V1-Long: AVG Sharpe > 1.6 AND MIN > 1.0
- MDD reduction: worst-year MDD < 20%

## Files to Create
```
nexus_quant/strategies/lead_lag_alpha.py
nexus_quant/strategies/volume_reversal_alpha.py
nexus_quant/strategies/basis_momentum_alpha.py
nexus_quant/strategies/vol_breakout_alpha.py
nexus_quant/strategies/rs_acceleration_alpha.py
configs/run_leadlag_*.json (per year)
configs/run_volrev_*.json (per year)
configs/run_basismom_*.json (per year)
```

## Reference
- V1-Long config: `configs/run_nexus_alpha_v1_long.json`
- Strategy base class: `nexus_quant/strategies/base.py`
- Registry: `nexus_quant/strategies/registry.py`
- Backtest results: `artifacts/phase58_ensemble_dd_results.json`
