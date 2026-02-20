# NEXUS Project #3: Commodity Futures CTA (Trend + Carry)
# Implementation Plan — For Claude Terminal

## GOAL
Build a commodity futures CTA-style project inside NEXUS platform.
Target: Sharpe > 0.8, beat SG CTA Index (Sharpe ~0.5-0.7).

## CONTEXT
- NEXUS platform is at `/Users/qtmobile/Desktop/Nexus - Quant Trading `
- Multi-project architecture already exists: `nexus_quant/projects/`
- MarketDataset has extensible `features` dict and `market_type` field
- Dynamic strategy registry supports project-contributed strategies
- BacktestEngine skips funding PnL when `dataset.has_funding` is False
- Memory hierarchy: `memory/L0_universal/` (shared), `memory/L2_commodity_cta/` (project)
- Python: `/usr/bin/python3` (3.9.6), stdlib-only preferred
- Volume momentum overlay validated in crypto → test transferability here

## PHASE 1: Data Infrastructure (Day 1-2)

### 1.1 Create project directory
```
nexus_quant/projects/commodity_cta/
├── __init__.py
├── project.yaml
├── providers/
│   ├── __init__.py
│   └── yahoo_futures.py     # Yahoo Finance futures data (free)
├── strategies/
│   ├── __init__.py           # STRATEGIES dict for dynamic registration
│   ├── trend_following.py    # Strategy #1: Multi-timeframe trend
│   ├── carry_roll.py         # Strategy #2: Roll yield / carry
│   ├── momentum_value.py     # Strategy #3: Momentum + value combo
│   └── cta_ensemble.py       # Strategy #4: Ensemble of all
├── costs/
│   ├── __init__.py
│   └── futures_fees.py       # CME/ICE futures commission model
└── features.py               # Commodity-specific feature engineering
```

### 1.2 Data Source: Yahoo Finance (Free)
Yahoo Finance provides continuous futures data for major commodities:
- Use `urllib.request` (stdlib) to fetch CSV data
- Tickers (continuous front-month):
  - **Energy**: CL=F (WTI Oil), NG=F (Natural Gas), BZ=F (Brent)
  - **Metals**: GC=F (Gold), SI=F (Silver), HG=F (Copper), PL=F (Platinum)
  - **Agriculture**: ZW=F (Wheat), ZC=F (Corn), ZS=F (Soybeans)
  - **Softs**: KC=F (Coffee), SB=F (Sugar), CT=F (Cotton)
  - **Livestock**: LE=F (Live Cattle), HE=F (Lean Hogs)
- Historical data: 2000-2026 (daily bars)
- URL pattern: `https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start}&period2={end}&interval=1d`

### 1.3 Yahoo Futures Provider (`providers/yahoo_futures.py`)
- Fetch daily OHLCV for 12-15 commodities
- Cache locally in `data/cache/yahoo_futures/`
- Handle missing data (some commodities have gaps)
- Output: `MarketDataset` with:
  - `market_type = "commodity"`
  - `perp_close` = adjusted close prices (continuous front-month)
  - `features = {`
    - `"volume": {"GC=F": [1234, ...]}` — daily volume
    - `"open": {...}` — open prices (for gap analysis)
    - `"high": {...}` — daily highs
    - `"low": {...}` — daily lows
    - `"atr_14": {...}` — 14-day ATR (computed)
    - `"roll_return": {...}` — contango/backwardation proxy
  - `}`
  - `funding = {}` (empty — no funding for commodity futures)

### 1.4 Roll Return / Carry Estimation
Since Yahoo only gives front-month, estimate carry via:
- Method 1: Price difference between current and prior contract roll dates
- Method 2: Use VIX-like proxies (if available)
- Method 3: Approximate from price momentum + seasonal patterns
- **Pragmatic**: Start with price momentum only, add carry signals later
  when CME data is available

## PHASE 2: Feature Engineering (Day 2-3)

### 2.1 Commodity Features (`features.py`)
Compute from OHLCV data:
- **Trend signals**:
  - `mom_20d`: 20-day price momentum (return)
  - `mom_60d`: 60-day price momentum
  - `mom_120d`: 120-day price momentum (main CTA signal)
  - `ewma_12_26_cross`: EWMA crossover signal
  - `breakout_52w`: 52-week high/low breakout
- **Volatility signals**:
  - `atr_14`: 14-day Average True Range
  - `rv_20d`: 20-day realized volatility
  - `vol_regime`: vol percentile (low/medium/high)
  - `vol_mom_z`: volume momentum z-score (transfer from crypto!)
- **Mean-reversion signals**:
  - `rsi_14`: 14-day RSI
  - `zscore_60d`: Z-score vs 60-day mean
  - `bollinger_pct`: Bollinger band percentile
- **Cross-commodity signals**:
  - `sector_momentum`: average momentum within sector (energy, metals, agri)
  - `correlation_break`: pairwise correlation regime changes

## PHASE 3: CTA Strategies (Day 3-5)

### 3.1 Strategy #1: Multi-Timeframe Trend (`trend_following.py`)
**Signal**: Classic CTA trend-following with NEXUS improvements
- 3 timeframes: 20d, 60d, 120d momentum
- Position sizing: inverse-vol weighted (risk parity across commodities)
- Max gross leverage: 2.0x
- Rebalance: daily
- Vol target: 10% annualized
- **Key improvement over traditional CTA**: NEXUS agentic loop optimizes
  lookback windows and weighting per commodity

### 3.2 Strategy #2: Carry / Roll Yield (`carry_roll.py`)
**Signal**: Contango vs backwardation determines long/short bias
- Long commodities in backwardation (positive carry)
- Short commodities in contango (negative carry)
- Combined with momentum filter (only trade carry in trending direction)
- Position sizing: equal risk contribution
- Rebalance: weekly (carry signals are slow-moving)

### 3.3 Strategy #3: Momentum + Value Combo (`momentum_value.py`)
**Signal**: Combine momentum (trend) with value (mean-reversion of relative price)
- Momentum: 120d return, cross-sectional rank
- Value: distance from 5-year average price (z-score)
- Combined score: 0.6 * momentum_rank + 0.4 * value_rank
- Long top 4, short bottom 4 (of 12-15 commodities)
- Rebalance: monthly

### 3.4 Strategy #4: CTA Ensemble (`cta_ensemble.py`)
**Ensemble**: Static weight combination (lesson from crypto: static > adaptive)
- Trend (40%) + Carry (30%) + MomValue (30%)
- Vol tilt overlay: reduce leverage when vol_regime = "high" (transfer from crypto!)
- Rebalance: daily (underlying signals update at their own frequency)

### 3.5 Strategy Registration
In `strategies/__init__.py`:
```python
from .trend_following import TrendFollowingStrategy
from .carry_roll import CarryRollStrategy
from .momentum_value import MomentumValueStrategy
from .cta_ensemble import CTAEnsembleStrategy

STRATEGIES = {
    "cta_trend": TrendFollowingStrategy,
    "cta_carry": CarryRollStrategy,
    "cta_mom_value": MomentumValueStrategy,
    "cta_ensemble": CTAEnsembleStrategy,
}
```

## PHASE 4: Backtesting (Day 5-6)

### 4.1 Cost Model (`costs/futures_fees.py`)
- CME E-mini futures: ~$2.25/contract round-trip
- As percentage of notional: ~0.01-0.02% (very cheap)
- Slippage estimate: 0.05% per trade (daily bars = low frequency)
- Implement as `ExecutionCostModel`:
  - `maker_fee = 0.0001` (0.01%)
  - `taker_fee = 0.0002` (0.02%)
  - `slippage_bps = 5`

### 4.2 Backtest Configs
Create configs:
- `configs/cta_trend.json` — trend-following only
- `configs/cta_carry.json` — carry/roll only
- `configs/cta_mom_value.json` — momentum+value only
- `configs/cta_ensemble.json` — full ensemble
- `configs/cta_production.json` — best validated config

### 4.3 Benchmark Comparison
- Benchmark: Buy-and-hold equal-weight commodity basket
- Target: Beat by Sharpe > 0.3
- Also compare vs: SG CTA Index returns (if available via Yahoo)

## PHASE 5: Validation & Deployment (Day 6-7)

### 5.1 Walk-Forward Validation
- Train: 2005-2015 (covers GFC, commodity supercycle)
- Test: 2016-2020 (flat commodity markets)
- OOS: 2021-2026 (inflation, Ukraine war, commodity boom/bust)
- Must pass: MIN period Sharpe > 0.3, AVG > 0.8

### 5.2 Robustness Checks
- Parameter perturbation: ±20% on lookback windows → Sharpe stable?
- Universe sensitivity: remove 2-3 commodities → still profitable?
- Correlation with crypto_perps: target < 0.2 (truly diversified)

### 5.3 Knowledge Transfer
- Test if vol momentum overlay (from crypto) improves commodity CTA
- If it works → promote to L0 universal memory (works across markets!)
- Write results to `memory/L2_commodity_cta/`

## KEY FILES TO CREATE (ORDERED)
1. `nexus_quant/projects/commodity_cta/__init__.py`
2. `nexus_quant/projects/commodity_cta/project.yaml`
3. `nexus_quant/projects/commodity_cta/providers/__init__.py`
4. `nexus_quant/projects/commodity_cta/providers/yahoo_futures.py`
5. `nexus_quant/projects/commodity_cta/features.py`
6. `nexus_quant/projects/commodity_cta/costs/__init__.py`
7. `nexus_quant/projects/commodity_cta/costs/futures_fees.py`
8. `nexus_quant/projects/commodity_cta/strategies/__init__.py`
9. `nexus_quant/projects/commodity_cta/strategies/trend_following.py`
10. `nexus_quant/projects/commodity_cta/strategies/carry_roll.py`
11. `nexus_quant/projects/commodity_cta/strategies/momentum_value.py`
12. `nexus_quant/projects/commodity_cta/strategies/cta_ensemble.py`
13. `configs/cta_trend.json`
14. `configs/cta_ensemble.json`
15. `memory/L2_commodity_cta/wisdom.json`

## SAFETY
- ONLY touch files inside the project directory
- NEVER modify system files
- All existing crypto_perps code must continue to work unchanged
- Use `market_type="commodity"` in MarketDataset
- Use `features` dict for commodity data, NOT hardcoded fields
- Yahoo Finance data is free but may have delays/gaps — handle gracefully

## SUCCESS CRITERIA
- [ ] Yahoo data provider fetches 12+ commodities (2005-2026)
- [ ] Feature engineering computes trend, vol, value signals
- [ ] At least 1 strategy achieves Sharpe > 0.8 in backtest
- [ ] Walk-forward validation passes (MIN period > 0.3)
- [ ] Correlation with crypto_perps < 0.2
- [ ] Vol momentum overlay tested for cross-market transfer
- [ ] All code passes syntax check
- [ ] Committed + pushed to GitHub

## BENCHMARKS TO BEAT
| Benchmark | Sharpe | Notes |
|-----------|--------|-------|
| Buy-and-hold commodity basket | ~0.2-0.3 | Equal weight, rebalance monthly |
| Simple 200-day MA crossover | ~0.4-0.5 | Classic CTA |
| SG CTA Index | ~0.5-0.7 | Industry benchmark |
| **NEXUS target** | **> 0.8** | **With agentic optimization** |
