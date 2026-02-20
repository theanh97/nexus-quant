# NEXUS Project #2: Crypto Options / Volatility (Deribit)
# Implementation Plan — For Claude Terminal

## GOAL
Build a crypto options volatility trading project inside NEXUS platform.
Target: Sharpe > 1.0, correlation with spot crypto < 0.3.

## CONTEXT
- NEXUS platform is at `/Users/qtmobile/Desktop/Nexus - Quant Trading `
- Multi-project architecture already exists: `nexus_quant/projects/`
- MarketDataset has extensible `features` dict and `market_type` field
- Dynamic strategy registry supports project-contributed strategies
- BacktestEngine skips funding PnL when `dataset.has_funding` is False
- Memory hierarchy: `memory/L1_crypto/` (shared), `memory/L2_crypto_options/` (project)
- Python: `/usr/bin/python3` (3.9.6), stdlib-only preferred

## PHASE 1: Data Infrastructure (Day 1-2)

### 1.1 Create project directory
```
nexus_quant/projects/crypto_options/
├── __init__.py
├── project.yaml
├── providers/
│   ├── __init__.py
│   └── deribit_rest.py      # Deribit REST API client
├── strategies/
│   ├── __init__.py           # STRATEGIES dict for dynamic registration
│   ├── variance_premium.py   # Strategy #1: Short variance premium
│   ├── skew_trade.py         # Strategy #2: Skew mean-reversion
│   └── term_structure.py     # Strategy #3: Calendar spread
├── costs/
│   ├── __init__.py
│   └── deribit_fees.py       # Deribit fee model (maker/taker)
├── vol_surface.py            # Vol surface builder
└── greeks.py                 # Black-Scholes Greeks calculator
```

### 1.2 Deribit Data Provider (`providers/deribit_rest.py`)
- Deribit REST API: `https://www.deribit.com/api/v2/public/`
- Endpoints needed:
  - `get_instruments` — list all options (BTC, ETH)
  - `get_tradingview_chart_data` — OHLCV for options
  - `get_book_summary_by_currency` — current order book
  - `get_historical_volatility` — realized vol
  - `ticker` — current IV, mark price, Greeks
- Historical data: Use `get_tradingview_chart_data` with resolution=60 (1h)
- Store as CSV cache in `data/cache/deribit/`
- Output: `MarketDataset` with:
  - `market_type = "options"`
  - `perp_close` = underlying price (BTC-PERPETUAL, ETH-PERPETUAL)
  - `features = {`
    - `"iv_atm": {"BTC": [0.65, 0.63, ...]}` — ATM implied vol per bar
    - `"iv_25d_put": {...}` — 25-delta put IV
    - `"iv_25d_call": {...}` — 25-delta call IV
    - `"skew_25d": {...}` — 25d put IV - 25d call IV
    - `"rv_realized": {...}` — realized vol (close-to-close)
    - `"term_spread": {...}` — front-month IV - back-month IV
    - `"put_call_ratio": {...}` — put/call volume ratio
    - `"max_pain": {...}` — max pain strike level
  - `}`
  - `funding = {}` (empty — no funding for options)

### 1.3 Deribit API Notes
- No API key needed for public endpoints (historical data)
- Rate limit: 20 req/sec (public)
- BTC options: strikes every $500-$1000, expiries weekly + monthly + quarterly
- ETH options: strikes every $25-$50
- Historical data available from ~2019

## PHASE 2: Vol Surface & Greeks Engine (Day 2-3)

### 2.1 Black-Scholes Calculator (`greeks.py`)
Implement stdlib-only (no numpy/scipy):
- `bs_price(S, K, T, r, sigma, option_type)` — option price
- `bs_delta(S, K, T, r, sigma, option_type)`
- `bs_gamma(S, K, T, r, sigma)`
- `bs_vega(S, K, T, r, sigma)`
- `bs_theta(S, K, T, r, sigma, option_type)`
- `implied_vol(market_price, S, K, T, r, option_type)` — Newton-Raphson solver
- Use `math.erf` for normal CDF (Python 3.2+)

### 2.2 Vol Surface Builder (`vol_surface.py`)
- Input: list of (strike, expiry, IV) from Deribit
- Build vol surface interpolation:
  - X-axis: moneyness (log(K/S))
  - Y-axis: time to expiry
  - Z-axis: implied vol
- Extract key features per timestamp:
  - ATM IV (moneyness ≈ 0)
  - 25-delta skew (put IV - call IV)
  - Butterfly spread (25d put + 25d call - 2*ATM)
  - Term structure slope (front vs back)
- Simple linear interpolation (stdlib-only, no scipy)

## PHASE 3: Volatility Strategies (Day 3-5)

### 3.1 Strategy #1: Variance Risk Premium (`variance_premium.py`)
**Signal**: RV(realized) < IV(implied) → short vol is profitable
- Compute: VRP = IV_atm - RV_realized_21d
- When VRP > threshold: short straddle (sell ATM put + call)
- When VRP < threshold: go flat or long vol
- Position sizing: Vega-neutral, max 5% of equity risk
- Rebalance: daily
- **This is the most robust signal — variance premium exists in ALL option markets**

### 3.2 Strategy #2: Skew Mean-Reversion (`skew_trade.py`)
**Signal**: 25-delta skew deviates from mean → trade the reversion
- Compute: skew_z = (current_skew - mean_skew_30d) / std_skew_30d
- When skew_z > 1.5: sell puts, buy calls (skew too expensive)
- When skew_z < -1.5: buy puts, sell calls (skew too cheap)
- Delta-hedge with underlying
- Rebalance: daily

### 3.3 Strategy #3: Term Structure Calendar Spread (`term_structure.py`)
**Signal**: Front-month IV vs back-month IV spread mean-reverts
- Compute: term_spread = IV_front - IV_back
- When term_spread > mean + 1σ: sell front, buy back (contango too steep)
- When term_spread < mean - 1σ: buy front, sell back (backwardation)
- Vega-neutral construction
- Rebalance: weekly

### 3.4 Strategy Registration
In `strategies/__init__.py`:
```python
from .variance_premium import VariancePremiumStrategy
from .skew_trade import SkewTradeStrategy
from .term_structure import TermStructureStrategy

STRATEGIES = {
    "crypto_vrp": VariancePremiumStrategy,
    "crypto_skew_mr": SkewTradeStrategy,
    "crypto_term_structure": TermStructureStrategy,
}
```

## PHASE 4: Backtesting (Day 5-6)

### 4.1 Options Backtest Adapter
Since options PnL is different from linear instruments, create:
- `OptionsBacktestEngine` that extends base engine
- PnL components: delta PnL + gamma PnL + theta PnL + vega PnL + vol PnL
- OR: simplified approach — treat each strategy's output as `target_weights`
  on the underlying (delta-equivalent exposure), then use existing engine

**Recommended: Start with simplified approach** — compute delta-equivalent
weights and let existing BacktestEngine handle price PnL. Add options-specific
PnL breakdown later.

### 4.2 Cost Model (`costs/deribit_fees.py`)
- Deribit fees: Maker 0.03% of underlying, Taker 0.05% of underlying
- Options: Maker 0.01% of underlying (capped at 12.5% of option price)
- Min fee: 0.0001 BTC per contract
- Implement as `ExecutionCostModel` subclass

### 4.3 Backtest Configs
Create configs:
- `configs/crypto_options_vrp.json` — variance premium backtest
- `configs/crypto_options_skew.json` — skew mean-reversion backtest
- `configs/crypto_options_ensemble.json` — all 3 combined

## PHASE 5: Validation & Deployment (Day 6-7)

### 5.1 Walk-Forward Validation
- Train: 2019-2022, Test: 2023-2024, OOS: 2025-2026
- Same framework as crypto_perps (yearly Sharpe per period)
- Must pass: MIN year Sharpe > 0.5, AVG > 1.0

### 5.2 Correlation Check
- Compute daily return correlation vs crypto_perps P91b strategy
- Target: correlation < 0.3 (uncorrelated alpha)

### 5.3 Memory Integration
- Write results to `memory/L2_crypto_options/`
- If VRP signal validates → promote to `memory/L1_crypto/` (applies to all crypto)

## KEY FILES TO CREATE (ORDERED)
1. `nexus_quant/projects/crypto_options/__init__.py`
2. `nexus_quant/projects/crypto_options/project.yaml`
3. `nexus_quant/projects/crypto_options/greeks.py`
4. `nexus_quant/projects/crypto_options/vol_surface.py`
5. `nexus_quant/projects/crypto_options/providers/__init__.py`
6. `nexus_quant/projects/crypto_options/providers/deribit_rest.py`
7. `nexus_quant/projects/crypto_options/costs/__init__.py`
8. `nexus_quant/projects/crypto_options/costs/deribit_fees.py`
9. `nexus_quant/projects/crypto_options/strategies/__init__.py`
10. `nexus_quant/projects/crypto_options/strategies/variance_premium.py`
11. `nexus_quant/projects/crypto_options/strategies/skew_trade.py`
12. `nexus_quant/projects/crypto_options/strategies/term_structure.py`
13. `configs/crypto_options_vrp.json`
14. `memory/L1_crypto/` (update wisdom)
15. `memory/L2_crypto_options/wisdom.json`

## SAFETY
- ONLY touch files inside the project directory
- NEVER modify system files
- All existing crypto_perps code must continue to work unchanged
- Use `market_type="options"` in MarketDataset
- Use `features` dict for vol surface data, NOT hardcoded fields

## SUCCESS CRITERIA
- [ ] Deribit data provider fetches historical vol data (2019-2026)
- [ ] Vol surface builder extracts ATM IV, skew, term structure
- [ ] At least 1 strategy (VRP) achieves Sharpe > 1.0 in backtest
- [ ] Walk-forward validation passes (MIN year > 0.5)
- [ ] Correlation with crypto_perps < 0.3
- [ ] All code passes syntax check
- [ ] Committed + pushed to GitHub
