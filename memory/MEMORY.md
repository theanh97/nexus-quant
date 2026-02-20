# Nexus Quant — Persistent Memory

## Champion Table (latest wins)
| Phase | Config | AVG | MIN | Notes |
|-------|--------|-----|-----|-------|
| 81 | V1+I437_bw168_k4+I600+F144 | 2.010 | 1.245 | 22.5/22.5/15/40 (k4 single-lb balanced) |
| 84 | V1+I437k4+I460k4+I410k4+I600+F144 | 2.040 | 1.431 | triple-lb balanced |
| **91b** | **V1+I460bw168k4+I415bw216k4+F144** | **2.010** | **1.576** | **27.47/19.67/32.47/20.39 — PROD BASE** |
| **146** | **Breadth regime switching (3 weight sets)** | **2.156** | **1.577** | **OBJ=1.8945 — PRODUCTION v2.2.0** |

## Regime-Adaptive Weight Switching (P144-146) — CURRENT PRODUCTION
**Production config v2.2.0 — breadth_regime_switching DEPLOYED**

| Label | V1 | I460bw168 | I415bw216 | F144 | Regime |
|-------|-----|-----------|-----------|------|---------|
| prod  | 27.47% | 19.67% | 32.47% | 20.39% | LOW (<33rd pct breadth) |
| mid   | 16.00% | 22.00% | 37.00% | 25.00% | MID (33-67%) |
| p143b |  5.00% | 25.00% | 45.00% | 25.00% | HIGH (>67%) |

**Classifier:** % symbols with positive 168h return → rolling 168h percentile → p_low=0.33, p_high=0.67
**Validation:** P144 IS +0.1820 OBJ | P145 LOYO 4/5 capture=87.5% | P146 WF avg_delta=+0.4020
**Key insight:** Vol-rank ≠ regime. 2022 bear ALSO prefers momentum-heavy weights.
True signal = funding/momentum richness (breadth), NOT realized volatility.

## API Notes (BacktestEngine)
- `BacktestConfig(costs=cost_model)` — ONLY costs param, nothing else
- `engine.run(dataset, strategy)` → BacktestResult with `.returns` list
- `dataset.close(symbol, idx)` — close price at bar idx
- `dataset.timeline` — timestamp array (len = n_bars)
- `dataset.funding_rate_at(symbol, ts)` — funding at timestamp ts
- **NO** `dataset.bars_for()` — does not exist!

## Critical Strategy Names
- `"nexus_alpha_v1"` ← CORRECT (NOT "v1_standard" — throws UnknownStrategy!)
- `"idio_momentum_alpha"` — params: lookback_bars, beta_window_bars, k_per_side
- `"funding_momentum_alpha"` — param: funding_lookback_bars
- `"funding_contrarian_alpha"` ← WRONG, never use

## Key Findings
- **I600**: REMOVE entirely (I600=0% strictly dominates)
- **F144 optimal**: 20.39% (ultra-fine; 22.5% = coarse artifact)
- **lb=415 global optimal**: cliff at lb=420 (2023: 1.143→0.758)
- **bw=216 upgrade**: use for idio signals lb≤460 (improves 2022/2023 floor)
- **k=4**: best for I437/I460/I410/I474; k=2 only for V1, F144, I600
- **Vol regime overlay**: threshold=0.5, scale=0.5, F144_boost=0.2 (Phase 129)

## Failed Signals / Approaches
- FX EMA, Bonds CTA (P138): FAIL — crypto only
- Universe expansion 10→20 (P140): FAIL
- Vol term structure (P134), Funding level (P135): NO IMPROVEMENT
- 5th signal candidates (P139): ALL FAIL
- I437_bw216 (2023=0.175), Idio_800_bw168: FAIL
- Vol-rank as regime proxy: WRONG — 2022 needs i415-heavy despite high vol

## Multi-Model Panel Rules (live in repo)
- Pass 1: GPT-5 mini + Gemini 2.5 Flash + DeepSeek reasoner (cheap/wide)
- Pass 2: Highest-dissent model debates
- Pass 3: GPT-5 or Sonnet 4 (arbiter)
- Max 1 expensive call per decision

## Phase Milestones
P91b: Production champion | P129: Vol regime overlay | P137: Breadth overlay
P141: idio_48h optimal | P142: idio_72h marginal | P143: Weight opt (2025 fails, keep prod)
P144: Regime-adaptive switching VALIDATED | P145: Breadth classifier LOYO 4/5
P146: WF validated, prod v2.2.0 deployed
P147: Conditional vol overlay test — CONFIRMED production optimal (no change)
P148: Funding dispersion boost (std>75th pct → ×1.15) VALIDATED — LOYO 3/5, WF 2/2, OBJ=1.8886 | prod v2.3.0
**Next: P149** — intraday seasonality or short-term reversal overlay
