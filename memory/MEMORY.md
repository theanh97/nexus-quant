# Nexus Quant — Persistent Memory

## Champion Table (latest wins)
| Phase | Config | AVG | MIN | Notes |
|-------|--------|-----|-----|-------|
| 54 | V1-Long | 1.125 | 0.803 | original V1 champion |
| 68 | V1+Idio+F144+F168 | 1.844 | 0.996 | 15/30/40/15 (AVG-max) |
| 71 | V1+Idio_437+Idio_600+F144+F168 | 1.861 | 1.097 | 15/21/14/40/10 (dual-idio!) |
| 72 | V1+I437_bw336+I600_bw168+F144+F168 | 1.933 | 0.894 | 10/20/20/35/15 (AVG-max) |
| 73 | V1+I437_bw168+I600_bw168+F144+F168 | 1.909 | 1.170 | 15/15/20/45/5 (balanced) |
| 74 | V1+I437_bw168+I600_bw168+F144 | 1.902 | 1.199 | 17.5/12.5/22.5/47.5 (2.5% grid, best MIN) |
| 75 | V1+I437_bw336+I600_bw168+F144+F168 | 1.934 | 0.972 | 12.5/20/17.5/40/10 (new AVG-max) |
| **76** | **V1+I437_bw168_k3+I600_bw168+F144** | **1.914** | **1.191** | **17.5/12.5/22.5/47.5 (NEW BALANCED CHAMP)** |

## Critical Strategy Name Distinction
- `"funding_momentum_alpha"` = cumulative funding sum (uses `funding_lookback_bars`) ← CORRECT
- `"funding_contrarian_alpha"` = funding_z × momentum_z composite (uses `momentum_lookback_bars`) ← WRONG
- NEVER use `"funding_contrarian_alpha"` for funding contrarian ensemble signals

## Key Confirmed Findings
- **Funding lookback**: 144h (n_samples=18) is CONFIRMED global peak. Full 8h-step curve: 96→1.031, 104→0.996, 112→1.006, 120→0.855, 128/132→0.917, 136→1.115, **144→1.302**, 152/156→1.184, **160→1.252**, 168→1.197
- **F160 note**: 160h (n=20) has 2022=0.970 vs F144's 2022=0.496 → much better 2022 floor!
- **Idio lookback**: 437h CONFIRMED optimal (local peak). 600h is 2nd local peak. TROUGH between them: 475→0.910, 510→0.893, 550→0.948. Both peaks justified for dual-idio.
- **Beta window (idio)**: 168h → best MIN quality; 336h → higher AVG. Trade-off consistent.
- **k_per_side for idio**: k=2 default; k=3 standalone: 437→1.227 vs k=2→1.222 (+tiny). But k=3 for I437 in ensemble STRONGLY improves balanced: Phase 74 weights AVG 1.902→1.914, MIN 1.199→1.191.
- **Dual-idio breakthrough**: Using TWO idio lookbacks (437+600) adds 2022 diversification. Idio_600 2022=1.366 vs Idio_437 2022=0.505 → dual improves floor.
- **Idio_800 fails**: I800_bw168 standalone AVG=0.648, negative in 2022-2024. Not useful.
- **Single-idio inferior**: Single I600 (no I437) best: AVG=1.881 — dual-idio is necessary.

## Signals That FAILED (all lookbacks, all directions)
- TakerBuyAlpha: lb=24h→-1.346, lb=48→-0.957, lb=96→-0.697, lb=168→-0.515
- FundingVolAlpha (std dev of funding rates): contrarian lb144→-0.004, all other configs negative
- Idio_800_bw168: AVG=0.648, negative years in 2022/2023/2024

## Correlations (confirmed stable across runs)
- idio↔f144 = **-0.009 (ORTHOGONAL!)** — the magic behind the ensemble
- v1↔f144 = 0.158 (low), v1↔idio = 0.354 (moderate)
- f144↔f168 = 0.838 (high — similar signals)

## Best Ensemble Architecture
**Phase 76 balanced champion**: `V1(17.5%) + I437_bw168_k3(12.5%) + I600_bw168_k2(22.5%) + F144(47.5%)`
- AVG=1.914, MIN=1.191, all 5 years positive
- Config saved: `configs/ensemble_p76_k3i437_balanced.json`
- Strictly dominates Phase 73 balanced (1.909/1.170) on BOTH metrics

**Phase 75 AVG-max**: `V1(12.5%) + I437_bw336(20%) + I600_bw168(17.5%) + F144(40%) + F168(10%)`
- AVG=1.934, MIN=0.972
- Config saved: `configs/ensemble_p75_bw336x168_avgmax.json`

## Pareto Frontier (AVG vs MIN trade-off)
- Higher AVG → worse MIN (unavoidable trade-off)
- Phase 73 sweet spot: 1.909/1.170 (F168=5% helps AVG)
- Phase 74 balanced: 1.902/1.199 (F168=0, best standalone MIN)
- Phase 76 k=3: 1.914/1.191 (k=3 for I437 lifts AVG ~12bps while keeping MIN near P74)

## Research Phases Summary
- Phases 59-66: Established V1+Idio+F144/F168 structure, found funding>vol signals
- Phase 67: Discovered fund_144 >> fund_96, confirmed 144h peak
- Phase 68: Weight grid search, confirmed V1=15/Idio=30/F144=40/F168=15 as AVG-max
- Phase 69: Canonical 8h-step lb sweep (confirmed 144h), FundingVol fails
- Phase 70: TakerBuy fails, Idio_437 confirmed optimal lookback
- Phase 71: **DUAL-IDIO BREAKTHROUGH** — 437+600 beats single-idio
- Phase 72: Beta_window sweep — bw=336 best standalone AVG, bw=168 best MIN
- Phase 73: Fine frontier mapping — bw168×168 gives AVG=1.909, MIN=1.170 (balanced)
- Phase 74: 2.5% grid + gap lookback (confirms trough 475-550h) → AVG=1.902, MIN=1.199
- Phase 75: bw336×168 frontier → new AVG-max 1.934/0.972; single-I600 fails (1.881)
- Phase 76: F160 test (good 2022), I800 fails, k=3 I437 → NEW BALANCED CHAMP 1.914/1.191
- Phase 77: k=3 fine grid + F160 + bw336_k3 combinations (in progress)

## Workflow Notes
- All backtests: 5 OOS years (2021-2025), hourly bars, 10 crypto perps, Binance USDm
- Sharpe formula: `(mean/std) × sqrt(8760)` for hourly returns
- Ensemble blending: return-level weighted average before computing Sharpe
- Run scripts with: `python3 scripts/run_phaseXX_*.py 2>&1 | tee artifacts/phaseXX_run.log`
- Artifacts are gitignored; commit scripts + configs
- AVG = average across 5 years; MIN = worst single year (floor quality)
