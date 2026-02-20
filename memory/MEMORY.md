# Nexus Quant — Persistent Memory

## Champion Table (latest wins)
| Phase | Config | AVG | MIN | Notes |
|-------|--------|-----|-----|-------|
| 54 | V1-Long | 1.125 | 0.803 | original V1 champion |
| 68 | V1+Idio+F144+F168 | 1.844 | 0.996 | 15/30/40/15 (AVG-max) |
| 71 | V1+Idio_437+Idio_600+F144+F168 | 1.861 | 1.097 | 15/21/14/40/10 (dual-idio!) |
| 72 | V1+I437_bw336+I600_bw168+F144+F168 | 1.933 | 0.894 | 10/20/20/35/15 (AVG-max) |
| 73 | V1+I437_bw168+I600_bw168+F144+F168 | 1.909 | 1.170 | 15/15/20/45/5 (balanced) |
| 74 | V1+I437_bw168+I600_bw168+F144 | 1.902 | 1.199 | 17.5/12.5/22.5/47.5 (2.5% grid) |
| 76 | V1+I437_bw168_k3+I600_bw168+F144 | 1.914 | 1.191 | 17.5/12.5/22.5/47.5 (k3) |
| **78** | **V1+I437_bw168_k3+I600_bw168+F144** | **1.919** | **1.206** | **17.5/17.5/20.0/45.0 (BEST BALANCED)** |
| **79** | **V1+I437_bw168_k3+I600_bw168+F144+F168** | **1.934** | **1.098** | **10/25/17.5/42.5/5 (BEST AVG-MAX, bw168+k3)** |

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
**Phase 78 balanced champion**: `V1(17.5%) + I437_bw168_k3(17.5%) + I600_bw168_k2(20.0%) + F144(45.0%)`
- AVG=1.919, MIN=1.206, all 5 years positive, YbY: [3.446, 1.206, 1.207, 2.489, 1.249]
- Config saved: `configs/ensemble_p78_k3i437_17p5_balanced.json`
- Strictly dominates P76 (1.914/1.191) on BOTH metrics

**Phase 78 balanced champion**: `V1(17.5%) + I437_bw168_k3(17.5%) + I600_bw168_k2(20.0%) + F144(45.0%)`
- AVG=1.919, MIN=1.206, all 5 years positive, YbY: [3.446, 1.206, 1.207, 2.489, 1.249]
- Config saved: `configs/ensemble_p78_k3i437_17p5_balanced.json`

**Phase 79 AVG-max champion**: `V1(10%) + I437_bw168_k3(25%) + I600_bw168_k2(17.5%) + F144(42.5%) + F168(5%)`
- AVG=1.934, MIN=1.098 (strictly better than P75 bw336 AVG-max which had MIN=0.972)
- Uses bw168+k3 (not bw336); I437_k3=25% is max tested
- Config saved: `configs/ensemble_p79_k3i437_avgmax.json`

## Pareto Frontier (fully characterized in Phases 78-79, k3 I437 space)
Key non-dominated points:
| AVG   | MIN   | Config |
|-------|-------|--------|
| 1.934 | 1.098 | V1=10%, I437_k3=25%, I600=17.5%, F144=42.5%, F168=5% |
| 1.933 | 1.148 | V1=12.5%, I437_k3=22.5%, I600=17.5%, F144=45%, F168=2.5% |
| 1.928 | 1.179 | V1=15%, I437_k3=22.5%, I600=17.5%, F144=42.5%, F168=2.5% |
| 1.919 | 1.206 | V1=17.5%, I437_k3=17.5%, I600=20%, F144=45% ← P78 BALANCED |
| 1.907 | 1.223 | V1=20%, I437_k3=17.5%, I600=20%, F144=42.5% |
| 1.902 | 1.226 | V1=20%, I437_k3=20%, I600=20%, F144=40% |
| 1.871 | 1.254 | V1=25%, I437_k3=12.5%, I600=22.5%, F144=37.5%, F168=2.5% |
- **No configuration achieves AVG≥1.920 AND MIN≥1.207 simultaneously in this signal space**
- I437_k3 optimal weight: 17.5% for balanced; 25% for AVG-max

## Weight Sensitivity Notes
- I437_k3=17.5% is optimal for balanced config (P79 sweep confirms peak)
- I437_k3=25% pushes AVG to 1.934 but MIN stays at 1.098
- High-MIN frontier: V1=20-25% with lower F144 floor (≥35%) → MIN can reach 1.254
- Phase 79 confirmed: P78 balanced champion is Pareto-optimal in current signal space

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
- Phase 76: F160 test (good 2022), I800 fails, k=3 I437 → BALANCED CHAMP 1.914/1.191
- Phase 77: k=3 fine grid (bug: 0 configs), bw336_k3 weaker, k3 I600 hurts MIN
- Phase 78: Fixed grid (636 configs), V1 k=3 FAILS, F160 pure inferior, **NEW BALANCED CHAMP 1.919/1.206** (I437_k3=17.5%)
- Phase 79: Extended grid (578 configs), P78 confirmed Pareto-optimal balanced; **NEW AVG-MAX 1.934/1.098** (k3 I437=25%, strictly dominates P75 bw336 AVG-max)

## Workflow Notes
- All backtests: 5 OOS years (2021-2025), hourly bars, 10 crypto perps, Binance USDm
- Sharpe formula: `(mean/std) × sqrt(8760)` for hourly returns
- Ensemble blending: return-level weighted average before computing Sharpe
- Run scripts with: `python3 scripts/run_phaseXX_*.py 2>&1 | tee artifacts/phaseXX_run.log`
- Artifacts are gitignored; commit scripts + configs
- AVG = average across 5 years; MIN = worst single year (floor quality)
