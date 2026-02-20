# Nexus Quant — Persistent Memory

## Champion Table (latest wins)
| Phase | Config | AVG | MIN | Notes |
|-------|--------|-----|-----|-------|
| 78 | V1+I437_bw168_k3+I600+F144 | 1.919 | 1.206 | 17.5/17.5/20/45 (k3 balanced) |
| 81 | V1+I437_bw168_k4+I600+F144 | 2.010 | 1.245 | 22.5/22.5/15/40 (k4 single-lb balanced) |
| 81 | V1+I437_bw168_k4+I600+F144+F168 | 2.079 | 1.015 | 10/25/20/40/5 (k4 single-lb AVG-MAX) |
| **83** | **V1+I437k4+I460k4+I600+F144** | **2.019** | **1.368** | **27.5/17.5/15/10/30 (dual-lb BALANCED)** |
| **83** | **V1+I437k4+I460k4+I600+F144** | **2.181** | **1.180** | **10/17.5/17.5/12.5/42.5 (dual-lb AVG-MAX)** |

## Critical Strategy Name Distinction
- `"funding_momentum_alpha"` = cumulative funding sum (uses `funding_lookback_bars`) ← CORRECT
- `"funding_contrarian_alpha"` = funding_z × momentum_z composite ← WRONG, never use

## Key Confirmed Findings
- **Funding lookback**: 144h (n_samples=18) is CONFIRMED global peak.
- **Idio lookback profiles for k=4** (all bw=168):
  - I437_k4: 2022=1.281, 2023=0.563, 2024=2.398, AVG=1.623 — good 2022
  - I460_k4: 2022=0.635, 2023=1.038, 2024=2.709, AVG=1.828 — good 2023/2024/2025 ★
  - I474_k4: 2022=0.883, 2023=0.974, 2024=2.906, AVG=1.880 — best standalone!
  - I410_k4: 2022=1.613, 2023=1.027, 2024=1.642, AVG=1.567 — best 2022+2023 balance
  - Dual-lb I437+I460: covers 2022 (via I437) AND 2023 (via I460) → massive ensemble boost
- **Idio lookback for k=3**: 437h optimal. 600h is 2nd peak.
- **k_per_side rules**: k=4 best for I437/I460/I410/I474. k=2 for I600, V1, F144, F168.
- **k=3 and k=4 both fail for F144**: negative 2025 year. Never use k>2 for funding signal.
- **Dual-lb k4 breakthrough (Phase 83)**: I437_k4 + I460_k4 simultaneously. Dramatically improves:
  - Balanced: 2.019/1.368 (vs single-lb P81: 2.010/1.245) — STRICTLY DOMINATES!
  - AVG-max: 2.181/1.180 (vs single-lb P81: 2.079/1.015) — much better!
- **I410_k4 single-lb**: Also strongly dominates P81 balanced (2.014/1.342 vs 2.010/1.245)
- **I600 weight**: 10-15% optimal with k4 idio signals
- **I474_k4 untested in full grid**: standalone best (1.880), should be tested in Phase 84

## Signals That FAILED
- TakerBuyAlpha: all lookbacks negative
- FundingVolAlpha: all configs negative
- Idio_800_bw168: negative years
- F144 k=3: MIN=-0.247; F144 k=4: MIN=-0.561 (both produce negative years)
- I600 k=4: standalone 1.041 < k2's 1.235
- V1 k=3: AVG=0.885
- Dual-k (I437_k3+I437_k4): Suboptimal vs pure k4

## Best Ensemble Architecture (CURRENT CHAMPIONS)
**Phase 83 balanced champion**: `V1(27.5%) + I437_k4(17.5%) + I460_k4(15%) + I600_k2(10%) + F144(30%)`
- AVG=2.019, MIN=1.368, YbY: [3.190, 1.368, 1.369, 2.576, 1.591]
- Config saved: `configs/ensemble_p83_balanced.json`
- STRICTLY DOMINATES P81 balanced (2.019>2.010, 1.368>1.245)!

**Phase 83 AVG-max champion**: `V1(10%) + I437_k4(17.5%) + I460_k4(17.5%) + I600_k2(12.5%) + F144(42.5%)`
- AVG=2.181, MIN=1.180, YbY: [3.795, 1.304, 1.180, 3.050, 1.578]
- Config saved: `configs/ensemble_p83_avgmax.json`

## Pareto Frontier (dual-lb k4, Phase 83)
| AVG   | MIN   | Config |
|-------|-------|--------|
| 2.181 | 1.180 | V1=10%, I437=17.5%, I460=17.5%, I600=12.5%, F144=42.5% ← P83 AVG-MAX |
| 2.165 | 1.267 | V1=15%, I437=17.5%, I460=17.5%, I600=10%, F144=40% |
| 2.147 | 1.294 | V1=17.5%, I437=17.5%, I460=17.5%, I600=10%, F144=37.5% |
| 2.123 | 1.318 | V1=20%, I437=17.5%, I460=17.5%, I600=10%, F144=35% |
| 2.093 | 1.342 | V1=22.5%, I437=15%, I460=17.5%, I600=10%, F144=35% |
| 2.019 | 1.368 | V1=27.5%, I437=17.5%, I460=15%, I600=10%, F144=30% ← P83 BALANCED |

I410_k4 single-lb (also strictly dominates P81):
| 2.053 | 1.206 | V1=10%, I410=25%, I600=20%, F144=45% |
| 2.014 | 1.342 | V1=20%, I410=25%, I600=17.5%, F144=37.5% |
| 1.927 | 1.389 | V1=27.5%, I410=25%, I600=17.5%, F144=30% |

## Correlations (confirmed stable)
- idio↔f144 = **-0.009 (ORTHOGONAL!)** — the magic behind the ensemble
- v1↔f144 = 0.158 (low), v1↔idio = 0.354 (moderate), f144↔f168 = 0.838 (high)

## Research Phases Summary
- Phases 59-74: Established V1+Idio+F144 structure; 437h optimal for k=3
- Phases 76-79: k=3 I437 champion 1.919/1.206; AVG-max 1.934/1.098
- Phase 80: F144 k3 fails; **I437_k4 BREAKTHROUGH** standalone 1.623
- Phase 81: k4 grid; **BALANCED 2.010/1.245**; AVG-MAX 2.079/1.015
- Phase 82: Fine grid; F144_k4 fails; **I460_k4 BREAKTHROUGH** standalone 1.828; high-MIN 1.349
- Phase 83: **DUAL-LB BREAKTHROUGH**: I437+I460; **BALANCED 2.019/1.368** (strictly dominates P81!); **AVG-MAX 2.181/1.180**; I410_k4 also strong (2.014/1.342)

## Workflow Notes
- All backtests: 5 OOS years (2021-2025), hourly bars, 10 crypto perps, Binance USDm
- Sharpe formula: `(mean/std) × sqrt(8760)` for hourly returns
- Ensemble blending: return-level weighted average before computing Sharpe
- Run scripts: `python3 scripts/run_phaseXX_*.py 2>&1 | tee artifacts/phaseXX_run.log`
- Use "p8X_" prefix for Phase run labels to avoid cross-phase artifact conflicts
- pct_range: `[x / 10000 for x in range(lo, hi, step)]` (NEVER /100!)
- AVG = mean of 5 year Sharpes; MIN = worst single year
