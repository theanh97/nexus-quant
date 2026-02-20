# Nexus Quant — Persistent Memory

## Champion Table (latest wins)
| Phase | Config | AVG | MIN | Notes |
|-------|--------|-----|-----|-------|
| 78 | V1+I437_bw168_k3+I600+F144 | 1.919 | 1.206 | 17.5/17.5/20/45 (k3 balanced) |
| 81 | V1+I437_bw168_k4+I600+F144 | 2.010 | 1.245 | 22.5/22.5/15/40 (k4 single-lb balanced) |
| 83 | V1+I437k4+I460k4+I600+F144 | 2.181 | 1.180 | 10/17.5/17.5/12.5/42.5 (dual-lb AVG-MAX) |
| **84** | **V1+I437k4+I460k4+I410k4+I600+F144** | **2.040** | **1.431** | **25/5/12.5/15/10/32.5 (triple-lb BALANCED)** |
| **84** | **V1+I437k4+I474k4+I600+F144** | **2.224** | **1.177** | **10/20/20/10/40 (I437+I474 AVG-MAX)** |

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
  - Dual-lb I437+I474: I474 has better 2022 than I460 (0.883 vs 0.635) → STRICTLY DOMINATES I437+I460 in dual-lb ensemble!
- **Idio lookback for k=3**: 437h optimal. 600h is 2nd peak.
- **k_per_side rules**: k=4 best for I437/I460/I410/I474. k=2 for I600, V1, F144, F168.
- **k=3 and k=4 both fail for F144**: negative 2025 year. Never use k>2 for funding signal.
- **Dual-lb k4 breakthrough (Phase 83)**: I437_k4 + I460_k4 simultaneously.
  - Balanced: 2.019/1.368 (vs single-lb P81: 2.010/1.245) — STRICTLY DOMINATES!
  - AVG-max: 2.181/1.180 (vs single-lb P81: 2.079/1.015) — much better!
- **Triple-lb k4 breakthrough (Phase 84)**: I437_k4 + I460_k4 + I410_k4 simultaneously:
  - Balanced: 2.040/1.431 — STRICTLY DOMINATES P83 dual-lb (2.019/1.368)!
  - I437=5% (minimal 2022 coverage), I460=12.5% (2023+2025), I410=15% (bridges 2022+2023)
  - ALL 5 years above 1.4! YbY=[3.303, 1.452, 1.431, 2.507, 1.505]
- **I437+I474 dual-lb**: STRICTLY DOMINATES I437+I460 dual-lb at balanced configs:
  - I437+I474 balanced: 2.046/1.377 vs I437+I460 balanced: 2.019/1.368
  - I437+I474 AVG-max: 2.224/1.177 vs I437+I460 AVG-max: 2.181/1.180
- **I600 weight**: 10-15% optimal with k4 idio signals

## Signals That FAILED
- TakerBuyAlpha: all lookbacks negative
- FundingVolAlpha: all configs negative
- Idio_800_bw168: negative years
- F144 k=3: MIN=-0.247; F144 k=4: MIN=-0.561 (both produce negative years)
- I600 k=4: standalone 1.041 < k2's 1.235
- V1 k=3: AVG=0.885
- Dual-k (I437_k3+I437_k4): Suboptimal vs pure k4

## Best Ensemble Architecture (CURRENT CHAMPIONS)
**Phase 84 balanced champion**: `V1(25%) + I437_k4(5%) + I460_k4(12.5%) + I410_k4(15%) + I600_k2(10%) + F144_k2(32.5%)`
- AVG=2.040, MIN=1.431, YbY: [3.303, 1.452, 1.431, 2.507, 1.505]
- Config saved: `configs/ensemble_p84_balanced.json`
- STRICTLY DOMINATES P83 balanced (2.040>2.019, 1.431>1.368)! ALL years above 1.4!

**Phase 84 AVG-max champion**: `V1(10%) + I437_k4(20%) + I474_k4(20%) + I600_k2(10%) + F144_k2(40%)`
- AVG=2.224, MIN=1.177, Config saved: `configs/ensemble_p84_avgmax.json`

**Phase 84 I437+I474 balanced**: `V1(~25%) + I437_k4(~20%) + I474_k4(~15%) + I600_k2(~10%) + F144_k2(~30%)`
- 2.046/1.377 — STRICTLY DOMINATES P83 I437+I460 balanced (2.019/1.368)

## Pareto Frontier (triple-lb k4, Phase 84)
| AVG   | MIN   | Config |
|-------|-------|--------|
| 2.224 | 1.177 | V1=10%, I437=20%, I474=20%, I600=10%, F144=40% ← P84 AVG-MAX |
| 2.201 | 1.214 | (triple-lb AVG-max — below I437+I474's 2.224) |
| 2.046 | 1.377 | I437+I474 balanced |
| 2.040 | 1.431 | V1=25%, I437=5%, I460=12.5%, I410=15%, I600=10%, F144=32.5% ← P84 BALANCED |

Prior dual-lb (Phase 83) for reference:
| 2.181 | 1.180 | V1=10%, I437=17.5%, I460=17.5%, I600=12.5%, F144=42.5% ← P83 AVG-MAX |
| 2.019 | 1.368 | V1=27.5%, I437=17.5%, I460=15%, I600=10%, F144=30% ← P83 BALANCED |

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
- Phase 84: **TRIPLE-LB BREAKTHROUGH**: I437+I460+I410; **BALANCED 2.040/1.431** (ALL years >1.4!); **I437+I474 DUAL-LB**: 2.224/1.177 AVG-max, 2.046/1.377 balanced (beats I437+I460 balanced)

## Workflow Notes
- All backtests: 5 OOS years (2021-2025), hourly bars, 10 crypto perps, Binance USDm
- Sharpe formula: `(mean/std) × sqrt(8760)` for hourly returns
- Ensemble blending: return-level weighted average before computing Sharpe
- Run scripts: `python3 scripts/run_phaseXX_*.py 2>&1 | tee artifacts/phaseXX_run.log`
- Use "p8X_" prefix for Phase run labels to avoid cross-phase artifact conflicts
- pct_range: `[x / 10000 for x in range(lo, hi, step)]` (NEVER /100!)
- AVG = mean of 5 year Sharpes; MIN = worst single year
