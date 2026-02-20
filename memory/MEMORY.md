# Nexus Quant — Persistent Memory

## Champion Table (latest wins)
| Phase | Config | AVG | MIN | Notes |
|-------|--------|-----|-----|-------|
| 78 | V1+I437_bw168_k3+I600+F144 | 1.919 | 1.206 | 17.5/17.5/20/45 (k3 balanced) |
| 81 | V1+I437_bw168_k4+I600+F144 | 2.010 | 1.245 | 22.5/22.5/15/40 (k4 single-lb balanced) |
| 83 | V1+I437k4+I460k4+I600+F144 | 2.181 | 1.180 | 10/17.5/17.5/12.5/42.5 (dual-lb AVG-MAX) |
| 84 | V1+I437k4+I460k4+I410k4+I600+F144 | 2.040 | 1.431 | 25/5/12.5/15/10/32.5 (triple-lb balanced) |
| 84 | V1+I437k4+I474k4+I600+F144 | 2.224 | 1.177 | 10/20/20/10/40 (I437+I474 AVG-MAX) |
| **85** | **V1+I460k4+I410k4+I600+F144** | **2.001** | **1.468** | **27.5/13.75/20/10/28.75 (I460+I410 balanced)** |
| **85** | **V1+I437k4+I474k4+I600+F144** | **2.258** | **1.164** | **7.5/16.25/27.5/10/38.75 (NEW AVG-MAX RECORD)** |

## Critical Strategy Name Distinction
- `"funding_momentum_alpha"` = cumulative funding sum (uses `funding_lookback_bars`) ← CORRECT
- `"funding_contrarian_alpha"` = funding_z × momentum_z composite ← WRONG, never use

## Key Confirmed Findings
- **Funding lookback**: 144h (n_samples=18) is CONFIRMED global peak.
- **Idio lookback profiles for k=4** (all bw=168):
  - I437_k4: 2022=1.281, 2023=0.563, 2024=2.398, AVG=1.623 — good 2022
  - I460_k4: 2022=0.635, 2023=1.038, 2024=2.709, AVG=1.828 — good 2023/2024/2025 ★
  - I474_k4: 2022=0.883, 2023=0.974, 2024=2.906, AVG=1.880 — best standalone!
  - I410_k4: 2022=1.613, 2023=1.027, 2024=1.642, AVG=1.567 — best 2022+2023 balance ★★
  - Dual-lb I437+I460: covers 2022 (via I437) AND 2023 (via I460) → massive ensemble boost
  - Dual-lb I437+I474: STRICTLY DOMINATES I437+I460 in dual-lb setting
  - **I460+I410 dual-lb**: covers 2022 (via I410) AND 2023 (via I460) → best balanced combo
  - **I474+I410 dual-lb**: similar but I474 has better 2022 floor → slightly higher AVG
- **k_per_side rules**: k=4 best for I437/I460/I410/I474. k=2 for I600, V1, F144, F168.
- **k=3 and k=4 both fail for F144**: negative 2025 year. Never use k>2 for funding signal.
- **Triple-lb insight (Phase 84→85)**: I437 weight converges to 0% at fine resolution!
  - P84's triple-lb champion (I437=5%) → P85 fine grid → optimal is I437=0%
  - The real advance was I460+I410 dual-lb, not triple-lb per se
- **I474 weight in AVG-max**: Optimal is 27.5% (much higher than P84's 20%)
  - I437(16.25%)+I474(27.5%) → AVG=2.258 (new record)
- **I600 weight**: 10% optimal with k4 idio signals
- **Balanced architecture**: V1=27.5%, one "2023-good" idio, one "2022-good" idio, I600=10%, F144=~30%

## Signals That FAILED
- TakerBuyAlpha: all lookbacks negative
- FundingVolAlpha: all configs negative
- Idio_800_bw168: negative years
- F144 k=3: MIN=-0.247; F144 k=4: MIN=-0.561 (both produce negative years)
- I600 k=4: standalone 1.041 < k2's 1.235
- V1 k=3: AVG=0.885
- Dual-k (I437_k3+I437_k4): Suboptimal vs pure k4

## Best Ensemble Architecture (CURRENT CHAMPIONS)
**Phase 85 balanced champion**: `V1(27.5%) + I460_k4(13.75%) + I410_k4(20%) + I600_k2(10%) + F144_k2(28.75%)`
- AVG=2.001, MIN=1.468, YbY: [3.167, 1.468, 1.470, 2.381, 1.517]
- Config saved: `configs/ensemble_p85_balanced.json`
- STRICTLY DOMINATES P84 balanced (1.468 > 1.431)! I437=0% is optimal!

**Phase 85 AVG-max champion**: `V1(7.5%) + I437_k4(16.25%) + I474_k4(27.5%) + I600_k2(10%) + F144_k2(38.75%)`
- AVG=2.258, MIN=1.164, YbY: [3.825, 1.337, 1.164, 3.250, 1.715]
- Config saved: `configs/ensemble_p85_avgmax.json`
- NEW AVG-MAX RECORD! (2.258 > P84's 2.224)

## Pareto Frontier (Phase 85 balanced region, MIN≥1.43)
| AVG   | MIN   | Config |
|-------|-------|--------|
| 2.258 | 1.164 | V1=7.5%, I437=16.25%, I474=27.5%, I600=10%, F144=38.75% ← P85 AVG-MAX |
| 2.195 | 1.322 | V1=17.5%, I437=10%, I474=27.5%, I600=10%, F144=35% (dual-lb balanced) |
| 2.018 | 1.462 | V1=27.5%, I460=7.5%, I474=10%, I410=17.5%, I600=10%, F144=27.5% (quad-lb) |
| 2.015 | 1.464 | V1=27.5%, I474=13.75%, I410=17.5%, I600=10%, F144=31.25% (I474+I410 balanced) |
| 2.001 | 1.468 | V1=27.5%, I460=13.75%, I410=20%, I600=10%, F144=28.75% ← P85 BALANCED |

Also notable: I474+I410 at 2.015/1.464 — higher AVG, comparable MIN to balanced champion.

## Correlations (confirmed stable)
- idio↔f144 = **-0.009 (ORTHOGONAL!)** — the magic behind the ensemble
- v1↔f144 = 0.158 (low), v1↔idio = 0.354 (moderate), f144↔f168 = 0.838 (high)

## Research Phases Summary
- Phases 59-74: Established V1+Idio+F144 structure; 437h optimal for k=3
- Phases 76-79: k=3 I437 champion 1.919/1.206; AVG-max 1.934/1.098
- Phase 80: F144 k3 fails; **I437_k4 BREAKTHROUGH** standalone 1.623
- Phase 81: k4 grid; **BALANCED 2.010/1.245**; AVG-MAX 2.079/1.015
- Phase 82: Fine grid; F144_k4 fails; **I460_k4 BREAKTHROUGH** standalone 1.828; high-MIN 1.349
- Phase 83: **DUAL-LB BREAKTHROUGH**: I437+I460; **BALANCED 2.019/1.368** (strictly dominates P81!); **AVG-MAX 2.181/1.180**; I410_k4 also strong
- Phase 84: **TRIPLE-LB**: I437+I460+I410 → 2.040/1.431; I437+I474 dual-lb → 2.224/1.177 AVG-max
- Phase 85: **FINE-GRID**: I437=0% optimal in triple-lb; **I460+I410 balanced 2.001/1.468** (≡ pure dual-lb!); **I437+I474 AVG-MAX 2.258/1.164** (new record, I474=27.5%!)

## Workflow Notes
- All backtests: 5 OOS years (2021-2025), hourly bars, 10 crypto perps, Binance USDm
- Sharpe formula: `(mean/std) × sqrt(8760)` for hourly returns
- Ensemble blending: return-level weighted average before computing Sharpe
- Run scripts: `python3 scripts/run_phaseXX_*.py 2>&1 | tee artifacts/phaseXX_run.log`
- Use "p8X_" prefix for Phase run labels to avoid cross-phase artifact conflicts
- pct_range: `[x / 10000 for x in range(lo, hi, step)]` (NEVER /100!)
- AVG = mean of 5 year Sharpes; MIN = worst single year
- **PERFORMANCE**: blend_ensemble is slow in pure Python (~30min for 10K configs). Next phases must use numpy vectorization (W @ R matrix multiply) for ~100x speedup. See Solution 2 in phase85 analysis.
