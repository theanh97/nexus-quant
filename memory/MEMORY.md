# Nexus Quant — Persistent Memory

## Champion Table (latest wins)
| Phase | Config | AVG | MIN | Notes |
|-------|--------|-----|-----|-------|
| 54 | V1-Long | 1.125 | 0.803 | original V1 champion |
| 78 | V1+I437_bw168_k3+I600_bw168+F144 | 1.919 | 1.206 | 17.5/17.5/20.0/45.0 (k3 balanced) |
| 79 | V1+I437_bw168_k3+I600_bw168+F144+F168 | 1.934 | 1.098 | 10/25/17.5/42.5/5 (k3 AVG-max) |
| 81 | V1+I437_bw168_k4+I600_bw168+F144 | 2.010 | 1.245 | 22.5/22.5/15/40 (k4 BALANCED) |
| 81 | V1+I437_bw168_k4+I600_bw168+F144+F168 | 2.079 | 1.015 | 10/25/20/40/5 (k4 AVG-MAX) |
| **82** | **V1+I437_bw168_k4+I600_bw168+F144** | **2.000** | **1.292** | **25/26.25/10/38.75 (k4 HIGH-MIN)** |
| **82** | **V1+I437_bw168_k4+I600_bw168+F144** | **1.901** | **1.349** | **32.5/22.5/10/35 (k4 BEST-MIN)** |

## Critical Strategy Name Distinction
- `"funding_momentum_alpha"` = cumulative funding sum (uses `funding_lookback_bars`) ← CORRECT
- `"funding_contrarian_alpha"` = funding_z × momentum_z composite (uses `momentum_lookback_bars`) ← WRONG
- NEVER use `"funding_contrarian_alpha"` for funding contrarian ensemble signals

## Key Confirmed Findings
- **Funding lookback**: 144h (n_samples=18) is CONFIRMED global peak. Curve: 96→1.031, 136→1.115, **144→1.302**, 160→1.252, 168→1.197
- **Idio lookback for k=4**: **460h** is optimal standalone (AVG=1.828 vs 437h's 1.623!). Better 2023 (1.038 vs 0.563) and 2025 (2.162 vs 1.834), but worse 2022 (0.635 vs 1.281). 437h for k=3 was optimal but k=4 shifts the peak right to 460h.
- **Idio lookback for k=3**: 437h CONFIRMED optimal. 600h is 2nd local peak. Trough at 475-550h.
- **Beta window (idio)**: 168h → best MIN quality; 336h → higher AVG. Trade-off consistent.
- **k_per_side for I437**: k=4 dramatically outperforms k=3 in ensemble! k4 standalone: AVG=1.623 (vs k3=1.227). k4 balanced champion STRICTLY DOMINATES k3 balanced on BOTH AVG and MIN.
- **k_per_side for I460**: k=4 gives AVG=1.828 standalone. In P81 weights, I460_k4 → 2.067/1.222 (vs I437_k4 → 2.010/1.245). Trade-off: I460 better AVG but worse 2022 floor.
- **k_per_side for I600**: k=2 ONLY. k=4 for I600 fails (standalone 1.041 vs k2's 1.235).
- **k_per_side for V1**: k=2 ONLY. k=3 severely fails (0.885 vs 1.125).
- **k_per_side for F144**: k=2 ONLY. k=3 fails (MIN=-0.247); k=4 also fails (2025=-0.561, negative!).
- **Dual-idio breakthrough**: Using TWO idio lookbacks (437+600) adds 2022 diversification.
- **I600 weight in k4 ensemble**: 10-15% optimal (lower I600 leaves more room for high-V1 configs with better MIN).
- **High-MIN configs**: V1 must be raised to 25-32.5% to push MIN above 1.260. V1=32.5% gives MIN=1.349 (best-MIN, AVG=1.901).

## Signals That FAILED (all lookbacks, all directions)
- TakerBuyAlpha: lb=24h→-1.346, lb=48→-0.957, lb=96→-0.697, lb=168→-0.515
- FundingVolAlpha (std dev of funding rates): all configs negative
- Idio_800_bw168: AVG=0.648, negative years in 2022/2023/2024
- F144 k=3: AVG=1.226, MIN=-0.247 (2025 negative!) — do not use
- F144 k=4: AVG=1.269, MIN=-0.561 (2025 negative, even worse!) — do not use
- I600 k=4: standalone 1.041 vs k2 1.235 — do not use
- V1 k=3: AVG=0.885 — do not use
- Dual-k (I437_k3+I437_k4): Suboptimal. Best 1.962/1.192 — dominated by pure k4.

## Correlations (confirmed stable across runs)
- idio↔f144 = **-0.009 (ORTHOGONAL!)** — the magic behind the ensemble
- v1↔f144 = 0.158 (low), v1↔idio = 0.354 (moderate)
- f144↔f168 = 0.838 (high — similar signals)

## Best Ensemble Architecture
**Phase 82 high-MIN champion (AVG≥2.0)**: `V1(25%) + I437_bw168_k4(26.25%) + I600_bw168_k2(10%) + F144(38.75%)`
- AVG=2.000, MIN=1.292, all 5 years positive, YbY: [3.295, 1.423, 1.292, 2.603, 1.387]
- Config saved: `configs/ensemble_p82_k4i437_balanced.json`
- Higher MIN than P81 balanced (+0.047) but slightly lower AVG (-0.010)

**Phase 82 absolute best-MIN**: `V1(32.5%) + I437_bw168_k4(22.5%) + I600_bw168_k2(10%) + F144(35%)`
- AVG=1.901, MIN=1.349, all 5 years positive, YbY: [3.017, 1.390, 1.361, 2.389, 1.349]
- Config saved: `configs/ensemble_p82_k4i437_highmin.json`

**Phase 81 balanced champion (I437_k4)**: `V1(22.5%) + I437_bw168_k4(22.5%) + I600_bw168_k2(15%) + F144(40%)`
- AVG=2.010, MIN=1.245, Config: `configs/ensemble_p81_k4i437_balanced.json`

**Phase 81 AVG-max champion**: `V1(10%) + I437_bw168_k4(25%) + I600_bw168_k2(20%) + F144(40%) + F168(5%)`
- AVG=2.079, MIN=1.015, Config: `configs/ensemble_p81_k4i437_avgmax.json`

**Phase 82 I460_k4 (untested full grid)**: In P81 weights, I460_k4 → AVG=2.067, MIN=1.222
- Key: better AVG than I437 but worse 2022 (0.635 vs 1.281). 2023 is much better (1.038 vs 0.563).

## Pareto Frontier (k4 signal space, updated Phase 82)
I437_k4 based (all I600=k2):
| AVG   | MIN   | Config |
|-------|-------|--------|
| 2.079 | 1.015 | V1=10%, I437_k4=25%, I600=20%, F144=40%, F168=5% ← P81 AVG-MAX |
| 2.068 | 1.191 | V1=17.5%, I437_k4=27.5%, I600=12.5%, F144=42.5% |
| 2.029 | 1.264 | V1=22.5%, I437_k4=27.5%, I600=10%, F144=40% |
| **2.010** | **1.245** | **V1=22.5%, I437_k4=22.5%, I600=15%, F144=40% ← P81 BALANCED** |
| 2.000 | 1.292 | V1=25%, I437_k4=26.25%, I600=10%, F144=38.75% ← P82 HIGH-MIN |
| 1.995 | 1.297 | V1=25%, I437_k4=25%, I600=10%, F144=40% |
| 1.989 | 1.302 | V1=25%, I437_k4=23.75%, I600=10%, F144=41.25% |
| 1.932 | 1.335 | V1=30%, I437_k4=22.5%, I600=10%, F144=37.5% |
| 1.901 | 1.349 | V1=32.5%, I437_k4=22.5%, I600=10%, F144=35% ← P82 BEST-MIN |

- **k4 I437 Pareto: AVG range 1.901-2.079, MIN range 1.015-1.349**
- I460_k4 (Phase 82 D): P81 weights → 2.067/1.222 (non-dominated; Phase 83 will map full grid)
- k3 Pareto for reference: AVG range 1.871-1.934, MIN range 1.098-1.254 (fully dominated by k4)

## Weight Sensitivity Notes (k4 space)
- I437_k4 optimal weight: ~22.5% for balanced; 25% for AVG-max
- V1 optimal: 22.5% for balanced (higher than k3-era's 17.5%), 10-12.5% for AVG-max
- I600 optimal: 15% in k4 ensemble (lower than k2-era's 20%)
- F144 optimal: 40-45% (consistent across k2/k3/k4 eras)
- F168 adds marginal AVG at cost of MIN — use only in AVG-max configs (2.5-5%)
- Phase 81 confirmed: P81 balanced (2.010/1.245) is Pareto-optimal in current signal space

## Research Phases Summary
- Phases 59-66: Established V1+Idio+F144/F168 structure, found funding>vol signals
- Phase 67: Discovered fund_144 >> fund_96, confirmed 144h peak
- Phase 68: Weight grid search, confirmed V1=15/Idio=30/F144=40/F168=15 as AVG-max
- Phase 70: TakerBuy fails, Idio_437 confirmed optimal lookback
- Phase 71: **DUAL-IDIO BREAKTHROUGH** — 437+600 beats single-idio
- Phase 72: Beta_window sweep — bw=336 best standalone AVG, bw=168 best MIN
- Phase 74: 2.5% grid + gap lookback (confirms trough 475-550h) → AVG=1.902, MIN=1.199
- Phase 76: k=3 I437 → BALANCED CHAMP 1.914/1.191
- Phase 77: k=3 fine grid (bug: 0 configs), bw336_k3 weaker, k3 I600 hurts MIN
- Phase 78: Fixed grid (636 configs), V1 k=3 FAILS, F160 pure inferior, **NEW BALANCED CHAMP 1.919/1.206**
- Phase 79: Extended grid (578 configs), P78 confirmed Pareto-optimal; **AVG-MAX 1.934/1.098** (k3 I437=25%)
- Phase 80: F144 k3 fails (MIN=-0.247!); **I437_k4 BREAKTHROUGH** standalone 1.623; P78+k4 → **2.015/1.168 (FIRST ABOVE 2.0!)**
- Phase 81: Full k4 grid (719 configs); k4 Pareto frontier found; **k4 BALANCED CHAMP 2.010/1.245** (strictly dominates P78!); **k4 AVG-MAX 2.079/1.015**; Dual-k suboptimal; I600_k4 fails
- Phase 82: Fine grid (778 configs) + high-MIN push + F144_k4 + lookback sweep. F144_k4 FAILS. **I460_k4 BREAKTHROUGH** (AVG=1.828 standalone > I437's 1.623; 2023=1.038!). **NEW HIGH-MIN CHAMP 2.000/1.292** (V1=25%, I437_k4=26.25%). **BEST-MIN 1.901/1.349** (V1=32.5%). I460 in P81 weights → 2.067/1.222 (Phase 83 will map full I460 grid).

## Workflow Notes
- All backtests: 5 OOS years (2021-2025), hourly bars, 10 crypto perps, Binance USDm
- Sharpe formula: `(mean/std) × sqrt(8760)` for hourly returns
- Ensemble blending: return-level weighted average before computing Sharpe
- Run scripts with: `python3 scripts/run_phaseXX_*.py 2>&1 | tee artifacts/phaseXX_run.log`
- Artifacts are gitignored; commit scripts + configs
- AVG = average across 5 years; MIN = worst single year (floor quality)
- pct_range: `[x / 10000 for x in range(lo_10ths, hi_10ths_exclusive, step_10ths)]` (NEVER /100!)
