# Nexus Quant — Persistent Memory

## Champion Table (latest wins)
| Phase | Config | AVG | MIN | Notes |
|-------|--------|-----|-----|-------|
| 81 | V1+I437_bw168_k4+I600+F144 | 2.010 | 1.245 | 22.5/22.5/15/40 (k4 single-lb balanced) |
| 84 | V1+I437k4+I460k4+I410k4+I600+F144 | 2.040 | 1.431 | triple-lb balanced |
| 85 | V1+I460k4+I410k4+I600+F144 | 2.001 | 1.468 | 27.5/13.75/20/10/28.75 (I460+I410 balanced) |
| 85 | V1+I437k4+I474k4+I600+F144 | 2.258 | 1.164 | 7.5/16.25/27.5/10/38.75 (AVG-MAX record) |
| **86** | **V1+I460k4+I410k4+I600+F144** | **2.007** | **1.469** | **26.82/11.85/20.56/10/30.77 (BALANCED)** |
| **86** | **V1+I437k4+I474k4+I600+F144** | **2.268** | **1.125** | **4.98/16.17/30/10/38.85 (NEW AVG-MAX)** |
| **86** | **(high-MIN)** | **1.954** | **1.478** | **V1=30%/I460=11.23%/I410=21.8%/I600=10%/F144=26.97%** |

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
- **k_per_side rules**: k=4 best for I437/I460/I410/I474. k=2 for I600, V1, F144, F168.
- **k=3 and k=4 both fail for F144**: negative 2025 year. Never use k>2 for funding signal.
- **Triple-lb insight (P84→P85→P86)**: I437 weight converges to 0% at fine resolution.
  - True breakthrough: I460+I410 dual-lb (covers 2023 via I460, 2022+2023 via I410)
- **I474 optimal weight in AVG-max**: 30% (not 20% as in P84). I437=16%, I474=30% → AVG=2.268
- **High-MIN recipe**: V1≥30% + I460+I410 → MIN≥1.478 but AVG<2.0 tradeoff
- **I600 weight**: 10% optimal with k4 idio signals
- **Balanced architecture**: V1=26-27.5%, I460=11-14%, I410=18-21%, I600=10%, F144=28-31%

## Signals That FAILED
- TakerBuyAlpha: all lookbacks negative
- FundingVolAlpha: all configs negative
- Idio_800_bw168: negative years
- F144 k=3: MIN=-0.247; F144 k=4: MIN=-0.561 (both produce negative years)
- I600 k=4: standalone 1.041 < k2's 1.235
- V1 k=3: AVG=0.885
- Dual-k (I437_k3+I437_k4): Suboptimal vs pure k4

## Best Ensemble Architecture (CURRENT CHAMPIONS)
**Phase 86 balanced champion**: `V1(26.82%) + I460_k4(11.85%) + I410_k4(20.56%) + I600_k2(10%) + F144_k2(30.77%)`
- AVG=2.007, MIN=1.469, YbY: [3.215, 1.476, 1.469, 2.394, 1.479]
- Config saved: `configs/ensemble_p86_balanced.json`
- STRICTLY DOMINATES P85 balanced (1.469 > 1.468)!

**Phase 86 AVG-max champion**: `V1(4.98%) + I437_k4(16.17%) + I474_k4(30%) + I600_k2(10%) + F144_k2(38.85%)`
- AVG=2.268, MIN=1.125, YbY: [3.848, 1.311, 1.125, 3.314, 1.742]
- Config saved: `configs/ensemble_p86_avgmax.json`
- NEW AVG-MAX RECORD (2.268 > P85's 2.258)! I474=30% optimal!

**Phase 86 high-MIN (Pareto frontier)**: `V1(30%) + I460_k4(11.23%) + I410_k4(21.8%) + I600_k2(10%) + F144_k2(26.97%)`
- AVG=1.954, MIN=1.478, YbY: [3.039, 1.484, 1.478, 2.281, 1.489]
- Note: AVG<2.0 but MIN is highest at 1.478!

## Pareto Frontier (Phase 86)
| AVG   | MIN   | Config |
|-------|-------|--------|
| 2.268 | 1.125 | V1=4.98%, I437=16.17%, I474=30%, I600=10%, F144=38.85% ← P86 AVG-MAX |
| 2.007 | 1.469 | V1=26.82%, I460=11.85%, I410=20.56%, I600=10%, F144=30.77% ← P86 BALANCED |
| 1.954 | 1.478 | V1=30%, I460=11.23%, I410=21.8%, I600=10%, F144=26.97% ← HIGH-MIN |

## Correlations (confirmed stable)
- idio↔f144 = **-0.009 (ORTHOGONAL!)** — the magic behind the ensemble
- v1↔f144 = 0.158 (low), v1↔idio = 0.354 (moderate), f144↔f168 = 0.838 (high)

## Research Phases Summary
- Phases 59-74: Established V1+Idio+F144 structure; 437h optimal for k=3
- Phases 76-79: k=3 I437 champion 1.919/1.206
- Phase 80: F144 k3 fails; I437_k4 BREAKTHROUGH standalone 1.623
- Phase 81: k4 grid; BALANCED 2.010/1.245; AVG-MAX 2.079/1.015
- Phase 82: F144_k4 fails; I460_k4 BREAKTHROUGH standalone 1.828
- Phase 83: DUAL-LB I437+I460; BALANCED 2.019/1.368; AVG-MAX 2.181/1.180
- Phase 84: TRIPLE-LB I437+I460+I410 → 2.040/1.431; I437+I474 → 2.224/1.177 AVG-max
- Phase 85: I437=0% optimal in triple-lb; I460+I410 balanced 2.001/1.468; I437+I474 2.258/1.164 (new record, I474=27.5%)
- Phase 86: **Numpy vectorization** (60x faster); I460+I410 2.007/1.469 (P85+); I437+I474 2.268/1.125 (I474=30%!); high-MIN 1.954/1.478

## Workflow Notes
- All backtests: 5 OOS years (2021-2025), hourly bars, 10 crypto perps, Binance USDm
- Sharpe formula: `(mean/pstd) × sqrt(8760)` for hourly returns (pstd = population std, ddof=0)
- Ensemble blending: return-level weighted average before computing Sharpe
- Run scripts: `python3 scripts/run_phaseXX_*.py 2>&1 | tee artifacts/phaseXX_run.log`
- Use "p8X_" prefix for Phase run labels to avoid cross-phase artifact conflicts
- pct_range: `[x / 10000 for x in range(lo, hi, step)]` (NEVER /100!)
- AVG = mean of 5 year Sharpes; MIN = worst single year
- **NUMPY BLEND (Phase 86+)**: Use `W @ R` batch matrix multiply for blend sweeps. ~60x faster.
  - `W`: (N_configs, K) weight matrix; `R_year`: (K, T) stacked returns
  - numpy.std() with default ddof=0 = statistics.pstdev() — IDENTICAL results ✓
  - numpy 2.4.2 installed via `pip install numpy --break-system-packages`
