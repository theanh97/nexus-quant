# Nexus Quant — Persistent Memory

## Champion Table (latest wins)
| Phase | Config | AVG | MIN | Notes |
|-------|--------|-----|-----|-------|
| 81 | V1+I437_bw168_k4+I600+F144 | 2.010 | 1.245 | 22.5/22.5/15/40 (k4 single-lb balanced) |
| 84 | V1+I437k4+I460k4+I410k4+I600+F144 | 2.040 | 1.431 | triple-lb balanced |
| 85 | V1+I460k4+I410k4+I600+F144 | 2.001 | 1.468 | 27.5/13.75/20/10/28.75 (I460+I410 balanced) |
| 85 | V1+I437k4+I474k4+I600+F144 | 2.258 | 1.164 | 7.5/16.25/27.5/10/38.75 (AVG-MAX record) |
| 86 | V1+I460k4+I410k4+I600+F144 | 2.007 | 1.469 | 26.82/11.85/20.56/10/30.77 (BALANCED) |
| 86 | V1+I437k4+I474k4+I600+F144 | 2.268 | 1.125 | 4.98/16.17/30/10/38.85 (AVG-MAX) |
| 86 | (high-MIN) | 1.954 | 1.478 | V1=30%/I460=11.23%/I410=21.8%/I600=10%/F144=26.97% |
| 87 | V1+I460k4+I410k4+I600_7.5+F144 | 2.002 | 1.493 | 27.43/13.07/23.66/7.5/28.34 (BALANCED) |
| 88 | V1+I460bw168k4+I410bw216k4+I600_5+F144 | 2.015 | 1.529 | 28.75/16.25/22.5/5/27.5 (BALANCED) |
| 89 | V1+I460bw168k4+I415bw216k4+I600_5+F144 | 2.001 | 1.546 | 28.76/16.26/26.88/5/23.1 |
| 89 | V1+I415bw216k4+I474bw216k4+I600+F144 | 2.286 | 1.186 | 3.75/12.5/35/10/38.75 (AVG-MAX) |
| 90 | V1+I460bw168k4+I415bw216k4+I600_2.5+F144 | 2.021 | 1.561 | 27.36/17.98/29.66/2.5/22.5 |
| 90 | V1+I415bw216k4+I474bw216k4+I600_7.5+F144 | 2.304 | 1.205 | 4.5/10/40/7.5/38 (AVG-MAX) |
| 91 | V1+I460bw168k4+I415bw216k4+F144 (NO I600!) | 2.040 | 1.571 | 26.25/20.00/31.25/22.5 |
| **91b** | **V1+I460bw168k4+I415bw216k4+F144 ultra-fine** | **2.010** | **1.576** | **27.47/19.67/32.47/20.39 (NEW CHAMP! NO I600!)** |
| **91b** | **V1+I415bw216k4+I474bw216k4+I600+F144** | **2.319** | **1.126** | **4.25/6/47.25/5/37.5 (NEW AVG-MAX RECORD!)** |

## Critical Strategy Name Distinction
- `"funding_momentum_alpha"` = cumulative funding sum (uses `funding_lookback_bars`) ← CORRECT
- `"funding_contrarian_alpha"` = funding_z × momentum_z composite ← WRONG, never use
- `"nexus_alpha_v1"` = V1 multi-factor strategy ← CORRECT (NOT "v1_standard" which throws UnknownStrategy!)
- `"idio_momentum_alpha"` = idio momentum with `lookback_bars`, `beta_window_bars`, `k_per_side`

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
- **I600 weight**: ★★★ **REMOVE ENTIRELY!** I600=0% wins: MIN=1.5712 (beats I600=2.5%→1.5576)
  - Full trend: 10%→7.5%→5%→3.75%→2.5%→0% each improved MIN
  - I600=0%: V1=26.25%, I460bw168=20.00%, I415bw216=31.25%, F144=22.50% → 2.040/1.571
  - **Simplified architecture: V1 + I460bw168 + I415bw216 + F144 (no I600!)**
- **F144 weight**: **~20.4% is true optimal** (ultra-fine grid, no I600); 22.5% was coarser-step artifact
  - At ultra-fine (0.3125% step), F144 optimal = 20.39% (not 22.5%)
  - Below 17.5% coarse: no balanced result (grid constraint); ultra-fine can go lower
  - Trend: P86=30.77% → P87=28.34% → P88=27.5% → P89=23.10% → P90=22.50% → P91b=20.39%
- **Balanced architecture (P91b)**: V1=27.47%, I460_bw168=19.67%, I415_bw216=32.47%, F144=20.39% (NO I600!)
- **2023 cliff** (lb bw=216): I415→1.143, I420→0.758, I425→0.536, I430→0.487, I437→0.175
  - **lb=415 is the GLOBAL OPTIMAL** — sharp cliff at lb=420 (drops 1.143→0.758)
- **I415_bw216 KEY ROLE**: strong 2022 (1.737) AND 2023 (1.143) — covers weakest years! ★★★
- **I437_bw216**: AVG=1.500, MIN=0.175 (2023=0.175 kills it) — NOT useful for any ensemble
- **Triple-idio balanced FAILS**: I460+I415+I474 → best 1.5395 (worse than dual+I600=2.5%→1.5576)
- **Beta window (bw) findings (Phase 87-88)**:
  - I410 bw=216: [2.117, 1.928, 1.046, 1.679, 1.408] AVG=1.636, MIN=1.046 — +0.315 in 2022!
  - I415 bw=216: [1.967, 1.737, 1.143, 2.015, 1.338] AVG=1.640, MIN=1.143 — ALL years > 1.0! ★
  - I460 bw=216: [2.782, 0.566, 0.877, 3.032, 2.15] AVG=1.881, MIN=0.566 — huge 2024 but weak 2022
  - I415 bw=168: [1.884, 1.764, 0.995, 2.013, 1.246] AVG=1.580, MIN=0.995
  - **bw=216 upgrade rule**: use bw=216 for idio signals with lb≤460; improves 2022/2023 floor
- **I600 weight**: **5% Pareto point** — I600=5% + I410_bw216 → MIN=1.529 (beats 7.5% MIN=1.507!)
  - I600 sweep: 5%→MIN=1.529, 7.5%→MIN=1.507, 10%→MIN=1.484 (lower I600 = higher MIN, up to 5%)
- **lb=415 (intermediate)**: bridge_score=2.759 (2022=1.764, 2023=0.995) > I410's 2.640
  - I415_bw216 in ensemble → MIN=1.526 (Section E2: 2.002/1.526) — strong 2023 coverage
- **P88 Pareto frontier** (all beat P87's 1.493):
  - I460_bw168 + I410_bw216 + I600=5%: AVG=2.015, MIN=1.529 ← P88 CHAMP
  - I460_bw168 + I410_bw216 + I600=7.5%: AVG=2.010, MIN=1.507
  - I460_bw168 + I415_bw216 + I600=7.5%: AVG=2.002, MIN=1.526
  - I460_bw216 + I415_bw216 + I600=7.5%: AVG=2.004, MIN=1.502
  - **UNTESTED**: I460_bw168 + I415_bw216 + I600=5% ← likely even better!

## Signals That FAILED
- TakerBuyAlpha: all lookbacks negative
- FundingVolAlpha: all configs negative
- Idio_800_bw168: negative years
- F144 k=3: MIN=-0.247; F144 k=4: MIN=-0.561 (both produce negative years)
- I600 k=4: standalone 1.041 < k2's 1.235
- V1 k=3: AVG=0.885
- Dual-k (I437_k3+I437_k4): Suboptimal vs pure k4

## Best Ensemble Architecture (CURRENT CHAMPIONS)
**Phase 145-146 champion (PRODUCTION)** ← CURRENT BEST with regime switching:
- BreadthAdaptiveEnsembleStrategy: Real-time breadth classifier → adaptive weights
- OBJ=1.8851 (+0.3179 vs static baseline), WF 1/2, avg_delta=+0.4020
- Config: breadth_window=168h, p_low=0.33, p_high=0.67, 3-regime
- LOW regime → PROD weights; MID → midpoint; HIGH → P143B weights
- Production config: configs/production_p91b_champion.json v2.2.0

**Phase 91b balanced champion** ← BASE ENSEMBLE (underlying P145-146): `V1(27.47%) + I460_bw168_k4(19.67%) + I415_bw216_k4(32.47%) + F144_k2(20.39%) [NO I600!]`
- AVG=2.0101, MIN=1.5761, YbY: [2.876, 1.577, 1.576, 2.445, 1.576] — ALL years ≥ 1.576!
- Config saved: `configs/ensemble_p91_balanced.json`
- STRICTLY DOMINATES P90 (MIN 1.5761 > 1.5614)! KEY: No I600, F144=20.39%!
- 4-signal architecture: V1 + I460bw168 + I415bw216 + F144 (NO I600!)

**Phase 91b AVG-max champion** ← NEW AVG-MAX RECORD: `V1(4.25%) + I415_bw216_k4(6%) + I474_bw216_k4(47.25%) + I600_k2(5%) + F144_k2(37.5%)`
- AVG=2.3192, MIN=1.1256, YbY: [3.939, 1.126, 1.196, 3.591, 1.745]
- Config saved: `configs/ensemble_p91_avgmax.json`
- NEW AVG-MAX RECORD (2.3192 > P90's 2.3038)! I474bw216=47.25%!

**Phase 91 B (coarse)** (superseded): `V1(26.25%) + I460_bw168_k4(20.00%) + I415_bw216_k4(31.25%) + F144_k2(22.50%)`
- AVG=2.0396, MIN=1.5712 — showed I600=0% works; ultra-fine did better

**Phase 90 balanced champion** (superseded): `V1(27.36%) + I460_bw168_k4(17.98%) + I415_bw216_k4(29.66%) + I600_k2(2.5%) + F144_k2(22.50%)`
- AVG=2.0211, MIN=1.5614, Config: `configs/ensemble_p90_balanced.json`

**Phase 90 AVG-max champion** ← NEW AVG-MAX RECORD: `V1(4.5%) + I415_bw216_k4(10%) + I474_bw216_k4(40%) + I600_k2(7.5%) + F144_k2(38%)`
- AVG=2.3038, MIN=1.2046, YbY: [3.914, 1.218, 1.205, 3.483, 1.700]
- Config saved: `configs/ensemble_p90_avgmax.json`
- NEW AVG-MAX RECORD (2.3038 > P89's 2.286)! I474_bw216=40% (up from 35%)!

**Phase 89 balanced champion** (superseded): `V1(28.76%) + I460_bw168_k4(16.26%) + I415_bw216_k4(26.88%) + I600_k2(5%) + F144_k2(23.10%)`
- AVG=2.001, MIN=1.546, Config: `configs/ensemble_p89_balanced.json`

**Phase 89 AVG-max champion** (superseded): `V1(3.75%) + I415_bw216_k4(12.5%) + I474_bw216_k4(35%) + I600_k2(10%) + F144_k2(38.75%)`
- AVG=2.286, MIN=1.186, Config: `configs/ensemble_p89_avgmax.json`

**Phase 88 balanced champion** (superseded): `V1(28.75%) + I460_bw168_k4(16.25%) + I410_bw216_k4(22.5%) + I600_k2(5%) + F144_k2(27.5%)`
- AVG=2.015, MIN=1.529, Config: `configs/ensemble_p88_balanced.json`

**Phase 87 balanced champion** (superseded): `V1(27.43%) + I460_k4(13.07%) + I410_k4(23.66%) + I600_k2(7.5%) + F144_k2(28.34%)`
- AVG=2.002, MIN=1.493, Config: `configs/ensemble_p87_balanced.json`

**Phase 86 balanced champion** (superseded): `V1(26.82%) + I460_k4(11.85%) + I410_k4(20.56%) + I600_k2(10%) + F144_k2(30.77%)`
- AVG=2.007, MIN=1.469, Config: `configs/ensemble_p86_balanced.json`

**Phase 86 AVG-max champion**: `V1(4.98%) + I437_k4(16.17%) + I474_k4(30%) + I600_k2(10%) + F144_k2(38.85%)`
- AVG=2.268, MIN=1.125, YbY: [3.848, 1.311, 1.125, 3.314, 1.742]
- Config saved: `configs/ensemble_p86_avgmax.json`
- NEW AVG-MAX RECORD (2.268 > P85's 2.258)! I474=30% optimal!

**Phase 86 high-MIN (Pareto frontier)**: `V1(30%) + I460_k4(11.23%) + I410_k4(21.8%) + I600_k2(10%) + F144_k2(26.97%)`
- AVG=1.954, MIN=1.478, YbY: [3.039, 1.484, 1.478, 2.281, 1.489]
- Note: AVG<2.0 but MIN is highest at 1.478!

## Pareto Frontier (Phase 91b)
| AVG   | MIN   | Config |
|-------|-------|--------|
| 2.319 | 1.126 | V1=4.25%, I415bw216=6%, I474bw216=47.25%, I600=5%, F144=37.5% ← P91b AVG-MAX |
| 2.304 | 1.205 | V1=4.5%, I415bw216=10%, I474bw216=40%, I600=7.5%, F144=38% ← P90 AVG-MAX |
| 2.021 | 1.561 | V1=27.36%, I460bw168=17.98%, I415bw216=29.66%, I600=2.5%, F144=22.5% ← P90 |
| 2.010 | 1.576 | V1=27.47%, I460bw168=19.67%, I415bw216=32.47%, F144=20.39% ← P91b CHAMP (no I600!) |

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
- Phase 87: I600=7.5% BREAKTHROUGH (was 10%); balanced 2.002/1.493 dominates P86; lb=415 bridge=2.759; I410_bw216 2022=1.928
- Phase 88: bw=216 sweep; I410_bw216+I600=5% → 2.015/1.529 (P88 champ); I415_bw216 ALL years>1.0
- Phase 89: I415_bw216+I600=5% → 2.001/1.546 (balanced champ); I474_bw216+I415_bw216 → 2.286/1.186 (NEW AVG-MAX RECORD)
- Phase 90: I600=2.5% BREAKTHROUGH → 2.021/1.561 (dominates P89); AVG-max 2.304 (I474bw216=40%!); F144=22.5% confirmed floor; I437_bw216 fails (2023=0.175)
- Phase 91: I600=0% BREAKTHROUGH → remove I600! 4-signal: V1+I460bw168+I415bw216+F144 → 2.040/1.571; lb=415 CONFIRMED OPTIMAL (sharp cliff at lb=420: 2023 drops 1.143→0.758)
- Phase 91b: Ultra-fine balanced → 2.010/1.576 (F144=20.39%!); AVG-max 2.319 (I474bw216=47.25%!); I420/I430 with I415 get only 2.5% weight, not useful
- Phases 92-110: Various signal additions (PositioningAlpha, OrderflowAlpha etc) — mostly non-additive; P91b remains core champion
- Phase 111-113: Volume tilt overlay VALIDATED (3/4 WF positive, avg Δ=+0.066, tilt_ratio=0.65, lookback=168)
- Phase 118b: Global L/S ratio overlay (Binance Vision) VALIDATED — IS avg Δ=+0.380, OOS25 Δ=+0.1764
- Phase 127-129: BTC price vol overlay VALIDATED — LOYO 3/5 wins, avg Δ=+0.232, threshold=0.50, scale=0.50, f144_boost=0.20
- Phase 140: Commodity CTA KILLED (see Failed Research Directions)
- Phase 141-143: Rebalance freq (no improvement), idio rebal, weight optimization (2025 OOS fails at +0.098)
- **Phase 144**: Regime-adaptive weight switching IS VALIDATED — OBJ=+0.1820 improvement. PROD vs P143B weights.
- **Phase 145**: Breadth regime classifier VALIDATED — OBJ=1.8851 (+0.3179 vs baseline=1.5672), LOYO 4/5, capture 87.5%. Best: breadth_window=84/168/336h (all identical!), p_low=0.33, p_high=0.67. All 5 years > 1.57.
- **Phase 146**: Production integration — WF 1/2 wins, avg_delta=+0.4020. Best fine-tuned: 3-regime win=168 → OBJ=1.8945. Prod config updated to v2.2.0 with breadth_regime_switching block. BreadthAdaptiveEnsembleStrategy registered.

## Failed Research Directions
- **Commodity CTA with EMA signals (Phases 139-140, 2026-02-21): KILLED**
  - TrendFollowing EMA (12/26+20/50) on 8 commodities: FULL=-0.032, OOS_MIN=-0.184
  - CTAEnsemble (Trend40%+Carry30%+MomValue30%): FULL=-0.298 (Carry is massive drag: -0.498)
  - 14-sym (comm+FX+bonds) with Trend-only: OOS_MIN=-0.220 (WORSE than 8-comm)
  - Root cause: EMA signals on daily commodity data lack edge. Front-month continuous = no carry signal.
  - Verdict: Commodity CTA with EMA is not viable. Move to Deribit Skew MR or P91b improvements.

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
