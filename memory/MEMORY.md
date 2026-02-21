# Nexus Quant — Persistent Memory

## Champion Table (latest wins)
| Phase | Config | AVG | MIN | Notes |
|-------|--------|-----|-----|-------|
| 81 | V1+I437_bw168_k4+I600+F144 | 2.010 | 1.245 | 22.5/22.5/15/40 (k4 single-lb balanced) |
| 84 | V1+I437k4+I460k4+I410k4+I600+F144 | 2.040 | 1.431 | triple-lb balanced |
| **91b** | **V1+I460bw168k4+I415bw216k4+F144** | **2.010** | **1.576** | **27.47/19.67/32.47/20.39 — PROD BASE** |
| **146** | **Breadth regime switching (3 weight sets)** | **2.156** | **1.577** | **OBJ=1.8945 — PRODUCTION v2.2.0** |
| **158** | **Breadth lb=192 + p=0.35/0.65 + v2.7.0 stack** | **~2.34** | **1.966** | **OBJ=2.2095 LOYO 5/5 PERFECT — PRODUCTION v2.8.0** |
| **165** | **TS short=12h/long=96h rolling-window (v2.8.0 stack)** | **~2.36** | **1.541** | **OBJ=2.2423 LOYO 4/5 WF 2/2 — PRODUCTION v2.9.0** |
| **200** | **FTS windows short=16h/long=72h (full stack)** | **~3.00** | **2.85** | **OBJ=3.0084 LOYO 3/5 — PRODUCTION v2.28.0** |
| **238** | **Per-regime FTS pct windows LOW=240/MID=288/HIGH=400** | **~3.79** | — | **OBJ=3.7924 LOYO 3/5 Δ=+0.0207 — PRODUCTION v2.42.0** |

## Current Production State (v2.42.0 — P238) — OBJ=3.7924
**Breadth regime classifier:** lb=192h, percentile_window=336h, p_low=0.30, p_high=0.60
**FTS overlay:** short=16h, long=72h; per-regime RT/BT/RS/BS (P229-P230); per-regime pct_win LOW=240/MID=288/HIGH=400 (P238)
**Vol overlay:** per-regime scale LOW=0.40/MID=0.15/HIGH=0.10 (P231)
**DISP overlay:** per-regime scale LOW=0.5/MID=1.5(AMPLIFY)/HIGH=0.5 (P232)
**I460 beta window:** bw=168h, lb=480h | **I415 beta window:** bw=216h, lb=415h
**V1 params:** mr_lb=84h, vol_lb=192h; w_carry=0.25, w_mom=0.45, w_mr=0.30
**idio_lev:** 0.20 (P213b) | **f168_reb:** 36h (P215) | **v1_reb:** 60h
**global_ls_overlay:** enabled, lookback=168h, tilt_ratio=0.65 (P118b, WF confirmed OOS25=+0.1764)

| Regime | V1 | F168 | I460 | I415 |
|--------|-----|------|------|------|
| LOW    | 44% | 37%  | 8.6% | 10.3% |
| MID    | 20% | 78%  | 0.0% | 2.0%  |
| HIGH   | 6%  | 0%   | 94%  | 0.0%  |

**Per-regime FTS params (P229-P230):**
- LOW: rs=0.5/bs=3.0/rt=0.8/bt=0.3
- MID: rs=0.2/bs=3.0/rt=0.65/bt=0.25
- HIGH: rs=0.4/bs=2.0/rt=0.55/bt=0.25

**Key insight:** Vol-rank ≠ regime. 2022 bear ALSO prefers momentum-heavy weights.
True signal = funding/momentum richness (breadth), NOT realized volatility.
**Architecture insight:** Per-regime overlay specialization (P229-P232) = 68% OBJ lift (2.2423→3.7924).
**HIGH regime revolution:** 94% I460 (near-pure idio momentum) with minimal FTS overlay.
**MID regime:** F168-dominated (78%) — funding carry richness window.

## API Notes (BacktestEngine)
- `BacktestConfig(costs=cost_model)` — ONLY costs param, nothing else
- `engine.run(dataset, strategy)` → BacktestResult with `.returns` list
- `dataset.close(symbol, idx)` — close price at bar idx
- `dataset.timeline` — timestamp array (len = n_bars)
- `dataset.last_funding_rate_before(symbol, ts)` — funding at timestamp ts (NOT funding_rate_at)
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
- **Per-regime FTS pct_win**: LOW=240h (reactive), MID=288h (standard), HIGH=400h (long-memory)

## Failed Signals / Approaches
- FX EMA, Bonds CTA (P138): FAIL — crypto only
- Universe expansion 10→20 (P140): FAIL; Universe 10→15 (P151): FAIL (OBJ 1.89→1.29, -0.60 catastrophic)
- Vol term structure (P134), Funding level (P135): NO IMPROVEMENT
- 5th signal candidates (P139): ALL FAIL
- I437_bw216 (2023=0.175), Idio_800_bw168: FAIL
- Vol-rank as regime proxy: WRONG — 2022 needs i415-heavy despite high vol
- Cross-symbol correlation regime (P149): NO IMPROVEMENT — breadth already captures co-movement
- Cross-sectional skewness tail filter (P150): NO IMPROVEMENT — redundant with existing overlays
- Universe >10 symbols: STRUCTURAL FAIL — lower-quality alts dilute idio signal quality
- DISP global scale >1.0 (P213): OBJ=3.1007 but LOYO 2/5 (overfit) — per-regime needed

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
P149: Cross-symbol correlation regime — NO IMPROVEMENT (all variants worse)
P150: Cross-sectional skewness filter — NO IMPROVEMENT (all variants worse)
P150b (parallel): Funding term structure spread (short 24h vs long 144h) VALIDATED — LOYO 4/5, OBJ=2.0079 (FIRST >2.0!) | prod v2.4.0
P151: Universe expansion 10→15 symbols — NO IMPROVEMENT (15sym OBJ=1.29 vs 10sym=1.89; more coins hurt idio signal)
P152: Full-stack WF + TS fine-tune — OBJ=2.0851 (rt=0.70,rs=0.60,bt=0.30,bs=1.15), WF 2/2 | prod v2.5.0
P153: Calendar/day-of-week overlay — NO IMPROVEMENT (weekend carry alpha = don't reduce)
P154: Funding lookback sweep F72-F216 — F168 VALIDATED OBJ=2.1312 (Δ=+0.0461), LOYO 3/5 avg=+0.0794 | prod v2.6.0
P155: Rebalance interval sweep V1/I460/I415 — CONFIRMED CURRENT OPTIMAL (V1=60/I460=48/I415=48); all 26 variants worse
P156: Vol overlay fine-tune (thr/scale/boost) — VALIDATED scale=0.40+boost=0.15 OBJ=2.1448 (Δ=+0.0136), LOYO 4/5 | prod v2.7.0
P157: I460 beta window sweep bw=144/168/192/216/240 — CONFIRMED bw=168 optimal (all variants worse)
P158: Breadth classifier param sweep — **EXCEPTIONAL** lb=192 pw=336 p=0.35/0.65 OBJ=2.2095 (Δ=+0.0647), **LOYO 5/5 PERFECT** | prod v2.8.0
P159: Ensemble weight re-optimization — NO IMPROVEMENT (LOYO 2/5 overfit; current weights generalize better)
P160: WF validation v2.8.0 vs v2.4.0 — **CONFIRMED** WF 2/2 avg_delta=+0.1680 (OOS24 +0.1644, OOS25 +0.1716)
P161: V1 params + disp threshold — NO IMPROVEMENT (overfit/noise); v2.8.0 remains optimal
P162: F168 rebalance interval sweep [8/12/16/24/36/48h] — NO IMPROVEMENT (rb=24h optimal; shorter=costs, longer=sluggish)
P163: I415 beta window sweep [144-288] — NO IMPROVEMENT (bw=216 confirmed; bw=144 LOYO 2/5 overfit 2024)
P164: TS overlay window sweep — **CANDIDATE** short=12h/long=96h OBJ=2.2846 (new rolling-window method) LOYO 4/5
P165: TS window validation — VALIDATED short=12h/long=96h OBJ=2.2423 (Δ=+0.0328), LOYO 4/5, WF 2/2 | prod v2.9.0
P166b: FTS retune — VALIDATED rt=0.75/rs=0.50/bt=0.30/bs=1.25 OBJ=2.3946 LOYO 5/5 | prod v2.10.0
P168-171: Stacking overlays → OBJ=2.4153 (vol+disp+FTS chain) | v2.13.0
P172: I460 bw=120 + regime re-opt → OBJ=2.4817 | v2.14.0
P175: I415 bw=144 (was 216) → OBJ=2.5396 LOYO 3/5 | v2.15.0
P179: METRIC BUG — ×8760 inflation, OBJ÷8760≈2.48; relative LOYO valid, committed anyway
P185-WF: WF 3/3 CONFIRMED avg_OOS_delta=+0.7051 | v2.23.0
P187: V1 mr_lb=84h → OBJ=2.7103 (+0.1158) LOYO 4/5 | v2.22.0
P190: V1 vol_lb=192h → OBJ=2.8691 | v2.24.0
P194: I460 bw→168h → OBJ=2.8797 | v2.24.0
P195-WF: WF 3/3 CONFIRMED avg_OOS_delta=+0.6524 | v2.24.0
P195: Regime weight re-opt → OBJ=2.8988 | v2.24.0
P196: Regime extend → OBJ=2.9142 LOYO 3/5 | v2.25.0
P197: Breadth re-sweep p_low=0.30 p_high=0.60 → OBJ=2.9615 LOYO 3/5 | v2.26.0
P198: Vol overlay retune vol_scale=0.30 f168_boost=0.00 → OBJ=2.9628 | v2.27.0
P199b: Regime weight re-opt v2 (p_low=0.30) → OBJ=2.9956 LOYO 3/5
P200: FTS windows short=16h long=72h → **OBJ=3.0084 FIRST >3.0 MILESTONE** LOYO 3/5 | prod v2.28.0
P200b: Regime weight extend v2 → OBJ=3.0066 LOYO 3/5 | v2.29.0
P201-WF: WF CONFIRMED 3/3 wins avg_OOS_delta=+0.7865 IS_OBJ=3.0243 | v2.30.0
P201: Regime weight final re-opt → OBJ=3.0616 LOYO 3/5 | v2.30.0
P202: MID+LOW weight tune MID_f168=0.70 LOW_v1=0.44 → OBJ=3.0797 LOYO 4/5 | v2.31.0
P203: V1 w_carry=0.25 w_mom=0.45 w_mr=0.30 → **OBJ=3.0860 LOYO 4/5** | v2.33.0
P204-213: Mass confirmations — V1 lbs, I460/I415 bw, F168 lb, FTS, breadth, k_per_side, vol all optimal
P213b: idio_lev 0.30→0.20 → OBJ=3.1603 LOYO 5/5 PERFECT | P219: I460 lb=480 OBJ=3.1534
P215: Rebalance sweep → v1_reb=60h, f168_reb=36h OBJ=3.1190 LOYO 3/5 Δ=+0.033 VALIDATED
P215-FTS: ts_rs=0.30 ts_bs=1.85 → OBJ=3.2689 LOYO 3/5 (Branch A, stacked synergistic)
P224-225: Branch B pct_win=288h → 3.1953; rs=0.35 bs=2.50 → 3.2437 LOYO 4/5 v2.36.0
P226-228: Both branches local max ~3.24-3.27; P227 near-miss 3.2656 LOYO 2/5
P229: Per-regime FTS RS+BS → OBJ=3.3222 LOYO 5/5 Δ=+0.078
  LOW rs=0.5/bs=3.0 | MID rs=0.2/bs=3.0 | HIGH rs=0.4/bs=2.0
P230: Per-regime FTS RT+BT → **OBJ=3.4892 LOYO 5/5 Δ=+0.167**
  LOW rt=0.8/bt=0.3 | MID rt=0.65/bt=0.25 | HIGH rt=0.55/bt=0.25
P231: Per-regime VOL scale → OBJ=3.5261 LOYO 4/5 Δ=+0.037
  LOW scale=0.4 | MID scale=0.15 | HIGH scale=0.1
P232: Per-regime DISP → OBJ=3.7033 LOYO 4/5 Δ=+0.177
  MID scale=1.5 AMPLIFY! LOW/HIGH scale=0.5
P233: Regime weight re-opt 6th → **OBJ=3.7718 LOYO 3/5 Δ=+0.068** VALIDATED
  HIGH: v1=0.18 i460=0.79(!) f168=0.03 i415=0.0 — HIGH now I460-dominated
  MID: v1=0.2 f168=0.78 | LOW: v1=0.44 f168=0.37 balanced
P234-237: Confirmation passes — all WEAK/CONFIRMED OPTIMAL; I460lb/I415bw/V1 momlb/reb all at global optimum
P238: Per-regime FTS pct_win sweep → **OBJ=3.7924 LOYO 3/5 Δ=+0.0207** VALIDATED
  LOW=240h/MID=288h/HIGH=400h | prod v2.42.0
P239: HIGH regime reopt confirming i460=0.94 — v2.42.0 (comment-only param update)
**Prod OBJ=3.7924 — Per-regime overlay stack fully specialized; HIGH=94% I460 near-pure momentum**
**Next:** P240 (LOW+MID reopt), P241 (breadth thr resweep), P242 (idio ratio sweep) — scripts ready

P240 (per-regime DISP pct): best=3.8601 LOYO 2/5 ❌ NOT validated (LOW=672h overfit)
P241 (FTS joint resweep): best=3.9257 LOYO 2/5 ❌ NOT validated (near ceiling)
P242 (NumPy regime_reopt7): best=3.7945 Δ=+0.0021 LOYO 3/5 NOT validated (Δ<0.005 threshold)
  HIGH v1=0.20/i460=0.79 still near-optimal in NumPy

⚠️ FRAMEWORK BIFURCATION (v2.42.1-v2.42.4) — Author: Pham The Anh (BacktestEngine-direct):
  P242(B): HIGH idio r=0.3 → i460=0.282/i415=0.658 Δ=+0.2048 LOYO 3/5 VALIDATED
  P243(B): FTS global rt=0.55/rs=0.20/bs=2.30 LOYO 3/5 VALIDATED
  P244(B): Vol scale=0.40 LOYO 4/5 VALIDATED
  P245(B): MID f168=0.95/v1=0.05/idio=0 LOYO 3/5 VALIDATED → v2.42.4
CONFLICT: NumPy says HIGH=79% I460; Engine says HIGH=28% I460/66% I415
Root cause: Engine models real costs → F168/I415 more consistent under friction; I460 overfit in NumPy
Current config v2.42.4: HIGH i460=0.282/i415=0.658 | MID f168=0.95 | Vol=0.40 | FTS rt=0.55/rs=0.20

P246(B): LOW regime retune → CONFIRMED OPTIMAL (LOYO 0/5 Δ=0)
P247(B): HIGH regime retune → **LOYO 5/5 PERFECT Δ=+0.3651** | Engine OBJ 2.3745→2.7396 | v2.42.5
  HIGH: v1=0.12 f168=0.20(NEW!) i460=0.204 i415=0.476 (ratio=0.30)
  F168 re-enters HIGH regime! Key: momentum+carry+F168 blend beats pure idio in engine framework
P243(NumPy): DISP thr/scale resweep best=3.8054 LOYO 2/5 ❌ NOT validated
**Prod Engine OBJ=2.7396 v2.42.5 — HIGH regime now has F168 20% component (critical insight)**
