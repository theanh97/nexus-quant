# Nexus Quant — Persistent Memory

## Champion Table (key milestones)
| Phase | Config | OBJ | Notes |
|-------|--------|-----|-------|
| 91b | V1+I460bw168k4+I415bw216k4+F144 | ~2.01 | PROD BASE |
| 200 | FTS short=16h long=72h | 3.0084 | First >3.0 |
| 232 | Per-regime DISP | 3.7033 | MID AMPLIFY 1.5× |
| **238** | Per-regime FTS pct_win | **3.7924** | **v2.42.0** |
| **244** | Per-regime V1 weights (NumPy) | **3.9048** | **v2.43.0** |
| **245** | Regime weight reopt8 (NumPy VEC) | **4.0145** | **v2.44.0 LOYO 4/5** |

## Current Production State — TWO TRACKS

### NumPy Track: v2.44.0, OBJ=4.0145
Regime weights (P245):
- LOW: v1=0.35, i460=0.10, i415=0.15, f168=0.40
- MID: v1=0.15, i460=0.00, i415=0.20, f168=0.65 *(overwritten by Engine P257)*
- HIGH: v1=0.30, i460=0.70, i415=0.00, f168=0.00 *(overwritten by Engine P257)*

**Per-regime V1 weights (P244):**
- LOW: wc=0.10, wm=0.45, wmr=0.45 | MID/HIGH: wc=0.25, wm=0.50, wmr=0.25

**Per-regime overlays (P229-P238):**
- FTS LOW: rs=0.5/bs=3.0/rt=0.8/bt=0.3/pct_win=240h
- FTS MID: rs=0.2/bs=3.0/rt=0.65/bt=0.25/pct_win=288h
- FTS HIGH: rs=0.4/bs=2.0/rt=0.55/bt=0.25/pct_win=400h
- VOL LOW=0.40/MID=0.15/HIGH=0.10 | DISP LOW=0.5/MID=1.5(AMP)/HIGH=0.5

### Engine Track: v2.42.13, OBJ=2.9980
Current config regime weights (v2.42.13):
- LOW: v1=0.35, i460=0.10, i415=0.15, f168=0.40 (sum=1.0 ✅)
- MID: v1=0.15, i460=0.00, i415=0.00, f168=0.85 (sum=1.0 ✅ — P257b constrained fix)
- HIGH: v1=0.20, i460=0.18, i415=0.27, f168=0.35 (sum=1.0 ✅ — P257)
- V1: wc=0.15, wm=0.55, wmr=0.30, mom_lb=312h (P255/P256)
- p_high: 0.68 (P260 breadth threshold)
- Vol scales: LOW=0.40, MID=0.50, HIGH=0.35 (P260 retune)

## Running Processes
- **P246 NumPy** (PID 83413): Per-regime V1 lookback sweep (170+/410 precompute)

## API Notes (BacktestEngine)
- `BacktestConfig(costs=cost_model)` — ONLY costs param
- `engine.run(dataset, strategy)` → BacktestResult `.returns`
- `dataset.last_funding_rate_before(symbol, ts)` — use ts from timeline[i]
- **NO** `dataset.funding_rate()`, `dataset.bars_for()` — don't exist!

## Strategy Names
- `"nexus_alpha_v1"` ← CORRECT (NOT "v1_standard"!)
- `"idio_momentum_alpha"` — params: lookback_bars, beta_window_bars, k_per_side
- `"funding_momentum_alpha"` — param: funding_lookback_bars

## YEAR_RANGES (CRITICAL)
Integer keys, ISO string values:
```python
{2021: ("2021-02-01","2022-01-01"), 2022: ("2022-01-01","2023-01-01"),
 2023: ("2023-01-01","2024-01-01"), 2024: ("2024-01-01","2025-01-01"),
 2025: ("2025-01-01","2026-01-01")}
```

## Vectorization Pattern (P245 approach — FAST)
```python
# Precompute per year: regime_idx, overlay, fixed contributions, target signals
regime_idx = np.where(bpct < P_LOW, 0, np.where(bpct >= P_HIGH, 2, 1))
# Build overlay_mult per bar (VOL × DISP × FTS for active regime)
# Sweep: ens = fixed_ens + wv1*v1t_sc + wi460*i460_sc + ...
# ~1000× faster than Python bar loop; use P245 precompute_fast_eval_data() pattern
```

## Phase Milestones (P238-P259)
P238: Per-regime FTS pct_win LOW=240/MID=288/HIGH=400 | OBJ=3.7924 v2.42.0
P239: HIGH i460=0.94 confirmed optimal
P240(DISP): NOT validated LOYO 2/5 | P241(FTS joint): NOT validated LOYO 2/5
P242(NumPy reopt7): NOT validated (Δ=+0.0021 < threshold)
P242(B): HIGH idio ratio=0.3 → i460=0.282/i415=0.658 VALIDATED v2.42.1
P243(B): FTS rt=0.55/rs=0.20/bs=2.30 VALIDATED v2.42.2
P244(B): Vol scale=0.40 VALIDATED v2.42.3 | P245(B): MID f168=0.95 VALIDATED v2.42.4
P246(B): LOW CONFIRMED OPTIMAL | P247(B): HIGH PERFECT 5/5 Δ=+0.3651 v2.42.5
  HIGH: v1=0.12, f168=0.20, i460=0.204, i415=0.476 — F168 re-enters HIGH
P248(B): FTS retune VALIDATED LOYO 3/5 v2.42.6 | P249(B): Vol CONFIRMED OPTIMAL
P250(B): MID f168=0.95 CONFIRMED OPTIMAL | P251: Breadth CONFIRMED OPTIMAL
P252(B): F168 lb CONFIRMED OPTIMAL | P253(B): I415 CONFIRMED OPTIMAL
P254(B): I460 CONFIRMED OPTIMAL
P255(B): V1 lb mom=312h VALIDATED LOYO 4/5 Δ=+0.0275 v2.42.7
P256(B): V1 weights wc=0.15/wm=0.55/wr=0.30 VALIDATED LOYO 4/5 Δ=+0.1054 v2.42.8
P257(B)-HIGH: retune2 VALIDATED LOYO 3/5 Δ=+0.0916 v2.42.9
P257(B)-ALL: all-regime retune VALIDATED LOYO 4/5 Δ=+0.5812 v2.42.10 ← BIGGEST ENGINE WIN
  MID: v1=0.5/f168=1.0 [BUG: sum=1.5] | HIGH: v1=0.2/f168=0.35/i460=0.18/i415=0.27
P257b-HOTFIX: MID weights sum=1.0 — f168=0.85 v1=0.15 (constrained) v2.42.11
P244(NumPy): per-regime V1 weights VALIDATED OBJ=3.9048 v2.43.0 [commit 40e7878]
P245(NumPy VECTORIZED): regime weight reopt8 VALIDATED OBJ=4.0145 LOYO 4/5 Δ=+0.1097 v2.44.0
P258(B): MID retune2 CONFIRMED OPTIMAL (ran on unconstrained MID — ignore)
P259(B): FTS retune2 NOT validated LOYO 2/5
P260: breadth p_high=0.60→0.68 v2.42.12 OBJ=2.9941
P260-VOL: per-regime vol scale VALIDATED LOYO 3/5 Δ=+0.1942 v2.42.13
  HIGH=0.10→0.35, MID=0.15→0.50, LOW=0.40 (confirmed optimal)

## Key Architectural Insights
- **NumPy vs Engine OBJ scale**: NumPy ~4.0 vs Engine ~3.0 — different cost models
- **BIFURCATION**: Two tracks share one config; Engine Track P257 overwrote MID/HIGH
- **Vectorization**: 1000× speedup vs bar loop; ~seconds vs 20+ min (P245 pattern)
- **HIGH regime truth**: NumPy says I460-dominated; Engine says I460=18%+I415=27%+F168=35%+V1=20%
- **MID regime**: Both tracks agree f168=dominant (0.65-1.0)
- **LOW regime**: v1=0.35, f168=0.40, balanced idio — both tracks agree
- **V1 per-regime**: LOW is more mean-reversion heavy (wmr=0.45), MID/HIGH more momentum

## Next Phase Candidates
After P246 finishes:
- If validated: P260 regime weight reopt with new V1 lookbacks (NumPy vectorized)
- If not: P260 per-regime FTS retune (NumPy) with new ensemble weights from P245
- Engine track: VOL retune with new v2.42.10 weights | LOW regime fine-tune

## Bug History (avoid repeating)
- YEAR_RANGES: must use integer keys, ISO string values (not epoch)
- `get_config()` returns 7 values — unpack all 7
- `python` → `python3` (python not in PATH)
- NumPy track `_version`, Engine track `version` — two different keys in config
- make_weights() / make_mid_w(): does NOT enforce sum=1.0 constraint!
  If v1_w + f168_w > 1.0, remainder = max(0, 1-v1-f168) = 0 BUT weights still sum to >1
  Fix: cap candidates so v1+f168 ≤ 1.0, or normalize after sweep
  Symptom: OBJ monotonically increases with any weight = over-leverage artifact
