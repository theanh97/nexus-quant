# Research Roadmap: Phase 92 Onwards
_Last updated: 2026-02-20 — after Phase 91b_

---

## Current Champions (Phase 91b)

### Balanced Champion (maximize MIN Sharpe)
```
V1(27.47%) + I460_bw168_k4(19.67%) + I415_bw216_k4(32.47%) + F144_k2(20.39%)
AVG = 2.0101 | MIN = 1.5761
YbY: [2.876, 1.577, 1.576, 2.445, 1.576]
Saved: configs/ensemble_p91_balanced.json
```
**Architecture: 4-signal, NO I600.**
All 5 years ≥ 1.576 — 2022 and 2023 are the binding constraints.

### AVG-max Champion (maximize mean Sharpe)
```
V1(4.25%) + I415_bw216_k4(6%) + I474_bw216_k4(47.25%) + I600_k2(5%) + F144_k2(37.5%)
AVG = 2.3192 | MIN = 1.1256
YbY: [3.939, 1.126, 1.196, 3.591, 1.745]
Saved: configs/ensemble_p91_avgmax.json
```

---

## Phase 92 — Ready to Run
**Script:** `scripts/run_phase92_dual_bw_f144floor_avgmax.py`
**Run:** `python3 scripts/run_phase92_dual_bw_f144floor_avgmax.py 2>&1 | tee artifacts/phase92_run.log`

### Sections:
| Section | Test | Hypothesis |
|---------|------|-----------|
| A | Profile I410_bw216 (5 fresh runs) | Confirm 2022=1.928, 2023=1.046 |
| B | 5-signal: V1+I460bw168+I410bw216+I415bw216+F144 | I410(2022) + I415(2023) → push MIN > 1.58 |
| C | F144 floor sweep 14-22%, step 0.3125% | Can F144 go below 20.39%? |
| D | AVG-max: I474=44-55%, step 0.25% | I474 at 47.25% already — can it go higher? |
| E | Ultra-fine 0.3125% around best balanced | Converge to true optimal |
| F | Pareto + champion save | Save configs/ensemble_p92_balanced.json |

### Key bottleneck analysis:
- P91b YbY: `[2.876, **1.577**, **1.576**, 2.445, 1.576]` — 2022, 2023, 2025 all at floor ~1.576
- **I410_bw216** standalone: [2.117, **1.928**, 1.046, 1.679, 1.408] — strongest 2022 signal!
- **I415_bw216** standalone: [1.967, 1.737, **1.143**, 2.015, 1.338] — best 2023 of any signal
- Together they could cover both weak years simultaneously

---

## Phase 93+ — Open Research Questions

### Q1: Can we push MIN above 1.60?
Current trend: 1.468 → 1.493 → 1.529 → 1.546 → 1.561 → 1.571 → 1.5761
Each phase gain ~0.01-0.03. Next natural ceiling unknown.

**Directions:**
- Try I460_bw216 (not tested yet!) — bw=216 helped I410 and I415 significantly
- Try I460 with lower bw: bw=144, bw=192 — systematic bw sweep for I460
- Explore lb=408, lb=412 (finer grid around I410-I415 range)

### Q2: New signal alpha — can a 5th signal improve the ensemble?
Current 4-signal architecture is already strong. Candidates for a 5th signal:
- **Basis momentum** (`basis_momentum_alpha`) — configs created, needs profiling
- **Lead-lag alpha** (`lead_lag_alpha`) — configs created, needs profiling
- **Vol breakout** (`vol_breakout_alpha`) — configs created, needs profiling
- **Volume reversal** (`volume_reversal_alpha`) — configs created, needs profiling
- **RS acceleration** (`rs_acceleration_alpha`) — configs created, needs profiling

**Decision gate:** A new signal is useful only if it:
1. Has MIN > 0.8 standalone
2. Has low correlation with V1 / idio / funding
3. Has distinct 2022/2023 year profiles (covers weak years)

### Q3: Can F144 floor go lower?
Trend: 30.77% → 28.34% → 27.5% → 23.10% → 22.50% → 20.39%
F144 adds 2021=3.1 and 2024=2.1 (high years) but has 2023=0.664 (weak).
At ultra-fine, optimal converges toward ~20%. Testing 14-20% in Phase 92.

### Q4: AVG-max ceiling — how far can I474 go?
I474_bw216 trend: 27.5% (P85) → 30% (P86) → 35% (P89) → 40% (P90) → 47.25% (P91b)
Phase 92 Section D tests up to 55%. Watch for diminishing returns.

### Q5: Beta window (bw) sweep for I460
- I460_bw168: [2.595, 0.635, 1.038, 2.709, 2.162] MIN=0.635 (current, in P91b)
- I460_bw216: [2.782, 0.566, 0.877, 3.032, 2.150] — actually **worse** 2022/2023!
- Never tested: I460_bw144, I460_bw192, I460_bw240
- Hypothesis: bw<168 might improve 2022 for I460 (mirrors bw=216 benefit for lb<460)

### Q6: New lb profiles — systematic fine scan lb=400-420
Only I410 and I415 tested with bw=216 in the 400-420 range.
- I412_bw216, I413_bw216, I414_bw216 untested
- The cliff at lb=420 (2023: 1.143→0.758) suggests lb=415 is optimal but worth verifying the exact peak

---

## New Alpha Strategy Profiling (Phase 93 candidate)

Configs already in `configs/run_*_alpha_*.json` for years 2021-2025:
- `basis_momentum_alpha` (5×1h configs)
- `lead_lag_alpha` (5×1h configs)
- `rs_acceleration_alpha` (5×1h configs)
- `vol_breakout_alpha` (5×1h configs)
- `volume_reversal_alpha` (5×1h configs)

**Phase 93 plan:** Profile all 5 new strategies (25 backtest runs), then:
1. Check standalone AVG/MIN/YbY
2. Check correlation with V1 / I460 / I415 / F144
3. If promising: add to ensemble sweep

---

## Architecture Summary

```
Current 4-signal balanced architecture (P91b):
  V1 (27.47%)      — multi-factor: carry + momentum + mean-reversion
  I460_bw168 (19.67%) — idio momentum lb=460h, bw=168h; good 2023/2024/2025
  I415_bw216 (32.47%) — idio momentum lb=415h, bw=216h; good 2022 AND 2023 ★
  F144 (20.39%)    — funding contrarian lb=144h; good 2021/2024 (orthogonal to idio)

Key insight: idio ↔ funding correlation = -0.009 (ORTHOGONAL!)
  → Diversification is nearly perfect between momentum and funding signals
```

---

## Disk Space Notes
- `artifacts/phase90/`, `artifacts/phase91/` contain backtest result JSONs (~1-2GB total)
- Old phases 59-89 artifacts may still exist and can be deleted to free space
- Phase 92 artifacts will go to `artifacts/phase92/`
- Champion configs are small (< 1KB each) and should always be kept

---

## How to Continue (for other devs)

1. **Run Phase 92** (ready):
   ```bash
   python3 scripts/run_phase92_dual_bw_f144floor_avgmax.py 2>&1 | tee artifacts/phase92_run.log
   ```

2. **Check memory** for context: `memory/MEMORY.md` (auto-loaded by Claude Code)

3. **Key files**:
   - Champions: `configs/ensemble_p91_balanced.json`, `configs/ensemble_p91_avgmax.json`
   - Phase scripts: `scripts/run_phase*.py`
   - Logs: `artifacts/phase*_run.log`
   - Runbook: `NEXUS_QUANT_RUNBOOK.md`

4. **Signal profile references** (from memory):
   - All idio signals use `"idio_momentum_alpha"` with `lookback_bars`, `beta_window_bars`, `k_per_side`
   - Funding uses `"funding_momentum_alpha"` (NOT `funding_contrarian_alpha`!)
   - V1 uses `"nexus_alpha_v1"` (NOT `"v1_standard"` — that throws UnknownStrategy!)
