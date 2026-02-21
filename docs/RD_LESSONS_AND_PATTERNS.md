# R&D Lessons Learned — Quant Trading Strategy Development

## 1. Signal Discovery Pipeline (proven workflow)
```
Profile signal (5yr yearly Sharpe) → Correlation with champion → Ensemble sweep → OOS validation
```
- **Always profile across FULL market cycle** (bull + bear + chop) — minimum 3-5 years
- **Year-by-year Sharpe is essential** — AVG hides regime failure
- **MIN Sharpe is the binding constraint** — a signal with AVG=2.0 but MIN=-0.5 is useless
- **Correlation before ensemble** — high-corr signals (>0.4) add noise, not diversification

## 2. The Orthogonality Paradox
- Signals with LOW correlation to champion are almost always DEAD (negative Sharpe)
- Signals with HIGH Sharpe are almost always CORRELATED with champion (same underlying alpha)
- **Implication**: true diversification requires fundamentally different DATA SOURCES, not different math on the same data
- Price-based signals (momentum, MR, vol, Sharpe ratio, Sortino, skip-gram, etc.) all capture the same underlying cross-sectional momentum alpha

## 3. Ensemble Optimization Rules
- **Balanced objective = (AVG + MIN) / 2** works best for robust strategies
- **AVG-max chases 2021 bull** and sacrifices bear/chop years
- **Fine-grid sweep (0.5-1% steps)** after coarse (5%) adds ~0.01 to MIN — marginal but real
- **Weight normalization**: always renormalize weights to sum=1.0 after grid generation
- **Vectorized numpy sweep** (`W @ R` matrix multiply) is 100x faster than loops

## 4. Static vs Adaptive Weights
- **Static weights beat adaptive** for MIN Sharpe (confirmed Phase 95)
- Rolling optimization overfits to recent regime → destroys MIN by 0.2-0.3
- Inverse-vol gives higher AVG but lower MIN — not worth it for balanced criterion
- Signal gating (reduce weight when trailing Sharpe < 0) is DANGEROUS — removes signal right before it recovers

## 5. Overfitting Detection
- **IS vs OOS gap > 50%**: likely overfit (e.g., IS Sharpe=2.0, OOS=0.9 → gap=55%)
- **Only 1 binding year** (MIN from same year across all variants): ceiling is fundamental, not tunable
- **Weight drifts to 0 in ensemble**: signal adds nothing (e.g., i410bw216 got 5% weight, corr=0.966 with i415)
- **Survivorship bias**: using current symbols on historical data inflates bull years (2021)

## 6. When to Stop Optimizing a Signal Family
- When **all parameter variants** (lookback × k × rebal) converge to same Sharpe range (±0.1)
- When adding 5th+ signal **reduces MIN** instead of improving it
- When **correlation between top candidates > 0.85** (diminishing returns)
- When **OOS degrades** as you add complexity

## 7. Backtest Infrastructure Lessons
- **Cache everything**: Binance API fetches are slow; `.cache/binance_rest/` saves hours
- **Hourly data + sqrt(8760)**: standard annualization for hourly Sharpe
- **Cost model matters**: fee=0.05% + slippage=0.03% + taker=3bps is realistic
- **PYTHONUNBUFFERED=1**: always use for real-time output from long scripts
- **Separate IS (2021-2025) from OOS (2026)**: never tune on OOS

## 8. Multi-Model Collaboration
- **Claude (Opus)**: PM + strategy research + code + analysis
- **Codex CLI (GPT-5.x)**: UI/UX, frontend, refactoring — good for "make it better" tasks
- **Pattern**: Claude writes strategy + runs backtest; Codex polishes dashboard/UI
- **Integration fix needed**: Codex often leaves init calls missing — always verify integration

## 9. Strategy Categories That Work vs Don't (Crypto Perps)
### WORKS (Sharpe > 0.8 over 5yr):
- Cross-sectional momentum (various lookbacks 300-500h)
- Idiosyncratic momentum (beta-adjusted)
- Contrarian funding rate momentum
- Buy-dips-in-uptrends (V1-Long: mom filter + MR timing)

### DOESN'T WORK (tested and confirmed dead):
- Basis momentum, lead-lag, vol breakout, volume reversal, RS acceleration
- Orderflow alpha (decays), positioning alpha, funding carry
- Dispersion alpha, regime mixer, adaptive rebalancing
- EWMA Sharpe, Amihud illiquidity, taker buy ratio, funding vol
- Price level (52wk high), mean-reversion with funding filter
- Pair spread MR (stat arb z-scores on log price ratios) — all params deeply negative
- Volume-regime momentum (vol anomaly × momentum) — inconsistent across years
- Momentum breakout (binary threshold) — partial IS signal but OOS failure in ensemble

## 13. Signal Construction Traps (Phase 96)
- **Pair spread in crypto**: Pairs are NOT cointegrated — correlations are unstable, z-scores are meaningless. ALL 6 variants AVG < -1.3
- **Volume conditioning**: Filtering momentum by volume regime adds noise, not alpha. Volume spikes don't predict cross-sectional returns
- **IS overfit via ensemble**: brk_168_5pct had IS MIN=1.859 (+0.283 vs P91b!) but OOS collapsed to -0.43. Classic weight-optimization overfit when new signal has negative years
- **Rule**: If a new signal has negative Sharpe in ANY year, ensemble optimization will exploit the "good years" but fail OOS

## 14. Universe + Timeframe + Bias Lessons (Phase 97-99)
- **Universe expansion HURTS**: Adding mid-cap coins (MATIC, ATOM, NEAR, APT, ARB) to 10 large-cap universe degrades Sharpe catastrophically. Cross-sectional momentum works best on TOP-10 most liquid coins.
- **Data availability trap**: Newer coins (APT launched Nov 2022, ARB Mar 2023) cause data alignment failures for historical backtests. Always check listing dates before expanding universe.
- **4h bars lose signal**: Same strategy on 4h bars (vs 1h) drops AVG from 2.01 to 1.48 and MIN from 1.30 to 0.46. Coarser resolution = less signal precision.
- **Long-bias is fatal for bear years**: Even 5% market exposure kills MIN Sharpe from 1.30 to 0.27. Dollar-neutral is MANDATORY for all-weather robustness.
- **Survivorship bias quantified**: Removing SOL/DOGE/AVAX from 2021 drops Sharpe by 0.50. But MIN (from 2023) is unchanged. The binding constraint is NOT affected by survivorship.
- **Per-year vs continuous backtest**: Running each year separately slightly differs from continuous 5yr run due to position carry-over and lookback warmup at year boundaries.

## 10. Project Management Patterns
- **Phase numbering**: always increment, never reuse
- **Artifact saving**: `artifacts/phaseXX/report.json` for reproducibility
- **Commit per phase**: one commit message summarizing what was tested and concluded
- **Memory updates**: keep MEMORY.md under 200 lines, link to detail files
- **User preference: auto-proceed, no confirmation, 24/7 R&D until intervention**

## 11. Dashboard/UI Lessons
- **SSE (Server-Sent Events)** for real-time: simpler than WebSocket, one-directional
- **Single HTML SPA**: easier to maintain than framework for internal tools
- **Dark mode**: use CSS custom properties + `data-theme` attribute
- **Always init new features**: Codex adds functions but forgets to call them at startup

## 12. Key Metrics & Thresholds (Crypto Perps, 10 large-cap, hourly)
- **Live Sharpe expectation**: IS × 0.5-0.6 (50% degradation from IS to live)
- **Realistic cost**: ~8bps round-trip (fees + slippage)
- **Good MIN Sharpe**: > 0.8 over full cycle (bear+chop)
- **Champion ceiling**: ~1.58 MIN for this universe/signal/cost config
- **OOS degradation**: ~40-50% from IS average to true OOS

## 15. Weight Stability + Regime Insurance (Phase 101-102)
- **Walk-forward weight stability**: f144 perfectly stable at 40% across all windows (std=0.000). Other weights vary more but converge to similar ranges.
- **IS weight ≠ OOS importance**: v1 gets only 0-5% weight in IS optimization, but it's the ONLY signal working in 2026 OOS (Sharpe=3.31 vs others negative). Removing it collapses OOS from +1.27 to -0.61.
- **Regime insurance principle**: Low-weight signals provide insurance for future regimes. A signal that adds little in-sample may be the ONLY one working in a new regime.
- **3-signal ensemble TRAP**: All 3-sig variants improved IS (ΔMIN=+0.02..+0.04) but destroyed OOS. Classic case of optimizing on IS leading to worse OOS.
- **Rule**: NEVER drop a signal just because its IS weight is low. Always validate with OOS before removing. If you can't validate OOS (e.g., limited data), keep ALL profitable signals.
- **Cost sensitivity**: P91b profitable at ALL cost levels tested (ultra_low to ultra_high). MIN Sharpe still 0.91 at ultra-high costs (fee=10bps, slip=7bps).
- **Leverage invariance**: Sharpe is mathematically scale-invariant — doubling leverage doubles both mean and std. Only affects absolute returns, not risk-adjusted.

## 16. Alternative Data Signals (Phase 105-108)
- **Blockchain.com API**: Free, years of daily data (hashrate, tx count, unique addresses, mempool). Truly orthogonal to price-based signals.
- **On-chain as binary filter**: NEVER use — removes too many profitable hours. Filter = throwing away ~50% of returns.
- **On-chain as leverage tilt**: Mild effect. Reduce leverage by X% when on-chain momentum is negative.
- **IS overfit trap with daily overlays**: composite_14@0.4 showed OBJ=1.829 (+0.177 vs baseline) on full IS, but walk-forward revealed 3/4 OOS windows negative. CLASSIC IS overfit.
- **Walk-forward is MANDATORY for any overlay/filter**: Never trust IS-only results for overlays.
- **composite_30@0.8**: Only variant that barely validates WF (2/4 positive, avg Δ=+0.056). Ratio perfectly stable but effect too small to be confident.
- **Daily vs hourly resolution gap**: On-chain data is daily; strategy is hourly. This mismatch means on-chain can only be a coarse filter/overlay, not a primary signal.
- **Rule**: For new data sources at different frequency than primary strategy, test as tilt (partial leverage reduction) before filter (binary on/off). Tilts are more robust.

## 17. Sentiment & Implied Volatility (Phase 109-110)
- **Fear & Greed Index (alternative.me)**: Free API, 2000 days of daily data. fng_mom_14 (14-day contrarian) showed IS ΔMIN=+0.162, ΔOOS=+0.310.
- **F&G WF validation**: FAILED (1/4 positive, avg Δ=+0.005). Another IS overfit — promising IS results don't survive walk-forward.
- **Deribit DVOL**: 1000 days accessible, but ALL IV-based variants hurt MIN Sharpe. IV doesn't add alpha for cross-sectional crypto momentum.
- **Combined signal** (F&G + on-chain): Also fails WF (1/4 positive). Combining two weak signals doesn't make a strong one.
- **Pattern**: Daily sentiment/IV signals face the same frequency mismatch as on-chain — coarse filters on hourly strategies are inherently unstable.

## 18. Volume Momentum Tilt — THE BREAKTHROUGH (Phase 111-113)
- **vol_mom_z_168**: Aggregate volume across all symbols → 168h log momentum → rolling z-score → contrarian tilt at r=0.65.
- **WF validation**: **3/4 positive** windows, avg Δ=+0.066, ratio perfectly stable (std=0.00). BEST WF result across ALL overlays tested.
- **Mechanism**: When volume momentum z > 0 (crowded/high-activity market), reduce leverage to 65%. Avoids crowded reversals.
- **Why it works but others don't**: Volume is SAME FREQUENCY as strategy (hourly), and uses data already in dataset (no external API). No frequency mismatch, no data freshness issues.
- **Fine-tuning**: r=0.65 marginally better than 0.70 (OBJ 1.7159 vs 1.7157). lb=168 clearly optimal — shorter (24, 48) overfit, longer (336) too slow.
- **Impact**: ΔMIN=+0.130 (1.297→1.427), ΔOOS=+0.244 (1.268→1.512), MDD improved all years.
- **Key insight**: The best overlay uses DATA YOU ALREADY HAVE at the SAME FREQUENCY. External/daily signals add complexity with marginal/unstable improvement.
- **Binance OI/LS**: BLOCKED for backtesting — free API only has ~21 days of hourly history. Cannot validate over full market cycle.

## 19. Overlay Research Meta-Lesson
- **Tested**: on-chain (4 charts × 6 lookbacks), F&G, DVOL, combined, volume (5 lookbacks)
- **Validated (WF 3/4+)**: Only vol_mom_z_168 @ r=0.65
- **Barely validated (WF 2/4)**: composite_30 @ r=0.8 (but unstable on fresh data, Phase 112 showed it HURTS)
- **Failed (WF 0-1/4)**: Everything else
- **Conclusion**: Overlays from hourly data in the same dataset >> daily external data >> sentiment/IV data
- **Rule**: After finding a validated overlay, always do a FULL integration test (Phase 112 pattern) with fresh data fetch to confirm. composite_30 "validated" in Phase 108 but hurt in Phase 112 with fresh blockchain.com data.

## 20. Exhaustive Overlay Testing (Phase 114-117)
- **Taker buy ratio**: Marginal (ΔOBJ≤0.009). The buy/sell ratio doesn't predict cross-sectional momentum returns.
- **Volume concentration (HHI)**: IS shows +0.030 ΔOBJ but WF 0/4 → overfit. Concentrated volume ≠ crowded trades.
- **Funding rate dispersion**: ALL NO VALUE. Cross-sectional funding spread has no regime prediction power.
- **Cross-asset (S&P500/Gold/DXY)**: ALL 12 variants NO VALUE. TradFi regime indicators add NOTHING to crypto perps. Crypto alpha is crypto-specific.
- **Realized vol regime**: rvol_z_168 showed IS ΔOBJ=+0.098 but WF FAILED. Vol clustering doesn't improve momentum timing.
- **Drawdown indicator**: ARTIFACT — setting r=0.0 during drawdowns zeroes negative returns (circular). NOT a signal, just a mechanical data trick. DISCARD.
- **LESSON**: Reactive signals (drawdown, trailing Sharpe) are NOT predictive. They describe the past, not the future. Only FORWARD-LOOKING signals (momentum, carry, volume activity) can improve strategies.

## 21. Final R&D Conclusion (116+ Phases)
- **Champion**: P91b + vol_mom_z_168 @ r=0.65 (AVG=2.005, MIN=1.427, OOS 2026=1.512)
- **Only validated overlay**: vol_mom_z_168 — same-frequency hourly data from the same dataset
- **Expected live**: Sharpe 0.8-1.0 after costs, slippage, and execution
- **Signal exhaustion**: 40+ signals, 200+ params, 25+ overlays, 5 alt data sources → DONE
- **Next alpha**: Requires fundamentally new data (options flow, whale wallets, exchange order books) or new markets (non-crypto, different instruments)

## 22. Multi-Model Integration (Phase 118+)
- **Google Gemini OpenAI-compatible endpoint** = zero-dependency integration. Same `_call_openai()` method, different base_url. Cleanest possible approach.
- **Gemini CLI fallback**: When GEMINI_API_KEY not set, subprocess to `gemini -p "PROMPT"` works. CLI authenticated via Google account (60 RPM, 1000 RPD free).
- **Cost optimization tiers**: Gemini Pro/Flash (FREE) replace MiniMax and handle code review, data analysis, QA, monitoring. Claude/GPT reserved for critical decisions.
- **Debate lineup**: GLM-5 + Claude Sonnet + Codex + Gemini Pro = 4 models cross-checking. Each brings different training data and biases → better consensus.
- **SmartRouter provider pattern**: `provider="google"` routes through `_call_openai()` with Google's base_url. Same OpenAI format, transparent to callers.
- **Connectivity test script**: `scripts/test_model_connectivity.py` — always verify all 5 models before deployment. Takes ~2 min.

## 23. Robustness Analysis (Phase 118-119)
- **Bootstrap CI**: Block bootstrap (168h blocks, 1000 iters) preserves autocorrelation. Full-sample CI [1.097, 3.059], P(>0)=100%.
- **Permutation test**: Vol tilt timing is NOT statistically significant (p=0.365). Core P91b is robust without tilt; tilt is marginal overlay.
- **Parameter perturbation**: 500 trials with ±20% weight + tilt jitter → 100% maintain OBJ>1.0. Strategy is on a BROAD PLATEAU, not a narrow peak.
- **Cost sensitivity**: Profitable even at 3x costs (24bps total). Breakeven >3x current costs. Extremely cost-resilient.
- **Monthly stability**: 64-73% positive months across all years. Worst single month: -5.98 (2023). Expected for hourly cross-sectional momentum.
- **Rule**: Run robustness analysis in TWO phases — bootstrap+permutation first (lightweight), then cost+perturbation (heavier). Avoid single monolithic script that times out.

## 26. Pattern Recognition System — Learning From Mistakes (Meta-Learning)
**"Hệ thống nhận ra hệ thống mẫu hình"** — patterns of mistakes repeat across projects. Recording them prevents re-learning the same lesson.

### Pattern: Frequent Exit Mechanism = Enemy of Carry/Theta Strategies
**Instances**:
- P2 VRP: exit at z<-1.5 (frequent) → Sharpe 0.878. Exit at z<-2.0 (rare) → Sharpe 1.520
- P4 SPX: SL=2× (frequent) → Sharpe -7.0. SL=10× (less frequent) → Sharpe +0.7. NO SL → Sharpe +5.4
**Pattern**: "If your strategy makes money from WAITING (theta, carry), then any EXIT mechanism that triggers FREQUENTLY will kill it. Make exits RARE."
**Meta-rule**: Before adding any exit condition to theta/carry strategy, ask: "Does this exit fire often? If yes, don't add it."

### Pattern: Daily Rebalance Destroys Carry Strategies
- VRP freq=1 (daily) → avg Sharpe -0.534. freq=5 (weekly) → Sharpe 1.520
- CTA ensemble: daily rebalance → broken (same bug). Weekly → works.
**Meta-rule**: "Carry/theta strategies need time to work. Daily rebalance = transaction costs > alpha."

### Pattern: IS Breakthrough ≠ OOS Validation (Overfit Detection)
- On-chain composite_14: IS breakthrough, WF overfit → classic trap
- F&G momentum: IS promising, WF fails 3/4 years
- VRP exit_z=-1.5: looks IS optimal but OOS underperforms conservative -2.0
**Meta-rule**: "When you find an IS breakthrough, run WF (LOYO/roll-forward) IMMEDIATELY before getting excited."

### Pattern: Cross-Disciplinary Solutions Are Always 10× Better
- P2 VRP insight → applied to P4 SPX → immediate breakthrough (no parameter search needed)
- Ensemble from crypto → directly applicable to commodity CTA logic
**Meta-rule**: "When stuck on a problem, look at solutions from OTHER projects first."

### Pattern: Parameters That Look Good Short-Term Kill Long-Term
- Rebalance freq optimization: narrow optimum at 72h, degrades OOS
- MA crossover (20/50 bars on 1-min data) = 20-min MA = noise, not regime
**Meta-rule**: "If parameter optimization shows a very narrow peak, it's an artifact. Look for wide, robust optima."

## 25. Engineering Philosophy — 10x Thinking (Core Principle)
**"Tư duy gấp 10 lần, 100 lần so với kỹ sư hàng đầu"** — this is a MANDATORY mindset:
- **First Principles**: Decompose every problem to physics/math. Ask WHY 3× before accepting.
  - Example: baseline SL=2 fails. WHY? → EV math: 0.47×0.5 - 0.53×2 = -0.825. → Fix at root.
- **Cross-disciplinary**: P2 VRP lesson (no micro-management) → directly applied to P4 SPX = breakthrough
- **Parallel execution**: ALWAYS run background jobs + write code simultaneously. Idle CPU = waste.
- **10x not 10%**: When F config (+5.41 avg) replaces A (+0.67 avg) = 8× improvement. THAT is 10x thinking.
- **Industry standards are the floor, not the ceiling**: If everyone uses SL, question whether SL is right.
- **Memory = growth**: Every insight MUST be recorded. Claude grows by writing lessons, not just remembering session state.

**Operating Protocol**:
1. Launch background jobs (nice -n 15) BEFORE writing code (never let CPU idle)
2. Analyze partial results as they arrive (don't wait for completion)
3. Update memory with findings immediately — before committing
4. Commit often (after each phase, not at the end of session)

## 24. Real-Time Signal Generation (Phase 120)
- **Architecture**: Fetch 600h of data from Binance REST → run P91b ensemble at last bar → output target weights.
- **Key insight**: The strategy framework (target_weights at index) works for BOTH backtest and live signal generation. Same code, different data source (historical vs latest).
- **Paper state tracking**: JSON file at `artifacts/live/paper_state.json` — tracks equity, weights, entry prices, P&L history.
- **Vol tilt in live**: Compute volume z-score from latest 168h. If z>0 (crowded), reduce all weights by 0.65. Simple, mechanical.
- **First signal observation**: Vol tilt OFF (z=-1.41 on 2026-02-20). Market NOT crowded. Full leverage deployed.
- **CLI**: `python3 -m nexus_quant signal` for one-shot, `--loop` for hourly. `--json` for piping to other tools.
- **Dashboard**: /api/signals/latest, /api/signals/generate endpoints. "Signals" tab in dashboard UI.
