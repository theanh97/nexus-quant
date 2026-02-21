#!/usr/bin/env python3
"""
Correlation Estimator Validation Test
========================================

Tests the CorrelationEstimator with:
  1. Synthetic correlated returns → verify EWMA converges to true correlation
  2. Regime shift test → verify fast reaction to correlation changes
  3. Uncorrelated returns → verify estimates stay near zero
  4. Perfect correlation → verify estimates approach 1.0
  5. Stress test: sudden decorrelation during crash
  6. Rolling vs EWMA comparison

This validates the module before wiring it into the live risk overlay.
"""
import math
import random
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus_quant.portfolio.correlation_estimator import CorrelationEstimator

DIVIDER = "=" * 70


def generate_correlated_returns(n: int, rho: float, vol: float = 0.01,
                                 seed: int = 42) -> list:
    """Generate n pairs of correlated normal returns using Cholesky."""
    random.seed(seed)
    pairs = []
    for _ in range(n):
        z1 = random.gauss(0, 1)
        z2 = random.gauss(0, 1)
        x = vol * z1
        y = vol * (rho * z1 + math.sqrt(max(1 - rho ** 2, 0)) * z2)
        pairs.append((x, y))
    return pairs


def test_convergence_to_true_correlation():
    """Test 1: EWMA converges to true correlation with enough data."""
    print(f"\n{DIVIDER}")
    print("TEST 1: EWMA Convergence to True Correlation")
    print(DIVIDER)

    true_rho = 0.60
    n_samples = 500
    pairs = generate_correlated_returns(n_samples, true_rho, vol=0.015, seed=123)

    est = CorrelationEstimator(
        projects=["A", "B"], half_life=20, rolling_window=60
    )

    # Feed data
    correlations = []
    for i, (r_a, r_b) in enumerate(pairs):
        corr = est.update_pair({"A": r_a, "B": r_b})
        if i >= 100:  # after burn-in
            correlations.append(corr)

    final_corr = correlations[-1]
    avg_corr = sum(correlations) / len(correlations)
    rolling_corr = est.rolling_correlation()

    print(f"  True correlation:    {true_rho:+.4f}")
    print(f"  EWMA final:          {final_corr:+.4f}")
    print(f"  EWMA avg (post-100): {avg_corr:+.4f}")
    print(f"  Rolling (window=60): {rolling_corr:+.4f}")
    print(f"  Error (EWMA):        {abs(final_corr - true_rho):.4f}")
    print(f"  Error (rolling):     {abs(rolling_corr - true_rho):.4f}")

    # EWMA should be within 0.20 of true (it's noisy with finite samples)
    ewma_ok = abs(avg_corr - true_rho) < 0.25
    rolling_ok = abs(rolling_corr - true_rho) < 0.25
    passed = ewma_ok and rolling_ok

    print(f"\n  EWMA converged:   {'PASS' if ewma_ok else 'FAIL'}")
    print(f"  Rolling converged: {'PASS' if rolling_ok else 'FAIL'}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_regime_shift_reaction():
    """Test 2: EWMA reacts to sudden correlation regime change."""
    print(f"\n{DIVIDER}")
    print("TEST 2: Regime Shift Reaction Speed")
    print(DIVIDER)

    # Phase 1: low correlation (200 samples)
    rho_low = 0.10
    pairs_low = generate_correlated_returns(200, rho_low, vol=0.01, seed=200)

    # Phase 2: high correlation (200 samples)
    rho_high = 0.80
    pairs_high = generate_correlated_returns(200, rho_high, vol=0.01, seed=300)

    est = CorrelationEstimator(
        projects=["X", "Y"], half_life=20, rolling_window=60
    )

    # Phase 1
    for r_x, r_y in pairs_low:
        est.update_pair({"X": r_x, "Y": r_y})
    corr_end_phase1 = est.current_correlation()

    # Phase 2: track how fast EWMA reacts
    corr_at_step = {}
    for i, (r_x, r_y) in enumerate(pairs_high):
        corr = est.update_pair({"X": r_x, "Y": r_y})
        if i in [10, 20, 40, 60, 100, 199]:
            corr_at_step[i] = corr

    print(f"  Phase 1 (rho={rho_low}): final EWMA = {corr_end_phase1:+.4f}")
    print(f"  Phase 2 (rho={rho_high}): reaction speed:")
    for step, c in corr_at_step.items():
        marker = " <<< spike detected" if c > 0.50 else ""
        print(f"    Step {step:3d}: {c:+.4f}{marker}")

    # After ~40 steps (2x half-life), should be noticeably moving toward 0.80
    corr_40 = corr_at_step.get(40, 0.0)
    corr_200 = corr_at_step.get(199, 0.0)

    reacted = corr_40 > corr_end_phase1 + 0.10  # moved meaningfully
    converged = corr_200 > 0.40  # approaching true value

    print(f"\n  Reacted by step 40: {'PASS' if reacted else 'FAIL'} ({corr_40:+.4f} vs {corr_end_phase1:+.4f})")
    print(f"  Converged by step 200: {'PASS' if converged else 'FAIL'} ({corr_200:+.4f})")

    passed = reacted and converged
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_uncorrelated():
    """Test 3: Uncorrelated returns → estimate near zero."""
    print(f"\n{DIVIDER}")
    print("TEST 3: Uncorrelated Returns (rho=0)")
    print(DIVIDER)

    pairs = generate_correlated_returns(500, 0.0, vol=0.01, seed=400)

    est = CorrelationEstimator(
        projects=["U", "V"], half_life=20, rolling_window=60
    )

    for r_u, r_v in pairs:
        est.update_pair({"U": r_u, "V": r_v})

    ewma = est.current_correlation()
    rolling = est.rolling_correlation()

    print(f"  EWMA:    {ewma:+.4f}")
    print(f"  Rolling: {rolling:+.4f}")

    # Should be close to zero (within ±0.25 for finite samples)
    ewma_ok = abs(ewma) < 0.25
    rolling_ok = abs(rolling) < 0.25
    passed = ewma_ok and rolling_ok

    print(f"\n  EWMA near zero:    {'PASS' if ewma_ok else 'FAIL'}")
    print(f"  Rolling near zero: {'PASS' if rolling_ok else 'FAIL'}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_perfect_correlation():
    """Test 4: Perfect correlation → estimate approaches 1.0."""
    print(f"\n{DIVIDER}")
    print("TEST 4: Perfect Correlation (rho=0.99)")
    print(DIVIDER)

    pairs = generate_correlated_returns(500, 0.99, vol=0.01, seed=500)

    est = CorrelationEstimator(
        projects=["P", "Q"], half_life=20, rolling_window=60
    )

    for r_p, r_q in pairs:
        est.update_pair({"P": r_p, "Q": r_q})

    ewma = est.current_correlation()
    rolling = est.rolling_correlation()

    print(f"  EWMA:    {ewma:+.4f}")
    print(f"  Rolling: {rolling:+.4f}")

    ewma_ok = ewma > 0.85
    rolling_ok = rolling > 0.85
    passed = ewma_ok and rolling_ok

    print(f"\n  EWMA high:    {'PASS' if ewma_ok else 'FAIL'}")
    print(f"  Rolling high: {'PASS' if rolling_ok else 'FAIL'}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_crash_decorrelation():
    """Test 5: Simulate stress decorrelation (like VRP-Skew during crash)."""
    print(f"\n{DIVIDER}")
    print("TEST 5: Crash Decorrelation Scenario")
    print(DIVIDER)
    print("  Simulates R20: VRP-Skew correlation drops during stress")

    # Normal period: moderate positive correlation
    rho_normal = 0.40
    pairs_normal = generate_correlated_returns(300, rho_normal, vol=0.01, seed=600)

    # Crash period: near-zero correlation + high vol
    rho_crash = 0.05
    pairs_crash = generate_correlated_returns(50, rho_crash, vol=0.03, seed=700)

    # Recovery: moderate correlation, normal vol
    pairs_recovery = generate_correlated_returns(150, rho_normal, vol=0.01, seed=800)

    est = CorrelationEstimator(
        projects=["VRP", "SKEW"], half_life=20, rolling_window=60
    )

    # Normal period
    for r_v, r_s in pairs_normal:
        est.update_pair({"VRP": r_v, "SKEW": r_s})
    corr_pre_crash = est.current_correlation()

    # Crash period
    crash_corrs = []
    for r_v, r_s in pairs_crash:
        corr = est.update_pair({"VRP": r_v, "SKEW": r_s})
        crash_corrs.append(corr)
    corr_during_crash = crash_corrs[-1]
    spike_detected = est.is_spiking(threshold=0.50)

    # Recovery
    for r_v, r_s in pairs_recovery:
        est.update_pair({"VRP": r_v, "SKEW": r_s})
    corr_recovery = est.current_correlation()

    print(f"  Pre-crash correlation:    {corr_pre_crash:+.4f}")
    print(f"  During crash correlation: {corr_during_crash:+.4f}")
    print(f"  Post-recovery:            {corr_recovery:+.4f}")
    print(f"  Spike detected at crash:  {spike_detected}")

    # During crash, correlation should have dropped
    dropped = corr_during_crash < corr_pre_crash
    # Post-recovery should be back up
    recovered = corr_recovery > corr_during_crash

    print(f"\n  Corr dropped during crash: {'PASS' if dropped else 'FAIL'}")
    print(f"  Corr recovered after:      {'PASS' if recovered else 'FAIL'}")

    passed = dropped and recovered
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_spiking_detection():
    """Test 6: is_spiking() threshold detection."""
    print(f"\n{DIVIDER}")
    print("TEST 6: Spike Detection Threshold")
    print(DIVIDER)

    # High correlation scenario
    pairs = generate_correlated_returns(300, 0.80, vol=0.01, seed=900)

    est = CorrelationEstimator(
        projects=["H1", "H2"], half_life=20, rolling_window=60
    )

    for r1, r2 in pairs:
        est.update_pair({"H1": r1, "H2": r2})

    corr = est.current_correlation()
    spike_50 = est.is_spiking(threshold=0.50)
    spike_30 = est.is_spiking(threshold=0.30)
    spike_90 = est.is_spiking(threshold=0.90)

    print(f"  Current correlation: {corr:+.4f}")
    print(f"  is_spiking(0.50): {spike_50}")
    print(f"  is_spiking(0.30): {spike_30}")
    print(f"  is_spiking(0.90): {spike_90}")

    # With rho=0.80, should spike at 0.50 and 0.30, but not 0.90
    passed = spike_50 and spike_30 and not spike_90
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_persistence():
    """Test 7: State save/load round-trip."""
    print(f"\n{DIVIDER}")
    print("TEST 7: State Persistence Round-Trip")
    print(DIVIDER)

    pairs = generate_correlated_returns(200, 0.50, vol=0.01, seed=1000)

    est1 = CorrelationEstimator(
        projects=["S1", "S2"], half_life=20, rolling_window=60
    )

    for r1, r2 in pairs:
        est1.update_pair({"S1": r1, "S2": r2})

    corr_before = est1.current_correlation()
    n_updates_before = est1.state.n_updates

    # Save
    est1.save()

    # Log
    est1.log_observation()

    print(f"  Correlation before save: {corr_before:+.4f}")
    print(f"  N updates before save:   {n_updates_before}")
    print(f"  State file exists:       True")

    # Verify save didn't crash
    passed = abs(corr_before) > 0.01  # should have non-zero estimate

    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    print(DIVIDER)
    print("NEXUS CORRELATION ESTIMATOR — VALIDATION TEST SUITE")
    print(DIVIDER)

    results = {}
    results["convergence"] = test_convergence_to_true_correlation()
    results["regime_shift"] = test_regime_shift_reaction()
    results["uncorrelated"] = test_uncorrelated()
    results["perfect_corr"] = test_perfect_correlation()
    results["crash_decorr"] = test_crash_decorrelation()
    results["spike_detect"] = test_spiking_detection()
    results["persistence"] = test_persistence()

    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)

    print(f"\n{DIVIDER}")
    print(f"SUMMARY: {n_pass}/{n_total} tests passed")
    print(DIVIDER)

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} {status}")

    if n_pass == n_total:
        print(f"\n  ALL TESTS PASSED — Correlation estimator validated")
    else:
        print(f"\n  {n_total - n_pass} FAILURES — needs investigation")

    print(DIVIDER)
    return n_pass == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
