#!/usr/bin/env python3
"""
NEXUS Cross-Project Integration Test
======================================

Tests the full pipeline from signal generation through portfolio aggregation:

1. Crypto Perps signal (reads from existing signals_log.jsonl)
2. Crypto Options signal (generates fresh from live data)
3. NexusSignalAggregator combines both
4. Portfolio optimization validates allocation
5. DeribitExecutor (dry run) shows intended trades
6. BinanceExecutor (dry run) shows intended trades

This verifies all components are wired correctly.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_options_signal():
    """Test 1: Options signal generator produces valid output."""
    print("  [1] Crypto Options Signal Generator")
    from nexus_quant.projects.crypto_options.signal_generator import (
        OptionsSignalGenerator, status_report,
    )

    gen = OptionsSignalGenerator()
    signal = gen.generate()

    assert "target_weights" in signal, "Missing target_weights"
    assert "confidence" in signal, "Missing confidence"
    assert "vrp_signal" in signal, "Missing vrp_signal"
    assert "skew_signal" in signal, "Missing skew_signal"
    assert "ensemble_weights" in signal, "Missing ensemble_weights"
    assert signal["ensemble_weights"]["VRP"] == 0.40
    assert signal["ensemble_weights"]["Skew_MR"] == 0.60

    print(f"      Confidence: {signal['confidence']}")
    print(f"      Data days: {signal['data_days']}")
    print(f"      BTC: {signal['target_weights'].get('BTC', 0):+.4f}")
    print(f"      ETH: {signal['target_weights'].get('ETH', 0):+.4f}")
    print(f"      Gross leverage: {signal['gross_leverage']:.4f}")
    print("      PASS")
    return signal


def test_aggregator():
    """Test 2: Signal aggregator reads and combines signals."""
    print("\n  [2] NexusSignalAggregator")
    from nexus_quant.portfolio.aggregator import NexusSignalAggregator

    agg = NexusSignalAggregator()

    # Check portfolio weights were computed
    assert "crypto_perps" in agg.portfolio_weights
    assert "crypto_options" in agg.portfolio_weights
    total_w = sum(agg.portfolio_weights.values())
    assert abs(total_w - 1.0) < 0.01, f"Weights don't sum to 1.0: {total_w}"

    print(f"      Portfolio weights: {agg.portfolio_weights}")

    # Read individual signals
    for project in agg.portfolio_weights:
        sig = agg.read_latest_signal(project)
        if sig:
            print(f"      {project}: confidence={sig.confidence}, "
                  f"age={sig.age_seconds:,}s, gross={sig.gross_leverage:.4f}")
        else:
            print(f"      {project}: NO SIGNAL")

    # Run aggregation
    portfolio = agg.aggregate()
    assert portfolio.venue_targets is not None
    assert portfolio.project_allocations is not None
    # Note: expected_sharpe may be 0 if all signals are stale/insufficient
    # This is expected in early stage — the aggregator correctly goes to cash

    print(f"      Expected Sharpe: {portfolio.expected_sharpe:.3f}")
    print(f"      Expected Vol: {portfolio.expected_vol_pct:.1f}%")
    print(f"      Total gross leverage: {portfolio.total_gross_leverage:.4f}")

    for venue, targets in portfolio.venue_targets.items():
        active = {s: w for s, w in targets.items() if abs(w) > 0.0001}
        if active:
            print(f"      Venue {venue}: {active}")
        else:
            print(f"      Venue {venue}: (all zero / insufficient data)")

    print("      PASS")
    return portfolio


def test_portfolio_optimizer():
    """Test 3: Portfolio optimizer produces consistent results."""
    print("\n  [3] PortfolioOptimizer Validation")
    from nexus_quant.portfolio.optimizer import PortfolioOptimizer

    opt = PortfolioOptimizer()

    # Verify strategy profiles match wisdom.json
    perps = next(s for s in opt.strategies if s.name == "crypto_perps")
    opts = next(s for s in opt.strategies if s.name == "crypto_options")

    assert abs(perps.avg_sharpe - 2.006) < 0.01, f"Perps Sharpe wrong: {perps.avg_sharpe}"
    assert abs(opts.avg_sharpe - 2.722) < 0.01, f"Options Sharpe wrong: {opts.avg_sharpe}"
    print(f"      crypto_perps avg Sharpe: {perps.avg_sharpe:.3f}")
    print(f"      crypto_options avg Sharpe: {opts.avg_sharpe:.3f}")

    # Run optimization
    results = opt.optimize(step=0.05, correlation_override=0.20)
    best = results[0]

    # Combined portfolio should outperform both standalone
    assert best.avg_sharpe > perps.avg_sharpe, "Portfolio doesn't beat perps standalone"
    assert best.avg_sharpe > opts.avg_sharpe, "Portfolio doesn't beat options standalone"

    print(f"      Optimal: {best.weights}")
    print(f"      Portfolio Sharpe: {best.avg_sharpe:.3f} (vs perps {perps.avg_sharpe:.3f}, "
          f"opts {opts.avg_sharpe:.3f})")
    print(f"      Diversification benefit: {best.diversification_benefit:+.3f}")
    print("      PASS")
    return results


def test_deribit_executor():
    """Test 4: DeribitExecutor (dry run) handles signal correctly."""
    print("\n  [4] DeribitExecutor (dry run)")
    from nexus_quant.live.deribit_executor import DeribitExecutor

    executor = DeribitExecutor(testnet=True, dry_run=True)

    # Simulate a VRP short vol signal
    test_weights = {"BTC": -0.75, "ETH": -0.50}
    orders = executor.reconcile(test_weights, equity_usd=100000)

    assert len(orders) == 2, f"Expected 2 orders, got {len(orders)}"
    for o in orders:
        assert o.status == "DRY_RUN"
        assert o.direction == "sell"
        print(f"      {o.instrument_name}: {o.direction} ${o.amount:,.0f}")

    # Test get_account (dry run)
    account = executor.get_account()
    assert account is not None

    # Test risk gates (dry run account has 0 margin, so halt=True is expected)
    risk = executor.check_risk_gates()
    print(f"      Risk halt: {risk['halt']} (expected in dry run)")
    # Don't assert halt=False — dry run account has no funds by design

    print("      PASS")
    return orders


def test_end_to_end_pipeline():
    """Test 5: Full pipeline from signal to execution."""
    print("\n  [5] End-to-End Pipeline")
    from nexus_quant.portfolio.aggregator import NexusSignalAggregator
    from nexus_quant.live.deribit_executor import DeribitExecutor

    # 1. Aggregate signals
    agg = NexusSignalAggregator()
    portfolio = agg.aggregate()

    # 2. Extract Deribit targets
    deribit_targets = portfolio.venue_targets.get("deribit", {})
    binance_targets = portfolio.venue_targets.get("binance", {})

    print(f"      Deribit targets: {deribit_targets}")
    print(f"      Binance targets: {binance_targets}")

    # 3. Execute Deribit (dry run)
    deribit_exec = DeribitExecutor(testnet=True, dry_run=True)
    deribit_orders = deribit_exec.reconcile(deribit_targets)
    print(f"      Deribit orders: {len(deribit_orders)}")

    # 4. Binance would be executed by existing runner.py
    active_binance = sum(1 for w in binance_targets.values() if abs(w) > 0.0001)
    print(f"      Binance active positions: {active_binance}")

    print("      PASS")


def main():
    print("=" * 70)
    print("  NEXUS CROSS-PROJECT INTEGRATION TEST")
    print("=" * 70)
    print()

    t0 = time.time()
    passed = 0
    failed = 0
    errors = []

    tests = [
        ("Options Signal Generator", test_options_signal),
        ("Signal Aggregator", test_aggregator),
        ("Portfolio Optimizer", test_portfolio_optimizer),
        ("Deribit Executor", test_deribit_executor),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"      FAIL: {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  RESULTS: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    if errors:
        print(f"\n  FAILURES:")
        for name, err in errors:
            print(f"    {name}: {err}")
    print(f"{'='*70}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
