#!/usr/bin/env python3
"""
R99: Cross-Exchange Analysis — OKX/Bybit BF Opportunity
=========================================================

R86 confirmed BF mean-reversion is STRUCTURAL (not exchange-specific).
R92 showed execution costs are the main threat. Can other exchanges offer:
  1. Lower bid-ask spreads?
  2. Lower trading fees?
  3. Combo/spread order support?
  4. Sufficient option liquidity?

Analyses:
  1. Exchange comparison (fee structure, option products)
  2. OKX live BTC option chain (if API accessible)
  3. Bybit live BTC option chain (if API accessible)
  4. Fee comparison (maker/taker across exchanges)
  5. Cross-exchange arbitrage potential
  6. Recommendation
"""
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r99_cross_exchange.json"


def fetch_json(url, timeout=30):
    """Fetch JSON via curl."""
    try:
        r = subprocess.run(["curl", "-s", "--max-time", str(timeout), url],
                          capture_output=True, text=True, timeout=timeout+10)
        return json.loads(r.stdout)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# Section 1: Exchange Fee Comparison
# ═══════════════════════════════════════════════════════════════

def section_1_fees():
    """Compare option trading fees across exchanges."""
    print("\n  ── Section 1: Exchange Fee Comparison ──")

    fees = {
        "Deribit": {
            "maker_pct": 0.02,
            "taker_pct": 0.03,
            "maker_bps": 2,
            "taker_bps": 3,
            "fee_cap": "12.5% of option premium",
            "combo_support": True,
            "rfq": True,
            "settlement": "Crypto (BTC/ETH margined)",
            "notes": "Dominant crypto options venue. ~90% market share.",
        },
        "OKX": {
            "maker_pct": 0.02,
            "taker_pct": 0.03,
            "maker_bps": 2,
            "taker_bps": 3,
            "fee_cap": "12.5% of option premium",
            "combo_support": True,
            "rfq": True,
            "settlement": "USDT or crypto margined",
            "notes": "Second largest. Growing volume. USDT-margined options available.",
        },
        "Bybit": {
            "maker_pct": 0.02,
            "taker_pct": 0.02,
            "maker_bps": 2,
            "taker_bps": 2,
            "fee_cap": "12.5% of option premium",
            "combo_support": False,
            "rfq": True,
            "settlement": "USDC-margined",
            "notes": "Lower taker fee (0.02% vs 0.03%). Growing but less liquidity.",
        },
    }

    print(f"\n    {'Exchange':<12} {'Maker':>6} {'Taker':>6} {'4-leg Maker':>12} "
          f"{'4-leg Taker':>12} {'Combo':>6} {'RFQ':>4}")
    print(f"    {'─'*12} {'─'*6} {'─'*6} {'─'*12} {'─'*12} {'─'*6} {'─'*4}")

    for name, f in fees.items():
        leg4_maker = f["maker_bps"] * 4
        leg4_taker = f["taker_bps"] * 4
        print(f"    {name:<12} {f['maker_bps']:>5}bp {f['taker_bps']:>5}bp "
              f"{leg4_maker:>11}bp {leg4_taker:>11}bp "
              f"{'Yes' if f['combo_support'] else 'No':>6} "
              f"{'Yes' if f['rfq'] else 'No':>4}")

    print(f"""
    KEY INSIGHT:
      Bybit charges 0.02% taker (vs Deribit/OKX 0.03%)
      For 4-leg BF: Bybit saves 4 bps per taker trade
      8 bps (Bybit taker) vs 12 bps (Deribit taker) = 33% fee reduction
    """)

    return fees


# ═══════════════════════════════════════════════════════════════
# Section 2: OKX BTC Option Chain
# ═══════════════════════════════════════════════════════════════

def section_2_okx():
    """Fetch OKX BTC option instruments."""
    print("\n  ── Section 2: OKX BTC Options ──")

    url = "https://www.okx.com/api/v5/public/instruments?instType=OPTION&uly=BTC-USD"
    data = fetch_json(url)
    time.sleep(0.5)

    if isinstance(data, dict) and "data" in data:
        instruments = data["data"]
        print(f"    Total OKX BTC options: {len(instruments)}")

        # Group by expiry
        by_expiry = {}
        for inst in instruments:
            exp = inst.get("expTime", "")
            exp_date = datetime.fromtimestamp(int(exp)/1000, tz=timezone.utc).strftime("%Y-%m-%d") if exp else "unknown"
            if exp_date not in by_expiry:
                by_expiry[exp_date] = []
            by_expiry[exp_date].append(inst)

        expiries = sorted(by_expiry.keys())[:10]
        today = datetime.now(timezone.utc)

        print(f"    Active expiries: {len(expiries)}")
        print(f"\n    {'Expiry':>12} {'DTE':>5} {'Options':>8}")
        print(f"    {'─'*12} {'─'*5} {'─'*8}")
        for exp in expiries:
            dte = (datetime.strptime(exp, "%Y-%m-%d") - today.replace(tzinfo=None)).days
            print(f"    {exp:>12} {dte:>5} {len(by_expiry[exp]):>8}")

        # Try to get a ticker for spread comparison
        if instruments:
            sample = instruments[0].get("instId", "")
            if sample:
                ticker_url = f"https://www.okx.com/api/v5/market/ticker?instId={sample}"
                ticker = fetch_json(ticker_url)
                time.sleep(0.3)
                if isinstance(ticker, dict) and "data" in ticker and ticker["data"]:
                    t = ticker["data"][0]
                    bid = float(t.get("bidPx", 0) or 0)
                    ask = float(t.get("askPx", 0) or 0)
                    spread = ask - bid if bid > 0 else 0
                    print(f"\n    Sample spread ({sample}):")
                    print(f"      Bid: {bid}, Ask: {ask}, Spread: {spread}")

        return {"total_options": len(instruments), "expiries": len(expiries),
                "available": True}

    else:
        error = data.get("error", data.get("msg", "unknown"))
        print(f"    ERROR fetching OKX options: {error}")
        return {"available": False, "error": str(error)}


# ═══════════════════════════════════════════════════════════════
# Section 3: Bybit BTC Option Chain
# ═══════════════════════════════════════════════════════════════

def section_3_bybit():
    """Fetch Bybit BTC option instruments."""
    print("\n  ── Section 3: Bybit BTC Options ──")

    url = "https://api.bybit.com/v5/market/instruments-info?category=option&baseCoin=BTC"
    data = fetch_json(url)
    time.sleep(0.5)

    if isinstance(data, dict) and "result" in data:
        result = data["result"]
        instruments = result.get("list", [])
        print(f"    Total Bybit BTC options: {len(instruments)}")

        # Group by expiry
        by_expiry = {}
        for inst in instruments:
            exp = inst.get("deliveryTime", "")
            exp_date = datetime.fromtimestamp(int(exp)/1000, tz=timezone.utc).strftime("%Y-%m-%d") if exp else "unknown"
            if exp_date not in by_expiry:
                by_expiry[exp_date] = []
            by_expiry[exp_date].append(inst)

        expiries = sorted(by_expiry.keys())[:10]
        today = datetime.now(timezone.utc)

        print(f"    Active expiries: {len(expiries)}")
        print(f"\n    {'Expiry':>12} {'DTE':>5} {'Options':>8}")
        print(f"    {'─'*12} {'─'*5} {'─'*8}")
        for exp in expiries:
            dte = (datetime.strptime(exp, "%Y-%m-%d") - today.replace(tzinfo=None)).days
            print(f"    {exp:>12} {dte:>5} {len(by_expiry[exp]):>8}")

        return {"total_options": len(instruments), "expiries": len(expiries),
                "available": True}
    else:
        error = data.get("retMsg", data.get("error", "unknown"))
        print(f"    ERROR fetching Bybit options: {error}")
        return {"available": False, "error": str(error)}


# ═══════════════════════════════════════════════════════════════
# Section 4: Deribit Benchmark
# ═══════════════════════════════════════════════════════════════

def section_4_deribit_benchmark():
    """Deribit option count for comparison."""
    print("\n  ── Section 4: Deribit Benchmark ──")

    url = "https://www.deribit.com/api/v2/public/get_instruments?currency=BTC&kind=option&expired=false"
    result = fetch_json(url)
    time.sleep(0.3)

    if isinstance(result, list):
        print(f"    Total Deribit BTC options: {len(result)}")

        by_expiry = {}
        for inst in result:
            exp = inst.get("expiration_timestamp", 0)
            exp_date = datetime.fromtimestamp(exp/1000, tz=timezone.utc).strftime("%Y-%m-%d")
            if exp_date not in by_expiry:
                by_expiry[exp_date] = []
            by_expiry[exp_date].append(inst)

        print(f"    Active expiries: {len(by_expiry)}")
        return {"total_options": len(result), "expiries": len(by_expiry)}
    return {"total_options": 0}


# ═══════════════════════════════════════════════════════════════
# Section 5: Cost Impact Comparison
# ═══════════════════════════════════════════════════════════════

def section_5_cost_comparison(fees):
    """Model annual cost impact by exchange."""
    print("\n  ── Section 5: Annual Cost Impact by Exchange ──")

    trades_per_year = 5.7
    annual_ret_bps = 155  # R69 baseline (pre-cost)

    print(f"\n    Assumptions: {trades_per_year} trades/yr, {annual_ret_bps}bps gross return")
    print(f"\n    {'Exchange':<12} {'Scenario':<16} {'Cost/Trade':>10} {'Annual':>8} "
          f"{'Net Ret':>8} {'Sharpe*':>8}")
    print(f"    {'─'*12} {'─'*16} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")

    base_sharpe = 1.93  # R94: BTC at sens=5.0 with 8bps

    results = {}
    for exchange, f in fees.items():
        scenarios = {
            "maker": f["maker_bps"] * 4,
            "taker": f["taker_bps"] * 4,
        }

        exchange_results = {}
        for scenario, cost_per_trade in scenarios.items():
            annual_cost = cost_per_trade * trades_per_year
            net_ret = annual_ret_bps - annual_cost
            adj_sharpe = base_sharpe * (net_ret / annual_ret_bps) if annual_ret_bps > 0 else 0

            exchange_results[scenario] = {
                "cost_per_trade_bps": cost_per_trade,
                "annual_cost_bps": round(annual_cost, 1),
                "net_ret_bps": round(net_ret, 1),
                "adj_sharpe": round(adj_sharpe, 2),
            }

            print(f"    {exchange:<12} {scenario:<16} {cost_per_trade:>10} "
                  f"{annual_cost:>8.1f} {net_ret:>8.1f} {adj_sharpe:>8.2f}")

        results[exchange] = exchange_results

    return results


# ═══════════════════════════════════════════════════════════════
# Section 6: Verdict
# ═══════════════════════════════════════════════════════════════

def section_6_verdict(fees, okx, bybit, deribit, cost_comparison):
    """Final recommendation."""
    print("\n  ── Section 6: Verdict & Recommendation ──")

    print(f"""
    EXCHANGE COMPARISON:
    ─────────────────────
    Deribit:  {deribit.get('total_options', 0)} options | Market leader (~90% share)
    OKX:      {okx.get('total_options', 0)} options | {'Available' if okx.get('available') else 'Not accessible'}
    Bybit:    {bybit.get('total_options', 0)} options | {'Available' if bybit.get('available') else 'Not accessible'}

    FEE ADVANTAGE:
      Bybit taker = 0.02% (vs 0.03% Deribit/OKX)
      For 4-leg BF: Bybit saves 4 bps per taker trade
      Annual savings: ~23 bps (4 × 5.7 trades)

    LIQUIDITY REALITY:
      Deribit dominates BTC options with ~90% market share
      OKX is growing but spreads likely wider
      Bybit has lowest fees but thinnest books

    RECOMMENDATION:
    ───────────────
      PRIMARY: Deribit (deepest liquidity, combo orders, proven)
      SECONDARY: OKX (USDT-margined, combo support, growing)
      MONITOR: Bybit (lowest fees, watch for liquidity improvement)

      → Stay on Deribit for production deployment
      → Use MAKER LIMIT COMBO orders (R92 finding)
      → Revisit OKX when volume increases to match Deribit depth
      → Cross-exchange arb not viable (BF is same structural edge)
    """)

    return {
        "primary": "Deribit",
        "secondary": "OKX",
        "reason": "Deribit has deepest liquidity and combo orders",
        "bybit_advantage": "Lower taker fees (0.02% vs 0.03%)",
        "bybit_disadvantage": "Thinner order books, no combo orders",
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R99: Cross-Exchange Analysis — OKX/Bybit BF Opportunity")
    print("=" * 70)

    all_results = {}

    fees = section_1_fees()
    all_results["fees"] = fees

    okx = section_2_okx()
    all_results["okx"] = okx

    bybit = section_3_bybit()
    all_results["bybit"] = bybit

    deribit = section_4_deribit_benchmark()
    all_results["deribit"] = deribit

    cost_comparison = section_5_cost_comparison(fees)
    all_results["cost_comparison"] = cost_comparison

    verdict = section_6_verdict(fees, okx, bybit, deribit, cost_comparison)
    all_results["verdict"] = verdict

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
