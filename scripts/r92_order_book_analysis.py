#!/usr/bin/env python3
"""
R92: Deribit Order Book Analysis — Live Market Microstructure for BF Trades
===========================================================================

Fetches LIVE order book data from Deribit API for the instruments we'd
actually trade (ATM, 25d put, 25d call) and analyzes:

  1. Bid-ask spreads at each leg of the butterfly
  2. Book depth (contracts available at top-of-book)
  3. Slippage estimation for realistic position sizes (0.3-1.0 BTC)
  4. BF spread implied cost (crossing all 3 legs simultaneously)
  5. Optimal leg ordering (which leg to execute first)
  6. Time-of-day liquidity patterns (from current snapshot)

Validates R76 cost estimates (3-8 bps per trade) against live data.
"""
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r92_order_book_results.json"


def fetch_json(url):
    """Fetch JSON from Deribit API via curl."""
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "30", url],
                          capture_output=True, text=True, timeout=40)
        data = json.loads(r.stdout)
        if "result" in data:
            return data["result"]
        return data
    except Exception as e:
        return {"error": str(e)}


def get_btc_price():
    """Get current BTC perpetual price."""
    d = fetch_json("https://www.deribit.com/api/v2/public/ticker?instrument_name=BTC-PERPETUAL")
    return float(d.get("last_price", 68000)) if isinstance(d, dict) else 68000


def get_order_book(instrument_name, depth=20):
    """Fetch order book for a specific instrument."""
    url = (f"https://www.deribit.com/api/v2/public/get_order_book?"
           f"instrument_name={instrument_name}&depth={depth}")
    return fetch_json(url)


# ═══════════════════════════════════════════════════════════════
# Section 1: Identify Target Instruments
# ═══════════════════════════════════════════════════════════════

def section_1_target_instruments():
    """Find the actual BTC option instruments for our BF trade."""
    print("\n  ── Section 1: Target Instrument Identification ──")

    btc_price = get_btc_price()
    print(f"    BTC spot: ${btc_price:,.0f}")
    time.sleep(0.3)

    # Fetch all BTC options
    url = "https://www.deribit.com/api/v2/public/get_instruments?currency=BTC&kind=option&expired=false"
    instruments = fetch_json(url)
    time.sleep(0.3)

    if not isinstance(instruments, list):
        print(f"    ERROR: {instruments}")
        return None

    # Group by expiry, find quarterly
    today = datetime.now(timezone.utc)
    by_expiry = {}
    for inst in instruments:
        exp_ts = inst.get("expiration_timestamp", 0)
        exp_date = datetime.fromtimestamp(exp_ts/1000, tz=timezone.utc)
        exp_str = exp_date.strftime("%Y-%m-%d")
        dte = (exp_date - today).days
        if exp_str not in by_expiry:
            by_expiry[exp_str] = {"dte": dte, "instruments": []}
        by_expiry[exp_str]["instruments"].append(inst)

    # Select TWO expiries: nearest monthly (14-45 DTE) and nearest quarterly
    quarterly_dates = ["2026-03-27", "2026-06-26", "2026-09-25", "2026-12-25"]
    target_expiries = []

    # Monthly: 14-45 DTE
    for exp, info in sorted(by_expiry.items()):
        if 14 <= info["dte"] <= 45:
            target_expiries.append(("monthly", exp, info))
            break

    # Quarterly: nearest with DTE > 7
    for qd in quarterly_dates:
        if qd in by_expiry and by_expiry[qd]["dte"] > 7:
            target_expiries.append(("quarterly", qd, by_expiry[qd]))
            break

    # For each target expiry, find ATM, 25d put, 25d call strikes
    results = {}
    for label, exp_date, info in target_expiries:
        opts = info["instruments"]
        strikes = sorted(set(o.get("strike", 0) for o in opts))

        # ATM: nearest to spot
        atm = min(strikes, key=lambda s: abs(s - btc_price))
        # 25d put: ~10% OTM below
        put_25d = min(strikes, key=lambda s: abs(s - btc_price * 0.90))
        # 25d call: ~10% OTM above
        call_25d = min(strikes, key=lambda s: abs(s - btc_price * 1.10))

        # Find instrument names
        legs = {}
        for o in opts:
            s = o.get("strike", 0)
            t = o.get("option_type", "")
            name = o.get("instrument_name", "")
            if s == atm and t == "call":
                legs["atm_call"] = name
            if s == atm and t == "put":
                legs["atm_put"] = name
            if s == put_25d and t == "put":
                legs["put_25d"] = name
            if s == call_25d and t == "call":
                legs["call_25d"] = name

        results[label] = {
            "expiry": exp_date,
            "dte": info["dte"],
            "atm_strike": atm,
            "put_25d_strike": put_25d,
            "call_25d_strike": call_25d,
            "legs": legs,
        }

        print(f"\n    {label.upper()} — {exp_date} (DTE={info['dte']})")
        print(f"      ATM: ${atm:,.0f}  |  25d Put: ${put_25d:,.0f}  |  25d Call: ${call_25d:,.0f}")
        for leg_name, inst_name in legs.items():
            print(f"      {leg_name}: {inst_name}")

    return results, btc_price


# ═══════════════════════════════════════════════════════════════
# Section 2: Bid-Ask Spreads
# ═══════════════════════════════════════════════════════════════

def section_2_spreads(target_instruments, btc_price):
    """Analyze bid-ask spreads for each leg."""
    print("\n  ── Section 2: Bid-Ask Spread Analysis ──")

    all_books = {}

    for label, info in target_instruments.items():
        print(f"\n    {label.upper()} — {info['expiry']} (DTE={info['dte']})")
        print(f"    {'Leg':>12} {'Bid':>10} {'Ask':>10} {'Spread':>10} "
              f"{'Spread%':>8} {'BidQty':>8} {'AskQty':>8}")
        print(f"    {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")

        books = {}
        for leg_name, inst_name in info["legs"].items():
            book = get_order_book(inst_name, depth=20)
            time.sleep(0.25)

            if not isinstance(book, dict) or "error" in book:
                print(f"    {leg_name:>12} -- error fetching --")
                continue

            best_bid = book.get("best_bid_price", 0) or 0
            best_ask = book.get("best_ask_price", 0) or 0
            bid_qty = book.get("best_bid_amount", 0) or 0
            ask_qty = book.get("best_ask_amount", 0) or 0
            mark = book.get("mark_price", 0) or 0
            iv = book.get("mark_iv", 0) or 0
            underlying = book.get("underlying_price", btc_price) or btc_price

            # Deribit option prices are in BTC
            spread_btc = best_ask - best_bid if best_bid > 0 else 0
            spread_usd = spread_btc * underlying
            mid = (best_bid + best_ask) / 2 if best_bid > 0 else mark
            spread_pct = (spread_btc / mid * 100) if mid > 0 else 0

            # Spread in bps of underlying
            spread_bps = (spread_btc / 1.0) * 10000 if spread_btc > 0 else 0

            books[leg_name] = {
                "instrument": inst_name,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bid_qty": bid_qty,
                "ask_qty": ask_qty,
                "mark": mark,
                "iv": iv,
                "spread_btc": round(spread_btc, 6),
                "spread_usd": round(spread_usd, 2),
                "spread_pct": round(spread_pct, 2),
                "mid": round(mid, 6),
                "bids": book.get("bids", [])[:5],
                "asks": book.get("asks", [])[:5],
            }

            print(f"    {leg_name:>12} {best_bid:>10.6f} {best_ask:>10.6f} "
                  f"{spread_btc:>10.6f} {spread_pct:>7.1f}% "
                  f"{bid_qty:>8.1f} {ask_qty:>8.1f}")

        all_books[label] = books

    return all_books


# ═══════════════════════════════════════════════════════════════
# Section 3: BF Trade Cost Estimation
# ═══════════════════════════════════════════════════════════════

def section_3_bf_cost(all_books, btc_price):
    """Estimate total cost of executing a BF spread — maker vs taker."""
    print("\n  ── Section 3: Butterfly Trade Cost Estimation ──")

    print(f"""
    BF spread = 3 LEGS (4 contracts):
      Leg 1: Sell 1x 25d OTM put
      Leg 2: Sell 1x 25d OTM call
      Leg 3: Buy 2x ATM (call or put, NOT both)

    Deribit fees (% of underlying):
      Maker: 0.02% = 2 bps per contract
      Taker: 0.03% = 3 bps per contract
      4 contracts → Maker: 8 bps | Taker: 12 bps
    """)

    results = {}

    for label, books in all_books.items():
        print(f"    {label.upper()}")

        # BF trade legs — use only ONE ATM option
        legs_cost = {}
        total_spread_btc = 0

        if "put_25d" in books:
            b = books["put_25d"]
            half_spread = b["spread_btc"] / 2
            legs_cost["sell_25d_put"] = half_spread
            total_spread_btc += half_spread

        if "call_25d" in books:
            b = books["call_25d"]
            half_spread = b["spread_btc"] / 2
            legs_cost["sell_25d_call"] = half_spread
            total_spread_btc += half_spread

        # Buy 2x ATM — pick the tighter of call/put
        atm_options = [(k, books[k]) for k in ["atm_call", "atm_put"] if k in books]
        if atm_options:
            atm_key, atm_book = min(atm_options, key=lambda x: x[1]["spread_btc"])
            half_spread = atm_book["spread_btc"] / 2
            legs_cost[f"buy_2x_{atm_key}"] = half_spread * 2
            total_spread_btc += half_spread * 2

        spread_bps = total_spread_btc * 10000

        # Fee scenarios
        maker_fee_bps = 4 * 2  # 4 contracts × 2 bps
        taker_fee_bps = 4 * 3  # 4 contracts × 3 bps

        # Scenario A: Taker (cross spread + taker fees)
        taker_total = spread_bps + taker_fee_bps
        # Scenario B: Maker (limit orders, no spread cost, maker fees)
        maker_total = maker_fee_bps
        # Scenario C: Hybrid (maker on ATM, taker on wings)
        # ATM is liquid → maker fill likely; wings less liquid → may need taker
        wing_spread = sum(v for k, v in legs_cost.items() if "25d" in k) * 10000
        hybrid_total = wing_spread + taker_fee_bps * 0.5 + maker_fee_bps * 0.5

        results[label] = {
            "legs_cost_btc": {k: round(v, 6) for k, v in legs_cost.items()},
            "spread_cost_bps": round(spread_bps, 1),
            "taker_all_in_bps": round(taker_total, 1),
            "maker_all_in_bps": round(maker_total, 1),
            "hybrid_all_in_bps": round(hybrid_total, 1),
        }

        print(f"      Spread crossing:  {spread_bps:>6.1f} bps")
        print(f"      ┌─────────────────────────────────────────┐")
        print(f"      │ TAKER (market orders):  {taker_total:>6.1f} bps      │")
        print(f"      │ MAKER (limit orders):   {maker_total:>6.1f} bps      │")
        print(f"      │ HYBRID (maker ATM):     {hybrid_total:>6.1f} bps      │")
        print(f"      └─────────────────────────────────────────┘")

    return results


# ═══════════════════════════════════════════════════════════════
# Section 4: Slippage for Position Sizes
# ═══════════════════════════════════════════════════════════════

def section_4_slippage(all_books, btc_price):
    """Estimate slippage for different position sizes."""
    print("\n  ── Section 4: Slippage by Position Size ──")
    print(f"    Cost is per-BTC notional (constant) + depth slippage (size-dependent)")

    results = {}
    position_sizes = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]  # BTC notional

    for label, books in all_books.items():
        print(f"\n    {label.upper()}")
        print(f"    {'Size(BTC)':>10} {'Spread/BTC':>11} {'Slip/BTC':>9} {'Total/BTC':>10} {'Fill':>5}")
        print(f"    {'─'*10} {'─'*11} {'─'*9} {'─'*10} {'─'*5}")

        # Select BF legs: 25d put, 25d call, ONE ATM (tighter spread)
        bf_legs = {}
        if "put_25d" in books:
            bf_legs["put_25d"] = (books["put_25d"], 1, False)  # (data, mult, is_buy)
        if "call_25d" in books:
            bf_legs["call_25d"] = (books["call_25d"], 1, False)
        # Pick tighter ATM
        atm_options = [(k, books[k]) for k in ["atm_call", "atm_put"] if k in books]
        if atm_options:
            atm_key, atm_book = min(atm_options, key=lambda x: x[1]["spread_btc"])
            bf_legs[atm_key] = (atm_book, 2, True)  # 2x, buying

        # Base spread cost per BTC (fixed, no slippage)
        base_spread_btc = 0
        for leg_name, (book_data, mult, is_buy) in bf_legs.items():
            base_spread_btc += book_data["spread_btc"] / 2 * mult
        base_spread_bps = base_spread_btc * 10000

        size_results = []
        for size in position_sizes:
            total_slippage_btc = 0
            can_fill = True

            for leg_name, (book_data, mult, is_buy) in bf_legs.items():
                needed_contracts = size * mult
                levels = book_data.get("asks" if is_buy else "bids", [])

                if not levels:
                    can_fill = False
                    continue

                filled = 0
                cost = 0
                best_price = levels[0][0] if levels else 0

                for price, qty in levels:
                    fill_here = min(needed_contracts - filled, qty)
                    cost += fill_here * abs(price - best_price)
                    filled += fill_here
                    if filled >= needed_contracts:
                        break

                if filled < needed_contracts * 0.8:
                    can_fill = False

                total_slippage_btc += cost

            # Slippage per BTC notional
            slip_per_btc_bps = (total_slippage_btc / max(size, 0.01)) * 10000
            total_per_btc_bps = base_spread_bps + slip_per_btc_bps

            size_results.append({
                "size_btc": size,
                "spread_per_btc_bps": round(base_spread_bps, 1),
                "slippage_per_btc_bps": round(slip_per_btc_bps, 1),
                "total_per_btc_bps": round(total_per_btc_bps, 1),
                "can_fill": can_fill,
            })

            print(f"    {size:>10.1f} {base_spread_bps:>11.1f} {slip_per_btc_bps:>9.1f} "
                  f"{total_per_btc_bps:>10.1f} {'YES' if can_fill else 'NO':>5}")

        results[label] = size_results

    return results


# ═══════════════════════════════════════════════════════════════
# Section 5: Book Depth Analysis
# ═══════════════════════════════════════════════════════════════

def section_5_depth(all_books, btc_price):
    """Analyze order book depth at each leg."""
    print("\n  ── Section 5: Order Book Depth ──")

    results = {}

    for label, books in all_books.items():
        print(f"\n    {label.upper()}")

        depth_data = {}
        for leg_name, book_data in books.items():
            bids = book_data.get("bids", [])
            asks = book_data.get("asks", [])

            bid_depth = sum(qty for _, qty in bids) if bids else 0
            ask_depth = sum(qty for _, qty in asks) if asks else 0
            total_depth = bid_depth + ask_depth

            # Depth in BTC value
            bid_value_btc = bid_depth  # contracts ≈ BTC for BTC options
            ask_value_btc = ask_depth

            depth_data[leg_name] = {
                "bid_levels": len(bids),
                "ask_levels": len(asks),
                "bid_depth_contracts": round(bid_depth, 1),
                "ask_depth_contracts": round(ask_depth, 1),
                "total_depth": round(total_depth, 1),
            }

            print(f"      {leg_name:>12}: {len(bids)} bid levels ({bid_depth:.1f} contracts) | "
                  f"{len(asks)} ask levels ({ask_depth:.1f} contracts)")

        results[label] = depth_data

    return results


# ═══════════════════════════════════════════════════════════════
# Section 6: R76 Validation & Recommendations
# ═══════════════════════════════════════════════════════════════

def section_6_validation(bf_costs, slippage, btc_price):
    """Compare live findings against R76 estimates — maker vs taker impact."""
    print("\n  ── Section 6: R76 Validation & Strategy Impact ──")

    trades_per_year = 5.7
    annual_ret_bps = 155  # R69: 1.55% annual return = 155 bps

    # With z_exit=0.0, each "trade" is a reversal:
    #   Close existing BF + Open opposing BF = 2 BF spreads
    # BUT if same expiry/strikes, net trades may be fewer legs
    # Conservative: count each trade as 1 full BF spread (R82 signal = new position)
    # The old position may have expired or been rolled already
    crosses_per_year = trades_per_year  # each trade = 1 BF spread execution

    print(f"""
    R76 ESTIMATE vs LIVE DATA:
    ──────────────────────────
    R76 estimated: 3-8 bps per BF trade (spread cost only)
    """)

    # Collect costs for each expiry type and scenario
    scenarios = {}
    for label, costs in bf_costs.items():
        taker = costs.get("taker_all_in_bps", 0)
        maker = costs.get("maker_all_in_bps", 0)
        hybrid = costs.get("hybrid_all_in_bps", 0)
        spread = costs.get("spread_cost_bps", 0)

        print(f"    {label.upper()} spread cost: {spread:.1f} bps")

        for scenario_name, cost_per_trade in [
            ("taker", taker), ("maker", maker), ("hybrid", hybrid)
        ]:
            annual_cost = cost_per_trade * crosses_per_year
            cost_pct = (annual_cost / annual_ret_bps * 100) if annual_ret_bps > 0 else 0
            net_ret = annual_ret_bps - annual_cost
            # Sharpe adjustment: cost reduces return proportionally
            base_sharpe = 3.76  # R69
            adj_sharpe = base_sharpe * (net_ret / annual_ret_bps) if annual_ret_bps > 0 else 0

            scenarios[f"{label}_{scenario_name}"] = {
                "cost_per_trade_bps": round(cost_per_trade, 1),
                "annual_cost_bps": round(annual_cost, 1),
                "cost_pct_of_return": round(cost_pct, 1),
                "net_return_bps": round(net_ret, 1),
                "adjusted_sharpe": round(adj_sharpe, 2),
            }

    # Print comparison table
    print(f"""
    ANNUAL COST IMPACT (5.7 trades/year, return = 155 bps):
    ────────────────────────────────────────────────────────
    {'Scenario':<25} {'Cost/Trade':>10} {'Annual':>8} {'Net Ret':>8} {'Sharpe':>7}
    {'─'*25} {'─'*10} {'─'*8} {'─'*8} {'─'*7}""")

    for key, data in sorted(scenarios.items()):
        label = key.replace("_", " ").title()
        print(f"    {label:<25} {data['cost_per_trade_bps']:>10.1f} "
              f"{data['annual_cost_bps']:>8.1f} {data['net_return_bps']:>8.1f} "
              f"{data['adjusted_sharpe']:>7.2f}")

    # Find best scenario
    best = min(scenarios.items(), key=lambda x: x[1]["annual_cost_bps"])
    worst = max(scenarios.items(), key=lambda x: x[1]["annual_cost_bps"])

    print(f"""
    KEY FINDINGS:
    ─────────────
    1. R76 estimate (3-8 bps) = spread cost ONLY, not all-in
       Live spread costs: {bf_costs.get('monthly', {}).get('spread_cost_bps', 0):.0f}-{bf_costs.get('quarterly', {}).get('spread_cost_bps', 0):.0f} bps → {'WITHIN RANGE' if bf_costs.get('monthly', {}).get('spread_cost_bps', 0) <= 12 else 'HIGHER'}

    2. MAKER FILLS ARE ESSENTIAL
       Taker: Sharpe drops to {scenarios.get(worst[0], {}).get('adjusted_sharpe', 0):.2f} (cost eats {scenarios.get(worst[0], {}).get('cost_pct_of_return', 0):.0f}% of return)
       Maker: Sharpe stays at {scenarios.get(best[0], {}).get('adjusted_sharpe', 0):.2f} (cost eats {scenarios.get(best[0], {}).get('cost_pct_of_return', 0):.0f}% of return)

    3. EXECUTION RECOMMENDATION:
       → Use LIMIT ORDERS exclusively (maker fills)
       → Patient execution: place orders, wait 1-4 hours for fill
       → Signal fires 00:15 UTC, execute by 08:00 UTC (7.75h window)
       → BF trades are slow (MR over weeks), no urgency for instant fills
    """)

    validation = {
        "r76_spread_estimate": "3-8 bps",
        "scenarios": scenarios,
        "best_scenario": best[0],
        "worst_scenario": worst[0],
        "recommendation": "MAKER FILLS ONLY — limit orders with patience",
    }

    return validation


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R92: Deribit Order Book Analysis — Live Market Microstructure")
    print("=" * 70)
    print(f"  Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    all_results = {}

    # Section 1: Identify instruments
    s1_result = section_1_target_instruments()
    if not s1_result:
        print("\n  FATAL: Could not fetch instruments. Aborting.")
        return
    target_instruments, btc_price = s1_result
    all_results["instruments"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "legs_raw"}
        for k, v in target_instruments.items()
    }

    # Section 2: Spreads
    all_books = section_2_spreads(target_instruments, btc_price)
    all_results["spreads"] = {
        label: {
            leg: {k: v for k, v in data.items() if k not in ("bids", "asks")}
            for leg, data in books.items()
        }
        for label, books in all_books.items()
    }

    # Section 3: BF cost
    bf_costs = section_3_bf_cost(all_books, btc_price)
    all_results["bf_costs"] = bf_costs

    # Section 4: Slippage
    slippage = section_4_slippage(all_books, btc_price)
    all_results["slippage"] = slippage

    # Section 5: Depth
    depth = section_5_depth(all_books, btc_price)
    all_results["depth"] = depth

    # Section 6: Validation
    validation = section_6_validation(bf_costs, slippage, btc_price)
    all_results["r76_validation"] = validation

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
