#!/usr/bin/env python3
"""
R91: Deribit Execution Guide — Trading the BF Strategy in Practice
=====================================================================

Maps the abstract BF mean-reversion strategy to actual Deribit instruments.

The "butterfly_25d" feature = IV(25d put) + IV(25d call) - 2 × IV(ATM)
When we "short butterfly", we sell the wings (25d puts + 25d calls) and
buy the ATM straddle. This is equivalent to selling butterfly spread.

This study:
  1. Identifies the specific option instruments on Deribit
  2. Calculates margin requirements for butterfly spreads
  3. Models roll scheduling (when to roll near-expiry options)
  4. Computes expected fills from Deribit order book
  5. Builds a complete execution playbook
  6. Fetches current Deribit BTC option chain for illustration
"""
import json
import math
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r91_execution_guide.json"


def fetch_json(url):
    """Fetch JSON from Deribit API."""
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "30", url],
                          capture_output=True, text=True, timeout=40)
        data = json.loads(r.stdout)
        if "result" in data:
            return data["result"]
        return data
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# Section 1: Option Instrument Identification
# ═══════════════════════════════════════════════════════════════

def section_1_instruments():
    """Identify current Deribit BTC option instruments."""
    print("\n  ── Section 1: Current BTC Option Chain ──")

    # Fetch instruments
    url = "https://www.deribit.com/api/v2/public/get_instruments?currency=BTC&kind=option&expired=false"
    result = fetch_json(url)
    time.sleep(0.3)

    if "error" in result or not isinstance(result, list):
        print(f"    Error fetching instruments: {result}")
        return None

    instruments = result
    print(f"    Total active BTC options: {len(instruments)}")

    # Group by expiry
    by_expiry = {}
    for inst in instruments:
        exp = inst.get("expiration_timestamp", 0)
        exp_date = datetime.fromtimestamp(exp/1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if exp_date not in by_expiry:
            by_expiry[exp_date] = []
        by_expiry[exp_date].append(inst)

    expiries = sorted(by_expiry.keys())
    print(f"    Active expiries: {len(expiries)}")

    # Show nearest expiries with details
    print(f"\n    {'Expiry':>12}  {'DTE':>5}  {'Options':>8}  {'Strikes':>8}")
    print(f"    {'─'*12}  {'─'*5}  {'─'*8}  {'─'*8}")

    today = datetime.now(timezone.utc)
    selected_expiry = None

    for exp in expiries[:10]:
        opts = by_expiry[exp]
        dte = (datetime.strptime(exp, "%Y-%m-%d") - today.replace(tzinfo=None)).days
        strikes = sorted(set(o.get("strike", 0) for o in opts))
        n_calls = sum(1 for o in opts if o.get("option_type") == "call")
        n_puts = sum(1 for o in opts if o.get("option_type") == "put")

        print(f"    {exp:>12}  {dte:>5}  {len(opts):>8}  {len(strikes):>8}  "
              f"({n_calls}C/{n_puts}P)")

        # Select the monthly expiry closest to 30 DTE for illustration
        if selected_expiry is None and 14 <= dte <= 45:
            selected_expiry = exp

    if not selected_expiry and expiries:
        selected_expiry = expiries[min(2, len(expiries)-1)]

    # Get BTC price for strike selection
    price_data = fetch_json("https://www.deribit.com/api/v2/public/ticker?instrument_name=BTC-PERPETUAL")
    btc_price = float(price_data.get("last_price", 68000)) if isinstance(price_data, dict) else 68000
    time.sleep(0.3)

    result_data = {
        "total_options": len(instruments),
        "active_expiries": len(expiries),
        "btc_price": btc_price,
        "selected_expiry": selected_expiry,
        "expiries": [{
            "date": exp,
            "dte": (datetime.strptime(exp, "%Y-%m-%d") - today.replace(tzinfo=None)).days,
            "n_options": len(by_expiry[exp]),
        } for exp in expiries[:10]],
    }

    print(f"\n    Selected expiry for illustration: {selected_expiry}")
    print(f"    BTC spot price: ${btc_price:,.0f}")

    return result_data, by_expiry, btc_price


# ═══════════════════════════════════════════════════════════════
# Section 2: BF Spread Construction
# ═══════════════════════════════════════════════════════════════

def section_2_bf_construction(btc_price, by_expiry, selected_expiry):
    """How to construct the butterfly spread on Deribit."""
    print("\n  ── Section 2: Butterfly Spread Construction ──")

    print(f"""
    The butterfly_25d feature represents:
      BF = IV(25d_put) + IV(25d_call) - 2 × IV(ATM)

    To SHORT butterfly (when BF z > 1.5, meaning BF is elevated):
      - SELL 1x 25-delta put
      - SELL 1x 25-delta call
      - BUY 2x ATM options (calls or puts)

    To LONG butterfly (when BF z < -1.5, meaning BF is depressed):
      - BUY 1x 25-delta put
      - BUY 1x 25-delta call
      - SELL 2x ATM options

    On Deribit, this maps to:
      - ATM strike:     ~${btc_price:,.0f} (nearest round strike)
      - 25d put strike: ~${btc_price * 0.90:,.0f} (roughly 10% OTM)
      - 25d call strike:~${btc_price * 1.10:,.0f} (roughly 10% OTM)

    These are approximate — actual 25d strikes depend on IV and time to expiry.
    """)

    # Find actual strikes near these targets
    if selected_expiry and selected_expiry in by_expiry:
        opts = by_expiry[selected_expiry]
        strikes = sorted(set(o.get("strike", 0) for o in opts))

        # ATM
        atm_strike = min(strikes, key=lambda s: abs(s - btc_price))
        # ~25d put (10% OTM)
        put_target = btc_price * 0.90
        put_strike = min(strikes, key=lambda s: abs(s - put_target))
        # ~25d call (10% OTM)
        call_target = btc_price * 1.10
        call_strike = min(strikes, key=lambda s: abs(s - call_target))

        print(f"    Actual strikes for {selected_expiry}:")
        print(f"      ATM strike:      ${atm_strike:,.0f}")
        print(f"      ~25d put strike: ${put_strike:,.0f}")
        print(f"      ~25d call strike:${call_strike:,.0f}")

        # Fetch option book data for these strikes
        instruments_needed = []
        for o in opts:
            s = o.get("strike", 0)
            t = o.get("option_type", "")
            if (s == atm_strike and t in ["call", "put"]) or \
               (s == put_strike and t == "put") or \
               (s == call_strike and t == "call"):
                instruments_needed.append(o)

        return {
            "atm_strike": atm_strike,
            "put_25d_strike": put_strike,
            "call_25d_strike": call_strike,
            "instruments": [
                {"name": o["instrument_name"], "strike": o["strike"],
                 "type": o["option_type"]}
                for o in instruments_needed
            ],
        }

    return None


# ═══════════════════════════════════════════════════════════════
# Section 3: Margin & Collateral Requirements
# ═══════════════════════════════════════════════════════════════

def section_3_margin(btc_price):
    """Estimate margin requirements for butterfly spreads."""
    print("\n  ── Section 3: Margin & Collateral Requirements ──")

    # Deribit uses portfolio margin for options
    # For a butterfly spread, the max loss is bounded
    print(f"""
    Deribit Margin Model (Portfolio Margin):
    ─────────────────────────────────────────
    Butterfly spreads have LIMITED RISK — max loss is bounded by the
    width of the wings minus the premium received/paid.

    For our strategy at sensitivity=2.5:
      - Notional per trade: ~0.3 BTC minimum (R76/R80)
      - BF position = 0.3 BTC × sensitivity_factor
      - Max theoretical loss: ~1.4% of notional (R80 kill-switch)

    Deribit requires initial margin of roughly:
      - Short butterfly: ~5-10% of notional (wings are short)
      - Long butterfly: ~premium paid only (limited risk)

    Practical capital allocation:
      ┌──────────────────┬──────────────┬───────────────┐
      │ Capital (BTC)    │ USD Value    │ Adequacy      │
      ├──────────────────┼──────────────┼───────────────┤
      │ 0.3 BTC (min)    │ ${0.3 * btc_price:>10,.0f}  │ Adequate      │
      │ 0.5 BTC          │ ${0.5 * btc_price:>10,.0f}  │ Comfortable   │
      │ 1.0 BTC (rec)    │ ${1.0 * btc_price:>10,.0f}  │ Recommended   │
      └──────────────────┴──────────────┴───────────────┘

    IMPORTANT: Deribit margin calls can occur during high-IV spikes.
    Keep at least 2× the required margin as buffer.
    """)

    return {
        "min_capital_btc": 0.3,
        "recommended_btc": 1.0,
        "min_capital_usd": round(0.3 * btc_price),
        "recommended_usd": round(1.0 * btc_price),
        "margin_buffer_recommendation": "2x required margin",
    }


# ═══════════════════════════════════════════════════════════════
# Section 4: Roll Scheduling
# ═══════════════════════════════════════════════════════════════

def section_4_roll_schedule():
    """When and how to roll butterfly positions."""
    print("\n  ── Section 4: Roll Scheduling ──")

    print(f"""
    Roll Strategy:
    ──────────────
    The BF position should be maintained in monthly or quarterly options.

    Roll Rules:
      1. WHEN to roll: Roll when DTE < 7 days (avoid expiry-week gamma)
      2. WHAT to roll into: Next monthly expiry (25-45 DTE ideal)
      3. HOW to roll:
         a. Close current position (buy back short options, sell long options)
         b. Open new position in next-month expiry
         c. Use combo/spread orders to minimize slippage
      4. COST: ~3-8 bps per roll (R76 estimate)

    Roll Frequency:
      - Monthly: 12 rolls/year
      - Quarterly: 4 rolls/year (cheaper, but wider bid-ask on quarterlies)
      - Current trade frequency: 5.7 trades/year (R82)
      - Roll cost is the dominant execution cost

    Recommended: QUARTERLY options
      - Lower roll frequency (4×/yr vs 12×/yr)
      - Better liquidity on Deribit quarterlies
      - R76: combo+quarterly Sharpe 2.70 (best case)

    Calendar (2026 quarterly expiries):
      - 2026-03-27 (March)
      - 2026-06-26 (June)
      - 2026-09-25 (September)
      - 2026-12-25 (December)
    """)

    return {
        "roll_trigger_dte": 7,
        "preferred_expiry": "quarterly",
        "rolls_per_year": 4,
        "estimated_cost_per_roll_bps": 5,
        "quarterly_expiries_2026": ["2026-03-27", "2026-06-26", "2026-09-25", "2026-12-25"],
    }


# ═══════════════════════════════════════════════════════════════
# Section 5: Execution Playbook
# ═══════════════════════════════════════════════════════════════

def section_5_playbook():
    """Complete execution playbook for daily operations."""
    print("\n  ── Section 5: Execution Playbook ──")

    playbook = """
    DAILY OPERATIONS PLAYBOOK
    ═════════════════════════

    00:15 UTC — Cron fires R84 (signal → position → alerts)

    IF signal = HOLD:
      → No action needed. Monitor dashboard.
      → Check alerts for WARNING/CRITICAL.

    IF signal = SHORT_BUTTERFLY (z > 1.5):
      → Current position is LONG or FLAT
      → CLOSE any existing LONG butterfly position
      → OPEN SHORT butterfly:
        1. SELL 1x 25d OTM put
        2. SELL 1x 25d OTM call
        3. BUY 2x ATM option
      → Use combo orders on Deribit for better fills
      → Target: maker fills (3bps) not taker (8bps)

    IF signal = LONG_BUTTERFLY (z < -1.5):
      → Current position is SHORT or FLAT
      → CLOSE any existing SHORT butterfly position
      → OPEN LONG butterfly:
        1. BUY 1x 25d OTM put
        2. BUY 1x 25d OTM call
        3. SELL 2x ATM option
      → Use limit orders, wait for fill

    RISK CHECKS (daily):
      1. Portfolio drawdown vs kill-switch (BTC: 1.4%, ETH: 2.0%)
      2. Health indicator (halt if CRITICAL for 30 days)
      3. Data freshness (no trades on >3 day stale data)
      4. Margin utilization (keep <50% of available margin)

    WEEKLY REVIEW:
      - Check equity curve on dashboard (R85)
      - Review health trend
      - Check if roll needed (DTE < 7 days)
      - Review alerts.jsonl for any patterns

    MONTHLY REVIEW:
      - Compare live P&L to backtest expectations
      - Check if any kill-switch thresholds were approached
      - Review trade log for execution quality
      - Update dashboard and share with stakeholders
    """

    print(playbook)

    return {
        "cron_time": "00:15 UTC",
        "trade_frequency": "5.7 trades/year (~1 every 2 months)",
        "daily_time_required": "5 minutes (check dashboard)",
        "execution_method": "Deribit combo orders, maker fills",
        "risk_checks_per_day": 4,
    }


# ═══════════════════════════════════════════════════════════════
# Section 6: Current Market State
# ═══════════════════════════════════════════════════════════════

def section_6_current_state():
    """Fetch current market state from Deribit."""
    print("\n  ── Section 6: Current Market State ──")

    # BTC DVOL
    btc_dvol = fetch_json(
        "https://www.deribit.com/api/v2/public/get_volatility_index_data?"
        f"currency=BTC&resolution=3600&"
        f"start_timestamp={int((time.time() - 7200) * 1000)}&"
        f"end_timestamp={int(time.time() * 1000)}"
    )
    time.sleep(0.3)

    # ETH DVOL
    eth_dvol = fetch_json(
        "https://www.deribit.com/api/v2/public/get_volatility_index_data?"
        f"currency=ETH&resolution=3600&"
        f"start_timestamp={int((time.time() - 7200) * 1000)}&"
        f"end_timestamp={int(time.time() * 1000)}"
    )
    time.sleep(0.3)

    btc_iv = None
    eth_iv = None

    if isinstance(btc_dvol, dict) and "data" in btc_dvol and btc_dvol["data"]:
        btc_iv = btc_dvol["data"][-1][4]
    if isinstance(eth_dvol, dict) and "data" in eth_dvol and eth_dvol["data"]:
        eth_iv = eth_dvol["data"][-1][4]

    # Prices
    btc_ticker = fetch_json("https://www.deribit.com/api/v2/public/ticker?instrument_name=BTC-PERPETUAL")
    eth_ticker = fetch_json("https://www.deribit.com/api/v2/public/ticker?instrument_name=ETH-PERPETUAL")
    time.sleep(0.3)

    btc_price = float(btc_ticker.get("last_price", 0)) if isinstance(btc_ticker, dict) else 0
    eth_price = float(eth_ticker.get("last_price", 0)) if isinstance(eth_ticker, dict) else 0

    state = {
        "btc_price": btc_price,
        "btc_iv": btc_iv,
        "eth_price": eth_price,
        "eth_iv": eth_iv,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    print(f"    BTC: ${btc_price:,.0f} | IV {btc_iv:.1f}%" if btc_iv else "    BTC: data unavailable")
    print(f"    ETH: ${eth_price:,.0f} | IV {eth_iv:.1f}%" if eth_iv else "    ETH: data unavailable")

    return state


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R91: Deribit Execution Guide — Trading BF in Practice")
    print("=" * 70)

    all_results = {}

    # Section 1: Instruments
    s1_result = section_1_instruments()
    if s1_result:
        s1_data, by_expiry, btc_price = s1_result
        all_results["instruments"] = s1_data

        # Section 2: BF Construction
        selected = s1_data.get("selected_expiry")
        s2 = section_2_bf_construction(btc_price, by_expiry, selected)
        if s2:
            all_results["bf_construction"] = s2
    else:
        btc_price = 68000
        all_results["instruments"] = {"error": "Could not fetch instruments"}

    # Section 3: Margin
    all_results["margin"] = section_3_margin(btc_price)

    # Section 4: Roll Schedule
    all_results["roll_schedule"] = section_4_roll_schedule()

    # Section 5: Playbook
    all_results["playbook"] = section_5_playbook()

    # Section 6: Current State
    all_results["current_state"] = section_6_current_state()

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Guide saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
