"""
Deribit Instrument Selector
=============================

Selects optimal option instruments for VRP straddle and Skew MR strategies.

Capabilities:
  1. Query available instruments from Deribit public API
  2. Select nearest-month expiry with target DTE (~30 days)
  3. Find ATM strike for straddles (VRP strategy)
  4. Find 25-delta strikes for strangles (Skew MR strategy)
  5. Position rolling: detect when current instruments need rolling

Key design decisions:
  - Uses Deribit public API (no auth needed) for instrument discovery
  - Rounds strikes to Deribit's standard intervals ($500 for BTC, $25 for ETH)
  - Prefers monthly expiries over weeklies for liquidity
  - Selects instruments 25-35 DTE; rolls at <7 DTE remaining

Usage:
    from nexus_quant.live.instrument_selector import InstrumentSelector
    selector = InstrumentSelector()
    straddle = selector.select_straddle("BTC", spot_price=95000.0)
    strangle = selector.select_strangle("BTC", spot_price=95000.0)
"""
from __future__ import annotations

import json
import logging
import math
import ssl
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.instrument_selector")

# Deribit public API
MAINNET_BASE = "https://www.deribit.com/api/v2"
TESTNET_BASE = "https://test.deribit.com/api/v2"

# Strike rounding intervals
STRIKE_INTERVALS = {
    "BTC": [500, 1000, 2000, 5000, 10000],    # BTC uses $500 intervals near ATM
    "ETH": [25, 50, 100, 250, 500],            # ETH uses $25 intervals near ATM
}

# Target DTE for strategy instruments
TARGET_DTE = 30
MIN_DTE_BEFORE_ROLL = 7
MAX_DTE = 45


@dataclass
class DeribitInstrument:
    """A Deribit option instrument."""
    name: str               # e.g. "BTC-28MAR26-96000-C"
    currency: str           # "BTC" or "ETH"
    kind: str               # "option"
    option_type: str        # "call" or "put"
    strike: float
    expiry_timestamp: int   # ms since epoch
    expiry_date: str        # "DDMMMYY" format
    dte: float              # days to expiry
    is_active: bool
    settlement_period: str  # "month" or "week"
    min_trade_amount: float
    tick_size: float
    contract_size: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "currency": self.currency,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiry_date": self.expiry_date,
            "dte": round(self.dte, 1),
            "settlement_period": self.settlement_period,
        }


@dataclass
class StraddleSelection:
    """ATM straddle instrument pair for VRP strategy."""
    call: DeribitInstrument
    put: DeribitInstrument
    strike: float
    expiry_date: str
    dte: float
    distance_from_atm_pct: float  # how far strike is from spot (%)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "straddle",
            "call": self.call.name,
            "put": self.put.name,
            "strike": self.strike,
            "expiry_date": self.expiry_date,
            "dte": round(self.dte, 1),
            "distance_from_atm_pct": round(self.distance_from_atm_pct, 2),
        }


@dataclass
class StrangleSelection:
    """25-delta strangle instrument pair for Skew MR strategy."""
    put: DeribitInstrument       # OTM put (25-delta)
    call: DeribitInstrument      # OTM call (25-delta)
    put_strike: float
    call_strike: float
    expiry_date: str
    dte: float
    width_pct: float             # distance between strikes as % of spot

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "strangle",
            "put": self.put.name,
            "call": self.call.name,
            "put_strike": self.put_strike,
            "call_strike": self.call_strike,
            "expiry_date": self.expiry_date,
            "dte": round(self.dte, 1),
            "width_pct": round(self.width_pct, 2),
        }


class InstrumentSelector:
    """
    Selects optimal Deribit option instruments for trading strategies.

    Queries Deribit public API for available instruments and selects
    the best matches for VRP straddles and Skew MR strangles.
    """

    def __init__(
        self,
        testnet: bool = False,
        target_dte: int = TARGET_DTE,
        min_dte_roll: int = MIN_DTE_BEFORE_ROLL,
        cache_ttl: int = 300,
    ) -> None:
        self.base_url = TESTNET_BASE if testnet else MAINNET_BASE
        self.target_dte = target_dte
        self.min_dte_roll = min_dte_roll
        self.cache_ttl = cache_ttl

        # Cache for instrument lists
        self._cache: Dict[str, Tuple[float, List[DeribitInstrument]]] = {}

    def _get_ssl_context(self) -> ssl.SSLContext:
        """Get SSL context, with fallback for cert issues."""
        if not hasattr(self, "_ssl_ctx"):
            try:
                ctx = ssl.create_default_context()
                # Test if default context works
                self._ssl_ctx = ctx
            except Exception:
                self._ssl_ctx = ssl._create_unverified_context()
        return self._ssl_ctx

    def _public_get(self, method: str, params: Dict[str, Any]) -> Optional[Any]:
        """Make a public GET request to Deribit API."""
        url = f"{self.base_url}/{method}?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "NEXUS-Quant/1.0")
            ctx = self._get_ssl_context()
            try:
                with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                    data = json.loads(resp.read())
                    return data.get("result")
            except (ssl.SSLError, urllib.error.URLError):
                # Retry with unverified context on SSL failures
                ctx = ssl._create_unverified_context()
                self._ssl_ctx = ctx
                with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                    data = json.loads(resp.read())
                    return data.get("result")
        except Exception as e:
            logger.error("Deribit API error (%s): %s", method, e)
            return None

    def get_instruments(
        self, currency: str = "BTC", kind: str = "option",
    ) -> List[DeribitInstrument]:
        """
        Fetch all available instruments for a currency.

        Uses cache to avoid hammering the API.
        """
        cache_key = f"{currency}:{kind}"
        now = time.time()

        # Check cache
        if cache_key in self._cache:
            cache_time, instruments = self._cache[cache_key]
            if now - cache_time < self.cache_ttl:
                return instruments

        result = self._public_get("public/get_instruments", {
            "currency": currency,
            "kind": kind,
            "expired": "false",
        })

        if not result:
            # Return cached data if API fails
            if cache_key in self._cache:
                return self._cache[cache_key][1]
            return []

        now_ms = int(now * 1000)
        instruments = []
        for inst in result:
            expiry_ms = inst.get("expiration_timestamp", 0)
            dte = (expiry_ms - now_ms) / (1000 * 86400)
            if dte <= 0:
                continue

            instruments.append(DeribitInstrument(
                name=inst.get("instrument_name", ""),
                currency=currency,
                kind=kind,
                option_type=inst.get("option_type", ""),
                strike=inst.get("strike", 0.0),
                expiry_timestamp=expiry_ms,
                expiry_date=inst.get("instrument_name", "").split("-")[1] if "-" in inst.get("instrument_name", "") else "",
                dte=dte,
                is_active=inst.get("is_active", False),
                settlement_period=inst.get("settlement_period", ""),
                min_trade_amount=inst.get("min_trade_amount", 0.1),
                tick_size=inst.get("tick_size", 0.0005),
                contract_size=inst.get("contract_size", 1.0),
            ))

        # Cache results
        self._cache[cache_key] = (now, instruments)
        logger.info("Fetched %d %s %s instruments", len(instruments), currency, kind)
        return instruments

    def _select_expiry(
        self, instruments: List[DeribitInstrument],
    ) -> Optional[str]:
        """
        Select the best expiry date for trading.

        Preference order:
        1. Monthly expiry with DTE closest to target (25-35 days ideal)
        2. Any expiry with DTE in [target-10, target+15] range
        """
        # Group by expiry date
        expiry_groups: Dict[str, List[DeribitInstrument]] = {}
        for inst in instruments:
            if inst.dte < 5:
                continue  # skip very near-term
            key = inst.expiry_date
            if key not in expiry_groups:
                expiry_groups[key] = []
            expiry_groups[key].append(inst)

        if not expiry_groups:
            return None

        # Score each expiry
        scored = []
        for expiry_date, group in expiry_groups.items():
            dte = group[0].dte
            is_monthly = group[0].settlement_period == "month"

            # Penalize deviation from target DTE
            dte_penalty = abs(dte - self.target_dte)

            # Bonus for monthly expiries (more liquid)
            monthly_bonus = -5.0 if is_monthly else 0.0

            # Penalize too-short or too-long DTE
            if dte < 10:
                dte_penalty += 20.0
            elif dte > MAX_DTE:
                dte_penalty += 10.0

            score = dte_penalty + monthly_bonus
            scored.append((score, expiry_date, dte))

        scored.sort()
        best_expiry = scored[0][1]
        best_dte = scored[0][2]
        logger.info("Selected expiry %s (DTE=%.1f)", best_expiry, best_dte)
        return best_expiry

    def _find_atm_strike(
        self,
        instruments: List[DeribitInstrument],
        spot_price: float,
    ) -> float:
        """Find the ATM strike closest to spot price."""
        strikes = sorted(set(inst.strike for inst in instruments))
        if not strikes:
            return spot_price

        # Find closest strike
        best = min(strikes, key=lambda k: abs(k - spot_price))
        return best

    def _find_delta_strike(
        self,
        instruments: List[DeribitInstrument],
        spot_price: float,
        iv: float,
        dte: float,
        target_delta: float,
        option_type: str,
    ) -> float:
        """
        Find the strike closest to a target delta.

        Uses Black-Scholes to compute delta for each available strike
        and selects the closest match.
        """
        from nexus_quant.projects.crypto_options.greeks import bs_delta

        T = dte / 365.0
        r = 0.05  # approximate risk-free rate

        best_strike = spot_price
        best_diff = float("inf")

        for inst in instruments:
            if inst.option_type != option_type:
                continue
            try:
                delta = bs_delta(spot_price, inst.strike, T, r, iv, option_type)
                diff = abs(delta - target_delta)
                if diff < best_diff:
                    best_diff = diff
                    best_strike = inst.strike
            except (ValueError, ZeroDivisionError):
                continue

        return best_strike

    def select_straddle(
        self,
        currency: str = "BTC",
        spot_price: Optional[float] = None,
    ) -> Optional[StraddleSelection]:
        """
        Select ATM straddle instruments for VRP strategy.

        Args:
            currency: "BTC" or "ETH"
            spot_price: current spot price (if None, fetched from API)

        Returns:
            StraddleSelection or None if no suitable instruments found
        """
        if spot_price is None:
            spot_price = self._get_spot_price(currency)
            if spot_price is None:
                logger.error("Could not get spot price for %s", currency)
                return None

        instruments = self.get_instruments(currency)
        if not instruments:
            logger.error("No instruments available for %s", currency)
            return None

        # Select expiry
        expiry_date = self._select_expiry(instruments)
        if not expiry_date:
            return None

        # Filter to selected expiry
        expiry_instruments = [i for i in instruments if i.expiry_date == expiry_date]
        dte = expiry_instruments[0].dte

        # Find ATM strike
        atm_strike = self._find_atm_strike(expiry_instruments, spot_price)

        # Find the call and put at ATM strike
        call = next((i for i in expiry_instruments
                     if i.strike == atm_strike and i.option_type == "call"), None)
        put = next((i for i in expiry_instruments
                    if i.strike == atm_strike and i.option_type == "put"), None)

        if call is None or put is None:
            logger.error("Missing ATM call or put at strike %.0f", atm_strike)
            return None

        distance_pct = abs(atm_strike - spot_price) / spot_price * 100

        selection = StraddleSelection(
            call=call,
            put=put,
            strike=atm_strike,
            expiry_date=expiry_date,
            dte=dte,
            distance_from_atm_pct=distance_pct,
        )

        logger.info("Selected straddle: %s strike=%.0f DTE=%.1f dist=%.2f%%",
                     currency, atm_strike, dte, distance_pct)
        return selection

    def select_strangle(
        self,
        currency: str = "BTC",
        spot_price: Optional[float] = None,
        iv: float = 0.60,
        target_delta: float = 0.25,
    ) -> Optional[StrangleSelection]:
        """
        Select 25-delta strangle instruments for Skew MR strategy.

        Args:
            currency: "BTC" or "ETH"
            spot_price: current spot price
            iv: current ATM implied volatility (annualized)
            target_delta: absolute delta for wings (default 0.25)

        Returns:
            StrangleSelection or None
        """
        if spot_price is None:
            spot_price = self._get_spot_price(currency)
            if spot_price is None:
                return None

        instruments = self.get_instruments(currency)
        if not instruments:
            return None

        expiry_date = self._select_expiry(instruments)
        if not expiry_date:
            return None

        expiry_instruments = [i for i in instruments if i.expiry_date == expiry_date]
        dte = expiry_instruments[0].dte

        # Find 25-delta put strike
        put_strike = self._find_delta_strike(
            expiry_instruments, spot_price, iv, dte,
            target_delta=-target_delta, option_type="put",
        )

        # Find 25-delta call strike
        call_strike = self._find_delta_strike(
            expiry_instruments, spot_price, iv, dte,
            target_delta=target_delta, option_type="call",
        )

        put = next((i for i in expiry_instruments
                    if i.strike == put_strike and i.option_type == "put"), None)
        call = next((i for i in expiry_instruments
                     if i.strike == call_strike and i.option_type == "call"), None)

        if put is None or call is None:
            logger.error("Missing 25d strangle instruments (put=%.0f, call=%.0f)",
                         put_strike, call_strike)
            return None

        width_pct = (call_strike - put_strike) / spot_price * 100

        selection = StrangleSelection(
            put=put,
            call=call,
            put_strike=put_strike,
            call_strike=call_strike,
            expiry_date=expiry_date,
            dte=dte,
            width_pct=width_pct,
        )

        logger.info("Selected strangle: %s put=%.0f call=%.0f width=%.1f%% DTE=%.1f",
                     currency, put_strike, call_strike, width_pct, dte)
        return selection

    def needs_rolling(
        self,
        current_expiry_date: str,
        instruments: Optional[List[DeribitInstrument]] = None,
    ) -> bool:
        """
        Check if current instruments need rolling to next expiry.

        Returns True if current instruments have < min_dte_roll days remaining.
        """
        if instruments is None:
            # Can't check without instruments
            return False

        for inst in instruments:
            if inst.expiry_date == current_expiry_date:
                if inst.dte < self.min_dte_roll:
                    logger.warning(
                        "Position in %s needs rolling: %.1f DTE < %d threshold",
                        inst.name, inst.dte, self.min_dte_roll,
                    )
                    return True
                return False

        # If we can't find the expiry in current instruments, assume it expired
        return True

    def _get_spot_price(self, currency: str) -> Optional[float]:
        """Fetch current spot price from Deribit perpetual."""
        ticker_name = f"{currency}-PERPETUAL"
        result = self._public_get("public/ticker", {
            "instrument_name": ticker_name,
        })
        if result:
            return result.get("last_price")
        return None

    def get_spot_and_iv(self, currency: str = "BTC") -> Optional[Dict[str, float]]:
        """
        Fetch current spot price and ATM IV from Deribit.

        Returns dict with: spot, iv_atm, funding_rate
        """
        # Spot from perpetual
        spot = self._get_spot_price(currency)
        if spot is None:
            return None

        # ATM IV: find nearest-month ATM option and get mark IV
        instruments = self.get_instruments(currency)
        if not instruments:
            return {"spot": spot, "iv_atm": None, "funding_rate": None}

        expiry_date = self._select_expiry(instruments)
        if not expiry_date:
            return {"spot": spot, "iv_atm": None, "funding_rate": None}

        expiry_instruments = [i for i in instruments if i.expiry_date == expiry_date]
        atm_strike = self._find_atm_strike(expiry_instruments, spot)

        # Get ticker for ATM call to read mark_iv
        atm_call = next(
            (i for i in expiry_instruments
             if i.strike == atm_strike and i.option_type == "call"),
            None,
        )

        iv_atm = None
        if atm_call:
            ticker = self._public_get("public/ticker", {
                "instrument_name": atm_call.name,
            })
            if ticker:
                iv_atm = ticker.get("mark_iv")
                if iv_atm is not None:
                    iv_atm = iv_atm / 100.0  # Deribit returns IV in %

        # Funding rate from perpetual
        perp_ticker = self._public_get("public/ticker", {
            "instrument_name": f"{currency}-PERPETUAL",
        })
        funding_rate = None
        if perp_ticker:
            funding_rate = perp_ticker.get("current_funding")

        return {
            "spot": spot,
            "iv_atm": iv_atm,
            "funding_rate": funding_rate,
        }

    def status(self, currency: str = "BTC") -> str:
        """Human-readable instrument selector status."""
        lines = []
        lines.append(f"Deribit Instrument Selector ({currency})")
        lines.append(f"  Target DTE: {self.target_dte}")
        lines.append(f"  Roll threshold: {self.min_dte_roll} DTE")

        data = self.get_spot_and_iv(currency)
        if data:
            lines.append(f"  Spot: ${data['spot']:,.0f}" if data["spot"] else "  Spot: N/A")
            if data.get("iv_atm"):
                lines.append(f"  ATM IV: {data['iv_atm']:.1%}")
            if data.get("funding_rate"):
                lines.append(f"  Funding rate: {data['funding_rate']:.6f}")

        instruments = self.get_instruments(currency)
        lines.append(f"  Available instruments: {len(instruments)}")

        # Show available expiries
        expiries: Dict[str, int] = {}
        for inst in instruments:
            key = inst.expiry_date
            if key not in expiries:
                expiries[key] = 0
            expiries[key] += 1

        sorted_expiries = sorted(expiries.items(),
                                  key=lambda x: next(i.dte for i in instruments if i.expiry_date == x[0]))
        for exp, count in sorted_expiries[:6]:
            dte = next(i.dte for i in instruments if i.expiry_date == exp)
            period = next(i.settlement_period for i in instruments if i.expiry_date == exp)
            marker = " <<<" if abs(dte - self.target_dte) < 10 else ""
            lines.append(f"    {exp}: {count:3d} instruments, DTE={dte:.0f} ({period}){marker}")

        return "\n".join(lines)
