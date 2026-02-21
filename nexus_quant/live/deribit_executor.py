"""
Deribit Options Execution Connector
=====================================

Handles authenticated interactions with Deribit for options trading:
- Account balance and position fetching
- Option order placement (sell straddles, buy/sell strangles)
- Position reconciliation with target weights from signal generator
- Instrument selection via InstrumentSelector (ATM straddles, 25d strangles)
- Automatic position rolling at <7 DTE
- Delta hedging via perpetual futures
- Portfolio Greeks aggregation and risk limits
- Emergency position closure

Security:
- API keys read from environment (DERIBIT_CLIENT_ID, DERIBIT_CLIENT_SECRET)
- OAuth2 authentication per Deribit spec
- Optional testnet mode (test.deribit.com)

Usage:
    from nexus_quant.live.deribit_executor import DeribitExecutor
    executor = DeribitExecutor(testnet=True)
    account = executor.get_account()
    executor.reconcile(target_weights={"BTC": -0.15, "ETH": -0.10})
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.deribit_executor")

PROJ_ROOT = Path(__file__).resolve().parents[2]
EXEC_LOG = PROJ_ROOT / "artifacts" / "crypto_options" / "execution_log.jsonl"

# Deribit endpoints
MAINNET_BASE = "https://www.deribit.com/api/v2"
TESTNET_BASE = "https://test.deribit.com/api/v2"

# Supported underlying assets
SUPPORTED_ASSETS = ["BTC", "ETH"]

# Default option expiry selection: nearest monthly ~30 DTE
DEFAULT_TARGET_DTE = 30


@dataclass
class DeribitAccountInfo:
    """Snapshot of Deribit account state."""
    equity: float          # total equity in BTC
    equity_usd: float      # estimated USD value
    margin_balance: float
    available_funds: float
    positions: Dict[str, "DeribitPositionInfo"]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity": round(self.equity, 8),
            "equity_usd": round(self.equity_usd, 2),
            "margin_balance": round(self.margin_balance, 8),
            "available_funds": round(self.available_funds, 8),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "timestamp": self.timestamp,
        }


@dataclass
class DeribitPositionInfo:
    """Single position on Deribit."""
    instrument_name: str
    kind: str             # "option" | "future"
    direction: str        # "buy" | "sell"
    size: float           # in contracts
    average_price: float
    mark_price: float
    delta: float
    vega: float
    theta: float
    unrealized_pnl: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument_name": self.instrument_name,
            "kind": self.kind,
            "direction": self.direction,
            "size": self.size,
            "average_price": round(self.average_price, 4),
            "mark_price": round(self.mark_price, 4),
            "delta": round(self.delta, 6),
            "vega": round(self.vega, 6),
            "theta": round(self.theta, 6),
            "unrealized_pnl": round(self.unrealized_pnl, 8),
        }


@dataclass
class DeribitOrderResult:
    """Result of an order placement on Deribit."""
    instrument_name: str
    order_id: str
    direction: str    # "buy" | "sell"
    amount: float
    price: float
    order_type: str   # "limit" | "market"
    status: str
    timestamp: str
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument_name": self.instrument_name,
            "order_id": self.order_id,
            "direction": self.direction,
            "amount": self.amount,
            "price": round(self.price, 4),
            "order_type": self.order_type,
            "status": self.status,
            "timestamp": self.timestamp,
        }


@dataclass
class PortfolioGreeks:
    """Aggregated portfolio-level Greeks."""
    net_delta: float = 0.0       # total delta exposure
    net_gamma: float = 0.0       # total gamma
    net_vega: float = 0.0        # total vega
    net_theta: float = 0.0       # total theta (daily)
    per_asset: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "net_delta": round(self.net_delta, 6),
            "net_gamma": round(self.net_gamma, 6),
            "net_vega": round(self.net_vega, 4),
            "net_theta": round(self.net_theta, 4),
            "per_asset": self.per_asset,
            "timestamp": self.timestamp,
        }


# Greeks risk limits
GREEKS_LIMITS = {
    "max_abs_delta": 0.50,    # max absolute portfolio delta per asset
    "max_abs_vega": 10000.0,  # max absolute vega in USD
    "max_abs_gamma": 1.0,     # max absolute gamma
}

# Delta hedge config
DELTA_HEDGE_CONFIG = {
    "tolerance": 0.10,        # hedge when |delta| > 10% of notional
    "instrument": "PERPETUAL", # hedge via perpetual futures
    "prefer_maker": True,      # use limit orders for better fees
}


class DeribitExecutor:
    """
    Deribit options executor with instrument selection and delta hedging.

    Trading approach:
    - VRP signal → sell ATM straddles (short vol) via InstrumentSelector
    - Skew MR signal → buy/sell 25d strangles (fade skew extremes)
    - Automatic position rolling at <7 DTE
    - Delta hedging via perpetual futures when |delta| > tolerance
    - Portfolio Greeks aggregation with risk limits
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        testnet: bool = True,
        dry_run: bool = True,
        max_notional_usd: float = 50000.0,
        delta_hedge: bool = True,
    ) -> None:
        self.client_id = client_id or os.environ.get("DERIBIT_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("DERIBIT_CLIENT_SECRET", "")
        self.testnet = testnet
        self.base_url = TESTNET_BASE if testnet else MAINNET_BASE
        self.dry_run = dry_run
        self.max_notional_usd = max_notional_usd
        self.delta_hedge = delta_hedge
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

        # Instrument selector for finding the right options to trade
        from .instrument_selector import InstrumentSelector
        self._selector = InstrumentSelector(testnet=testnet)

        # Track current positions by expiry for rolling
        self._current_expiry: Dict[str, str] = {}  # {asset: expiry_date}

    # ── Authentication ────────────────────────────────────────────────────────

    def _authenticate(self) -> None:
        """OAuth2 client credentials authentication."""
        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "Deribit API credentials not set. Set DERIBIT_CLIENT_ID and "
                "DERIBIT_CLIENT_SECRET environment variables."
            )

        params = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        result = self._public_request("public/auth", params)
        if result:
            self._access_token = result.get("access_token")
            expires_in = result.get("expires_in", 900)
            self._token_expiry = time.time() + expires_in - 60
            logger.info("Deribit authenticated (expires in %ds)", expires_in)

    def _ensure_auth(self) -> None:
        """Ensure we have a valid access token."""
        if self._access_token is None or time.time() >= self._token_expiry:
            self._authenticate()

    # ── API Methods ───────────────────────────────────────────────────────────

    def _public_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make a public API request."""
        url = f"{self.base_url}/{method}?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                return data.get("result")
        except Exception as e:
            logger.error("Deribit API error (%s): %s", method, e)
            return None

    def _private_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make an authenticated API request."""
        if self.dry_run:
            logger.info("[DRY RUN] Would call %s with %s", method, params)
            return None

        self._ensure_auth()
        url = f"{self.base_url}/{method}"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        body = json.dumps({"jsonrpc": "2.0", "method": method, "params": params}).encode()
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                return data.get("result")
        except Exception as e:
            logger.error("Deribit private API error (%s): %s", method, e)
            return None

    # ── Account & Positions ───────────────────────────────────────────────────

    def get_account(self, currency: str = "BTC") -> Optional[DeribitAccountInfo]:
        """Fetch account summary."""
        if self.dry_run:
            logger.info("[DRY RUN] get_account(%s)", currency)
            return DeribitAccountInfo(
                equity=0.0, equity_usd=0.0, margin_balance=0.0,
                available_funds=0.0, positions={},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        result = self._private_request("private/get_account_summary", {
            "currency": currency, "extended": True,
        })
        if not result:
            return None

        return DeribitAccountInfo(
            equity=result.get("equity", 0.0),
            equity_usd=result.get("equity", 0.0) * result.get("estimated_delivery_price", 0.0),
            margin_balance=result.get("margin_balance", 0.0),
            available_funds=result.get("available_funds", 0.0),
            positions={},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_positions(self, currency: str = "BTC") -> List[DeribitPositionInfo]:
        """Fetch all open positions for a currency."""
        if self.dry_run:
            logger.info("[DRY RUN] get_positions(%s)", currency)
            return []

        result = self._private_request("private/get_positions", {
            "currency": currency,
        })
        if not result:
            return []

        positions = []
        for p in result:
            positions.append(DeribitPositionInfo(
                instrument_name=p.get("instrument_name", ""),
                kind=p.get("kind", ""),
                direction=p.get("direction", ""),
                size=p.get("size", 0.0),
                average_price=p.get("average_price", 0.0),
                mark_price=p.get("mark_price", 0.0),
                delta=p.get("delta", 0.0),
                vega=p.get("vega", 0.0),
                theta=p.get("theta", 0.0),
                unrealized_pnl=p.get("floating_profit_loss", 0.0),
            ))
        return positions

    # ── Order Execution ───────────────────────────────────────────────────────

    def reconcile(
        self,
        target_weights: Dict[str, float],
        equity_usd: float = 100000.0,
        strategy: str = "vrp",
    ) -> List[DeribitOrderResult]:
        """
        Reconcile current positions with target weights using real instruments.

        For VRP (negative weight = short vol):
            - Selects ATM straddle via InstrumentSelector
            - Size = |weight| × equity / index_price

        For Skew MR:
            - Selects 25d strangle via InstrumentSelector
            - Positive weight = buy put + sell call (long skew)
            - Negative weight = sell put + buy call (short skew)

        Includes:
            - Automatic instrument selection (nearest monthly, ATM/25d)
            - Position rolling check (rolls at <7 DTE)
            - Delta hedging via perpetuals (if enabled)
        """
        orders: List[DeribitOrderResult] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for asset, weight in target_weights.items():
            if asset not in SUPPORTED_ASSETS:
                logger.warning("Unsupported asset %s, skipping", asset)
                continue

            if abs(weight) < 0.001:
                continue

            # Check if current positions need rolling
            if asset in self._current_expiry:
                instruments = self._selector.get_instruments(asset)
                if self._selector.needs_rolling(self._current_expiry[asset], instruments):
                    logger.warning("[%s] Position needs rolling — current expiry %s",
                                   asset, self._current_expiry[asset])
                    roll_orders = self._roll_position(asset, weight, equity_usd, strategy)
                    orders.extend(roll_orders)
                    continue

            # Select instruments based on strategy
            asset_orders = self._build_orders(
                asset, weight, equity_usd, strategy, now_iso,
            )
            orders.extend(asset_orders)

        # Delta hedge if enabled
        if self.delta_hedge:
            hedge_orders = self._check_delta_hedge(equity_usd)
            orders.extend(hedge_orders)

        self._log_orders(orders)
        return orders

    def _build_orders(
        self,
        asset: str,
        weight: float,
        equity_usd: float,
        strategy: str,
        now_iso: str,
    ) -> List[DeribitOrderResult]:
        """Build orders for a single asset using instrument selector."""
        orders: List[DeribitOrderResult] = []

        # Get spot price for sizing
        spot_data = self._selector.get_spot_and_iv(asset)
        spot_price = spot_data["spot"] if spot_data else None

        if spot_price is None or spot_price <= 0:
            logger.error("Cannot get spot price for %s, skipping", asset)
            return orders

        # Contract size: |weight| × equity / spot_price
        contracts = abs(weight) * equity_usd / spot_price
        direction = "sell" if weight < 0 else "buy"

        if strategy == "vrp":
            # VRP: ATM straddle
            straddle = self._selector.select_straddle(asset, spot_price)
            if straddle is None:
                logger.error("Could not select straddle for %s", asset)
                return orders

            self._current_expiry[asset] = straddle.expiry_date

            # Sell (or buy) both call and put at ATM
            for leg_inst, leg_type in [(straddle.call, "call"), (straddle.put, "put")]:
                order = DeribitOrderResult(
                    instrument_name=leg_inst.name,
                    order_id=f"DRY-{int(time.time())}-{leg_type}",
                    direction=direction,
                    amount=round(contracts, 1),
                    price=0.0,
                    order_type="market",
                    status="DRY_RUN" if self.dry_run else "PENDING",
                    timestamp=now_iso,
                )
                orders.append(order)
                logger.info("[%s] %s straddle %s: %s %.1f contracts (strike=%.0f DTE=%.0f)",
                            "DRY RUN" if self.dry_run else "ORDER",
                            direction.upper(), leg_type, asset,
                            contracts, straddle.strike, straddle.dte)

        elif strategy == "skew":
            # Skew MR: 25-delta strangle
            iv = spot_data.get("iv_atm", 0.60) if spot_data else 0.60
            strangle = self._selector.select_strangle(asset, spot_price, iv=iv)
            if strangle is None:
                logger.error("Could not select strangle for %s", asset)
                return orders

            self._current_expiry[asset] = strangle.expiry_date

            # Long skew (positive weight): buy put + sell call
            # Short skew (negative weight): sell put + buy call
            if weight > 0:
                put_dir, call_dir = "buy", "sell"
            else:
                put_dir, call_dir = "sell", "buy"

            for inst, d, leg_type in [
                (strangle.put, put_dir, "put"),
                (strangle.call, call_dir, "call"),
            ]:
                order = DeribitOrderResult(
                    instrument_name=inst.name,
                    order_id=f"DRY-{int(time.time())}-{leg_type}",
                    direction=d,
                    amount=round(contracts, 1),
                    price=0.0,
                    order_type="market",
                    status="DRY_RUN" if self.dry_run else "PENDING",
                    timestamp=now_iso,
                )
                orders.append(order)
                logger.info("[%s] strangle %s %s: %s %.1f contracts (strike=%.0f DTE=%.0f)",
                            "DRY RUN" if self.dry_run else "ORDER",
                            d.upper(), leg_type, asset,
                            contracts, inst.strike, strangle.dte)

        return orders

    # ── Position Rolling ───────────────────────────────────────────────────────

    def _roll_position(
        self,
        asset: str,
        weight: float,
        equity_usd: float,
        strategy: str,
    ) -> List[DeribitOrderResult]:
        """
        Roll expiring positions to next month.

        Sequence: close current position → open at new expiry.
        """
        orders: List[DeribitOrderResult] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        logger.info("[%s] Rolling %s position from %s",
                    asset, strategy, self._current_expiry.get(asset, "?"))

        # Close existing positions (reverse direction)
        close_weight = -weight
        close_orders = self._build_orders(
            asset, close_weight, equity_usd, strategy, now_iso,
        )
        for o in close_orders:
            o.status = "ROLL_CLOSE" if self.dry_run else "PENDING"
        orders.extend(close_orders)

        # Clear old expiry
        old_expiry = self._current_expiry.pop(asset, None)

        # Open new positions at next expiry
        new_orders = self._build_orders(
            asset, weight, equity_usd, strategy, now_iso,
        )
        for o in new_orders:
            o.status = "ROLL_OPEN" if self.dry_run else "PENDING"
        orders.extend(new_orders)

        new_expiry = self._current_expiry.get(asset, "?")
        logger.info("[%s] Rolled: %s → %s (%d orders)",
                    asset, old_expiry, new_expiry, len(orders))

        return orders

    # ── Delta Hedging ──────────────────────────────────────────────────────────

    def _check_delta_hedge(
        self, equity_usd: float,
    ) -> List[DeribitOrderResult]:
        """
        Check portfolio delta and hedge via perpetuals if needed.

        Uses R18 (bidirectional funding) and R19 (wide tolerance) from research:
        - Tolerance = 10% of notional per asset
        - Hedge via BTC-PERPETUAL / ETH-PERPETUAL
        - Prefer maker orders for -1bps rebate
        """
        orders: List[DeribitOrderResult] = []

        if self.dry_run:
            # In dry run, compute theoretical delta from positions
            # (no real positions available)
            return orders

        greeks = self.portfolio_greeks()
        if greeks is None:
            return orders

        tolerance = DELTA_HEDGE_CONFIG["tolerance"]
        now_iso = datetime.now(timezone.utc).isoformat()

        for asset in SUPPORTED_ASSETS:
            asset_greeks = greeks.per_asset.get(asset, {})
            delta = asset_greeks.get("delta", 0.0)

            if abs(delta) > tolerance:
                # Need to hedge: short delta → buy perp, long delta → sell perp
                hedge_direction = "buy" if delta < 0 else "sell"
                hedge_amount = abs(delta)

                instrument = f"{asset}-PERPETUAL"
                order = DeribitOrderResult(
                    instrument_name=instrument,
                    order_id=f"HEDGE-{int(time.time())}",
                    direction=hedge_direction,
                    amount=round(hedge_amount, 4),
                    price=0.0,
                    order_type="limit" if DELTA_HEDGE_CONFIG["prefer_maker"] else "market",
                    status="PENDING",
                    timestamp=now_iso,
                )
                orders.append(order)
                logger.info("[HEDGE] %s %s %.4f (delta was %.4f, tolerance %.2f)",
                            hedge_direction.upper(), instrument, hedge_amount,
                            delta, tolerance)

        return orders

    # ── Portfolio Greeks ───────────────────────────────────────────────────────

    def portfolio_greeks(self) -> Optional[PortfolioGreeks]:
        """
        Aggregate portfolio-level Greeks from all open positions.

        Returns PortfolioGreeks with per-asset breakdown.
        """
        greeks = PortfolioGreeks(
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        for currency in SUPPORTED_ASSETS:
            positions = self.get_positions(currency)
            asset_delta = 0.0
            asset_gamma = 0.0
            asset_vega = 0.0
            asset_theta = 0.0

            for pos in positions:
                sign = 1.0 if pos.direction == "buy" else -1.0
                asset_delta += sign * pos.delta * pos.size
                asset_vega += sign * pos.vega * pos.size
                asset_theta += sign * pos.theta * pos.size
                # Gamma not directly available from API, estimate from vega
                # gamma ≈ vega / (S * sigma * sqrt(T))
                # For now, use the delta as primary risk metric

            greeks.net_delta += asset_delta
            greeks.net_vega += asset_vega
            greeks.net_theta += asset_theta

            greeks.per_asset[currency] = {
                "delta": round(asset_delta, 6),
                "vega": round(asset_vega, 4),
                "theta": round(asset_theta, 4),
                "n_positions": len(positions),
            }

        return greeks

    def check_greeks_limits(self) -> Dict[str, Any]:
        """Check if portfolio Greeks exceed risk limits."""
        result = {"halt": False, "warnings": [], "breaches": []}

        greeks = self.portfolio_greeks()
        if greeks is None:
            result["warnings"].append("Could not compute portfolio Greeks")
            return result

        for asset, asset_greeks in greeks.per_asset.items():
            delta = abs(asset_greeks.get("delta", 0.0))
            vega = abs(asset_greeks.get("vega", 0.0))

            if delta > GREEKS_LIMITS["max_abs_delta"]:
                result["breaches"].append(
                    f"{asset} delta={delta:.4f} > limit={GREEKS_LIMITS['max_abs_delta']:.2f}"
                )
            if vega > GREEKS_LIMITS["max_abs_vega"]:
                result["breaches"].append(
                    f"{asset} vega={vega:.2f} > limit={GREEKS_LIMITS['max_abs_vega']:.0f}"
                )

        if result["breaches"]:
            result["halt"] = True
            logger.warning("Greeks limits breached: %s", result["breaches"])

        return result

    def close_all(self) -> List[DeribitOrderResult]:
        """Emergency: close all positions."""
        logger.warning("EMERGENCY CLOSE ALL called")
        if self.dry_run:
            logger.info("[DRY RUN] Would close all Deribit positions")
            return []

        orders = []
        for currency in SUPPORTED_ASSETS:
            positions = self.get_positions(currency)
            for pos in positions:
                if abs(pos.size) > 0:
                    close_dir = "sell" if pos.direction == "buy" else "buy"
                    order = DeribitOrderResult(
                        instrument_name=pos.instrument_name,
                        order_id="",
                        direction=close_dir,
                        amount=abs(pos.size),
                        price=0.0,
                        order_type="market",
                        status="PENDING",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    orders.append(order)

        self._log_orders(orders)
        return orders

    # ── Risk Gates ────────────────────────────────────────────────────────────

    def check_risk_gates(
        self,
        account: Optional[DeribitAccountInfo] = None,
        max_leverage: float = 3.0,
    ) -> Dict[str, Any]:
        """Check risk limits before trading (margin + Greeks)."""
        result = {"halt": False, "warnings": [], "reasons": []}

        if account is None:
            account = self.get_account()
        if account is None:
            result["warnings"].append("Could not fetch account info")
            return result

        # Check available margin
        if account.available_funds <= 0:
            result["halt"] = True
            result["reasons"].append("No available margin")

        # Check Greeks limits
        greeks_check = self.check_greeks_limits()
        if greeks_check["halt"]:
            result["halt"] = True
            result["reasons"].extend(greeks_check["breaches"])
        result["warnings"].extend(greeks_check["warnings"])

        return result

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_orders(self, orders: List[DeribitOrderResult]) -> None:
        """Log orders to JSONL."""
        EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(EXEC_LOG, "a") as f:
                for o in orders:
                    f.write(json.dumps(o.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.error("Failed to log orders: %s", e)
