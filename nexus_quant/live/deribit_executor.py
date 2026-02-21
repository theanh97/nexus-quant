"""
Deribit Options Execution Connector (Stub).

Handles authenticated interactions with Deribit for options trading:
- Account balance and position fetching
- Option order placement (sell straddles, buy/sell strangles)
- Position reconciliation with target weights from signal generator
- Emergency position closure

This is a STUB implementation for the execution interface. It will be
fully implemented when real Deribit data validation is complete (~April 2026).
Currently supports dry_run mode only.

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


class DeribitExecutor:
    """
    Deribit options executor.

    Trading approach (from VRP strategy):
    - VRP signal → sell ATM straddles (short vol)
    - Skew MR signal → buy/sell 25d strangles (fade skew extremes)

    Currently STUB — only dry_run mode is functional.
    Full implementation requires:
    1. OAuth2 authentication flow
    2. Option instrument selection (nearest monthly, ATM strike)
    3. Delta hedging with perpetual futures
    4. Position rolling before expiry
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        testnet: bool = True,
        dry_run: bool = True,
        max_notional_usd: float = 50000.0,
    ) -> None:
        self.client_id = client_id or os.environ.get("DERIBIT_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("DERIBIT_CLIENT_SECRET", "")
        self.testnet = testnet
        self.base_url = TESTNET_BASE if testnet else MAINNET_BASE
        self.dry_run = dry_run
        self.max_notional_usd = max_notional_usd
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

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
    ) -> List[DeribitOrderResult]:
        """
        Reconcile current positions with target weights.

        For VRP (negative weight = short vol):
            - Target is ATM straddle sell (call + put at nearest ATM strike)
            - Size = |weight| × equity / index_price

        For Skew MR:
            - Positive weight = buy 25d put / sell 25d call (long skew)
            - Negative weight = sell 25d put / buy 25d call (short skew)

        Currently STUB — logs intended trades without executing.
        """
        orders: List[DeribitOrderResult] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for asset, weight in target_weights.items():
            if asset not in SUPPORTED_ASSETS:
                logger.warning("Unsupported asset %s, skipping", asset)
                continue

            if abs(weight) < 0.001:
                continue

            notional = abs(weight) * equity_usd
            direction = "sell" if weight < 0 else "buy"

            # For now, log the intended trade
            order = DeribitOrderResult(
                instrument_name=f"{asset}-OPTION-ATM",
                order_id=f"DRY-{int(time.time())}",
                direction=direction,
                amount=notional,
                price=0.0,  # market order
                order_type="market",
                status="DRY_RUN" if self.dry_run else "PENDING",
                timestamp=now_iso,
            )
            orders.append(order)

            logger.info("[%s] %s %s: notional=$%.0f (weight=%.4f)",
                        "DRY RUN" if self.dry_run else "ORDER",
                        direction.upper(), asset, notional, weight)

        self._log_orders(orders)
        return orders

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
        """Check risk limits before trading."""
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
