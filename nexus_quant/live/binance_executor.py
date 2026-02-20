"""
Binance Futures Execution Connector.

Handles authenticated interactions with Binance USDM Futures:
- Account balance and position fetching
- Market/limit order placement
- Position reconciliation with target weights
- Emergency position closure

Security:
- API keys read from environment (BINANCE_API_KEY, BINANCE_API_SECRET)
- HMAC-SHA256 signature per Binance spec
- Optional testnet mode for paper trading on Binance testnet

Usage:
    from nexus_quant.live.binance_executor import BinanceExecutor
    executor = BinanceExecutor()  # reads keys from env
    account = executor.get_account()
    executor.reconcile(target_weights, account_equity)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJ_ROOT = Path(__file__).resolve().parents[2]
EXEC_LOG = PROJ_ROOT / "artifacts" / "live" / "execution_log.jsonl"

# Binance endpoints
MAINNET_BASE = "https://fapi.binance.com"
TESTNET_BASE = "https://testnet.binancefuture.com"


@dataclass
class AccountInfo:
    """Snapshot of Binance futures account."""
    total_balance: float
    available_balance: float
    unrealized_pnl: float
    positions: Dict[str, "PositionInfo"]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_balance": round(self.total_balance, 4),
            "available_balance": round(self.available_balance, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "timestamp": self.timestamp,
        }


@dataclass
class PositionInfo:
    """Single position on Binance futures."""
    symbol: str
    side: str  # "LONG" | "SHORT" | "FLAT"
    quantity: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    notional: float
    leverage: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": round(self.quantity, 6),
            "entry_price": round(self.entry_price, 4),
            "mark_price": round(self.mark_price, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "notional": round(self.notional, 4),
            "leverage": self.leverage,
        }


@dataclass
class OrderResult:
    """Result of an order placement."""
    symbol: str
    order_id: int
    side: str
    quantity: float
    price: float
    status: str
    type: str
    timestamp: str
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "order_id": self.order_id,
            "side": self.side,
            "quantity": round(self.quantity, 6),
            "price": round(self.price, 4),
            "status": self.status,
            "type": self.type,
            "timestamp": self.timestamp,
        }


class BinanceExecutor:
    """
    Authenticated Binance Futures executor.

    Supports mainnet and testnet modes.
    All orders are logged to artifacts/live/execution_log.jsonl.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        symbols: Optional[List[str]] = None,
        max_order_notional: float = 50000.0,
        dry_run: bool = False,
    ) -> None:
        self.api_key = api_key or os.environ.get("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("BINANCE_API_SECRET", "")
        self.testnet = testnet
        self.base_url = TESTNET_BASE if testnet else MAINNET_BASE
        self.symbols = symbols or [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
        ]
        self.max_order_notional = max_order_notional
        self.dry_run = dry_run

        # Symbol precision cache (fetched on first use)
        self._precision: Dict[str, Dict[str, int]] = {}

    def _check_keys(self) -> None:
        if not self.api_key or not self.api_secret:
            raise RuntimeError(
                "Binance API keys not set. Set BINANCE_API_KEY and BINANCE_API_SECRET "
                "environment variables, or pass them to BinanceExecutor()."
            )

    def _sign(self, params: Dict[str, str]) -> str:
        """Create HMAC-SHA256 signature."""
        query_string = urllib.parse.urlencode(params)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, str]] = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        """Make authenticated request to Binance API."""
        if signed:
            self._check_keys()

        params = dict(params or {})
        if signed:
            params["timestamp"] = str(int(time.time() * 1000))
            params["recvWindow"] = "5000"
            params["signature"] = self._sign(params)

        url = f"{self.base_url}{path}"
        if method == "GET" and params:
            url += "?" + urllib.parse.urlencode(params)

        headers = {"X-MBX-APIKEY": self.api_key} if signed or self.api_key else {}

        if method == "POST":
            data = urllib.parse.urlencode(params).encode("utf-8") if params else None
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            data = None

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body)
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(f"Binance HTTP {e.code}: {error_body[:500]}") from e

    def _load_precision(self) -> None:
        """Load symbol precision (quantity and price decimals)."""
        if self._precision:
            return
        data = self._request("GET", "/fapi/v1/exchangeInfo")
        for sym_info in data.get("symbols", []):
            symbol = sym_info["symbol"]
            qty_precision = int(sym_info.get("quantityPrecision", 3))
            price_precision = int(sym_info.get("pricePrecision", 2))
            self._precision[symbol] = {
                "qty": qty_precision,
                "price": price_precision,
            }

    def _round_qty(self, symbol: str, qty: float) -> float:
        self._load_precision()
        p = self._precision.get(symbol, {}).get("qty", 3)
        return round(qty, p)

    def _round_price(self, symbol: str, price: float) -> float:
        self._load_precision()
        p = self._precision.get(symbol, {}).get("price", 2)
        return round(price, p)

    def get_account(self) -> AccountInfo:
        """Fetch current account info and positions."""
        data = self._request("GET", "/fapi/v2/account", signed=True)

        positions: Dict[str, PositionInfo] = {}
        for pos in data.get("positions", []):
            sym = pos["symbol"]
            if sym not in self.symbols:
                continue
            qty = float(pos.get("positionAmt", 0))
            if abs(qty) < 1e-10:
                continue
            positions[sym] = PositionInfo(
                symbol=sym,
                side="LONG" if qty > 0 else "SHORT",
                quantity=abs(qty),
                entry_price=float(pos.get("entryPrice", 0)),
                mark_price=float(pos.get("markPrice", 0)),
                unrealized_pnl=float(pos.get("unrealizedProfit", 0)),
                notional=float(pos.get("notional", 0)),
                leverage=int(pos.get("leverage", 1)),
            )

        return AccountInfo(
            total_balance=float(data.get("totalWalletBalance", 0)),
            available_balance=float(data.get("availableBalance", 0)),
            unrealized_pnl=float(data.get("totalUnrealizedProfit", 0)),
            positions=positions,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_prices(self) -> Dict[str, float]:
        """Fetch current mark prices for all symbols."""
        data = self._request("GET", "/fapi/v1/premiumIndex")
        prices = {}
        for item in data:
            sym = item.get("symbol", "")
            if sym in self.symbols:
                prices[sym] = float(item.get("markPrice", 0))
        return prices

    def place_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> OrderResult:
        """Place a market order. side='BUY' or 'SELL'."""
        quantity = self._round_qty(symbol, abs(quantity))
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": str(quantity),
        }

        if self.dry_run:
            self._log_event({
                "type": "DRY_RUN_ORDER",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return OrderResult(
                symbol=symbol, order_id=0, side=side, quantity=quantity,
                price=0.0, status="DRY_RUN", type="MARKET",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        data = self._request("POST", "/fapi/v1/order", params=params, signed=True)

        result = OrderResult(
            symbol=symbol,
            order_id=int(data.get("orderId", 0)),
            side=data.get("side", side),
            quantity=float(data.get("executedQty", quantity)),
            price=float(data.get("avgPrice", 0)),
            status=data.get("status", "UNKNOWN"),
            type=data.get("type", "MARKET"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            raw=data,
        )

        self._log_event({
            "type": "ORDER_FILLED",
            **result.to_dict(),
        })

        return result

    def reconcile(
        self,
        target_weights: Dict[str, float],
        equity: Optional[float] = None,
        min_trade_notional: float = 20.0,
    ) -> List[OrderResult]:
        """
        Reconcile current positions with target weights.

        Computes the delta between current and target positions,
        then places market orders to reach the target.

        Args:
            target_weights: {symbol: weight} from signal generator
            equity: Account equity to use for sizing. If None, fetched from Binance.
            min_trade_notional: Skip trades smaller than this (in USD).

        Returns:
            List of OrderResults for orders placed.
        """
        account = self.get_account()
        if equity is None:
            equity = account.total_balance

        prices = self.get_prices()
        orders: List[OrderResult] = []

        for symbol in self.symbols:
            target_w = target_weights.get(symbol, 0.0)
            target_notional = equity * target_w
            target_qty = target_notional / prices.get(symbol, 1.0) if prices.get(symbol, 0) > 0 else 0.0

            # Current position
            pos = account.positions.get(symbol)
            current_qty = 0.0
            if pos:
                current_qty = pos.quantity if pos.side == "LONG" else -pos.quantity

            delta_qty = target_qty - current_qty

            # Skip small trades
            delta_notional = abs(delta_qty) * prices.get(symbol, 0)
            if delta_notional < min_trade_notional:
                continue

            # Safety: cap order size
            if delta_notional > self.max_order_notional:
                delta_qty = delta_qty * (self.max_order_notional / delta_notional)

            side = "BUY" if delta_qty > 0 else "SELL"
            qty = abs(delta_qty)

            try:
                result = self.place_market_order(symbol, side, qty)
                orders.append(result)
            except Exception as e:
                self._log_event({
                    "type": "ORDER_ERROR",
                    "symbol": symbol,
                    "side": side,
                    "quantity": qty,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        return orders

    def close_all(self) -> List[OrderResult]:
        """Emergency: close all open positions."""
        account = self.get_account()
        orders = []

        for symbol, pos in account.positions.items():
            if abs(pos.quantity) < 1e-10:
                continue
            # Close = reverse the position
            side = "SELL" if pos.side == "LONG" else "BUY"
            try:
                result = self.place_market_order(symbol, side, pos.quantity)
                orders.append(result)
            except Exception as e:
                self._log_event({
                    "type": "CLOSE_ERROR",
                    "symbol": symbol,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        return orders

    def check_risk_gates(
        self,
        account: AccountInfo,
        halt_conditions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Check risk halt conditions against current account state.

        Returns:
            {"halt": True/False, "reasons": [...], "warnings": [...]}
        """
        if halt_conditions is None:
            halt_conditions = {
                "max_drawdown_pct": 15.0,
                "max_daily_loss_pct": 5.0,
            }

        reasons = []
        warnings = []

        # Check unrealized P&L vs equity
        if account.total_balance > 0:
            pnl_pct = (account.unrealized_pnl / account.total_balance) * 100
            max_dd = halt_conditions.get("max_drawdown_pct", 15.0)
            if pnl_pct < -max_dd:
                reasons.append(f"Unrealized loss {pnl_pct:.1f}% exceeds halt threshold {max_dd}%")
            elif pnl_pct < -max_dd * 0.5:
                warnings.append(f"Unrealized loss {pnl_pct:.1f}% approaching halt threshold {max_dd}%")

        return {
            "halt": len(reasons) > 0,
            "reasons": reasons,
            "warnings": warnings,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _log_event(self, event: Dict[str, Any]) -> None:
        """Append event to execution log."""
        EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(EXEC_LOG, "a") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def status_summary(self) -> Dict[str, Any]:
        """Get a summary of the executor state."""
        return {
            "testnet": self.testnet,
            "dry_run": self.dry_run,
            "base_url": self.base_url,
            "api_key_set": bool(self.api_key),
            "api_secret_set": bool(self.api_secret),
            "symbols": self.symbols,
            "max_order_notional": self.max_order_notional,
        }
