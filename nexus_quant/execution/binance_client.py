"""
Binance USDM Futures authenticated REST client.
Pure stdlib (urllib + hmac) — no third-party deps.

Env vars required:
  BINANCE_API_KEY    — API key
  BINANCE_API_SECRET — API secret

Usage:
  client = BinanceFuturesClient()
  positions = client.get_positions()
  client.place_order("BTCUSDT", "BUY", quantity=0.001)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
_BASE_URL = "https://fapi.binance.com"

_EP_ACCOUNT = "/fapi/v2/account"
_EP_POSITION_RISK = "/fapi/v2/positionRisk"
_EP_ORDER = "/fapi/v1/order"
_EP_OPEN_ORDERS = "/fapi/v1/openOrders"
_EP_ALL_ORDERS = "/fapi/v1/allOrders"
_EP_CANCEL_ORDER = "/fapi/v1/order"
_EP_CANCEL_ALL = "/fapi/v1/allOpenOrders"
_EP_LEVERAGE = "/fapi/v1/leverage"
_EP_MARGIN_TYPE = "/fapi/v1/marginType"
_EP_EXCHANGE_INFO = "/fapi/v1/exchangeInfo"
_EP_TICKER_PRICE = "/fapi/v1/ticker/price"
_EP_KLINES = "/fapi/v1/klines"

# Rate limit: min sleep between signed requests
_RATE_SLEEP = 0.15
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    side: str           # "LONG" | "SHORT" | "BOTH"
    quantity: float     # signed (positive = long, negative = short)
    entry_price: float
    unrealized_pnl: float
    leverage: int
    notional: float     # absolute USD value


@dataclass
class OrderResult:
    order_id: int
    symbol: str
    side: str
    quantity: float
    price: float
    status: str
    type: str
    timestamp: int


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class BinanceFuturesClient:
    """Authenticated Binance USDM Futures client (pure stdlib)."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = _BASE_URL,
        testnet: bool = False,
    ):
        self._api_key = api_key or os.environ.get("BINANCE_API_KEY", "")
        self._api_secret = api_secret or os.environ.get("BINANCE_API_SECRET", "")
        if testnet:
            self._base_url = "https://testnet.binancefuture.com"
        else:
            self._base_url = base_url.rstrip("/")
        self._last_req_time = 0.0

    # ------------------------------------------------------------------
    # HTTP layer
    # ------------------------------------------------------------------

    def _sign(self, params: Dict[str, Any]) -> str:
        """Generate HMAC-SHA256 signature for query string."""
        query = urllib.parse.urlencode(params)
        sig = hmac.new(
            self._api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return sig

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = True,
    ) -> Any:
        """Make authenticated HTTP request with retry + rate limiting."""
        if not self._api_key:
            raise RuntimeError("BINANCE_API_KEY not set")

        params = dict(params or {})

        if signed:
            if not self._api_secret:
                raise RuntimeError("BINANCE_API_SECRET not set")
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            params["signature"] = self._sign(params)

        # Rate limit
        elapsed = time.time() - self._last_req_time
        if elapsed < _RATE_SLEEP:
            time.sleep(_RATE_SLEEP - elapsed)

        query = urllib.parse.urlencode(params)
        url = f"{self._base_url}{endpoint}"

        if method == "GET":
            url = f"{url}?{query}" if query else url
            data = None
        else:
            data = query.encode("utf-8")

        headers = {
            "X-MBX-APIKEY": self._api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        last_error = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(
                    url=url,
                    data=data,
                    headers=headers,
                    method=method,
                )
                ctx = _SSL_CTX or ssl.create_default_context()
                with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                    self._last_req_time = time.time()
                    body = resp.read().decode("utf-8")
                    return json.loads(body)
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                # Don't retry 4xx errors (except 429)
                if 400 <= e.code < 500 and e.code != 429:
                    raise RuntimeError(f"Binance HTTP {e.code}: {body[:500]}") from e
                last_error = e
                time.sleep(_RETRY_BACKOFF ** attempt)
            except Exception as e:
                last_error = e
                time.sleep(_RETRY_BACKOFF ** attempt)

        raise RuntimeError(f"Binance request failed after {_MAX_RETRIES} retries: {last_error}")

    def _get(self, endpoint: str, params: dict = None, signed: bool = True) -> Any:
        return self._request("GET", endpoint, params, signed)

    def _post(self, endpoint: str, params: dict = None) -> Any:
        return self._request("POST", endpoint, params, signed=True)

    def _delete(self, endpoint: str, params: dict = None) -> Any:
        return self._request("DELETE", endpoint, params, signed=True)

    # ------------------------------------------------------------------
    # Public endpoints (no auth)
    # ------------------------------------------------------------------

    def get_exchange_info(self) -> Dict:
        """Get exchange trading rules and symbol information."""
        return self._get(_EP_EXCHANGE_INFO, signed=False)

    def get_ticker_price(self, symbol: str = "") -> Any:
        """Get latest price for symbol(s)."""
        params = {"symbol": symbol} if symbol else {}
        return self._get(_EP_TICKER_PRICE, params, signed=False)

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 500) -> List:
        """Get recent klines."""
        return self._get(_EP_KLINES, {
            "symbol": symbol, "interval": interval, "limit": limit
        }, signed=False)

    # ------------------------------------------------------------------
    # Account & position endpoints
    # ------------------------------------------------------------------

    def get_account(self) -> Dict:
        """Get full account information (balances, positions)."""
        return self._get(_EP_ACCOUNT)

    def get_balance(self) -> float:
        """Get total USDT wallet balance."""
        account = self.get_account()
        for asset in account.get("assets", []):
            if asset["asset"] == "USDT":
                return float(asset["walletBalance"])
        return 0.0

    def get_positions(self, symbols: List[str] = None) -> List[Position]:
        """Get current open positions."""
        raw = self._get(_EP_POSITION_RISK)
        positions = []
        for p in raw:
            qty = float(p.get("positionAmt", 0))
            if qty == 0:
                continue
            sym = p["symbol"]
            if symbols and sym not in symbols:
                continue
            positions.append(Position(
                symbol=sym,
                side="LONG" if qty > 0 else "SHORT",
                quantity=qty,
                entry_price=float(p.get("entryPrice", 0)),
                unrealized_pnl=float(p.get("unRealizedProfit", 0)),
                leverage=int(p.get("leverage", 1)),
                notional=abs(float(p.get("notional", 0))),
            ))
        return positions

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,           # "BUY" | "SELL"
        quantity: float,
        order_type: str = "MARKET",
        price: float = 0.0,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> OrderResult:
        """Place a new order. Returns OrderResult."""
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": f"{quantity:.8f}".rstrip("0").rstrip("."),
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if order_type == "LIMIT":
            params["price"] = f"{price:.8f}".rstrip("0").rstrip(".")
            params["timeInForce"] = time_in_force

        resp = self._post(_EP_ORDER, params)
        return OrderResult(
            order_id=int(resp.get("orderId", 0)),
            symbol=resp.get("symbol", symbol),
            side=resp.get("side", side),
            quantity=float(resp.get("origQty", quantity)),
            price=float(resp.get("avgPrice", 0) or resp.get("price", 0)),
            status=resp.get("status", "UNKNOWN"),
            type=resp.get("type", order_type),
            timestamp=int(resp.get("updateTime", 0)),
        )

    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel a specific order."""
        return self._delete(_EP_CANCEL_ORDER, {
            "symbol": symbol, "orderId": order_id
        })

    def cancel_all_orders(self, symbol: str) -> Dict:
        """Cancel all open orders for a symbol."""
        return self._delete(_EP_CANCEL_ALL, {"symbol": symbol})

    def get_open_orders(self, symbol: str = "") -> List[Dict]:
        """Get all open orders."""
        params = {"symbol": symbol} if symbol else {}
        return self._get(_EP_OPEN_ORDERS, params)

    # ------------------------------------------------------------------
    # Account config
    # ------------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """Set leverage for a symbol."""
        return self._post(_EP_LEVERAGE, {
            "symbol": symbol, "leverage": leverage
        })

    def set_margin_type(self, symbol: str, margin_type: str = "CROSSED") -> Dict:
        """Set margin type (CROSSED or ISOLATED)."""
        try:
            return self._post(_EP_MARGIN_TYPE, {
                "symbol": symbol, "marginType": margin_type
            })
        except RuntimeError as e:
            # "No need to change margin type" is not an error
            if "No need to change" in str(e):
                return {"msg": "Already set"}
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get trading rules for a specific symbol (precision, filters)."""
        info = self.get_exchange_info()
        for s in info.get("symbols", []):
            if s["symbol"] == symbol:
                return s
        return None

    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to symbol's step size."""
        info = self.get_symbol_info(symbol)
        if not info:
            return round(quantity, 3)
        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                step = float(f["stepSize"])
                if step > 0:
                    precision = len(f["stepSize"].rstrip("0").split(".")[-1])
                    return round(quantity - (quantity % step), precision)
        return round(quantity, 3)

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to symbol's tick size."""
        info = self.get_symbol_info(symbol)
        if not info:
            return round(price, 2)
        for f in info.get("filters", []):
            if f["filterType"] == "PRICE_FILTER":
                tick = float(f["tickSize"])
                if tick > 0:
                    precision = len(f["tickSize"].rstrip("0").split(".")[-1])
                    return round(price - (price % tick), precision)
        return round(price, 2)

    def ping(self) -> bool:
        """Test connectivity and API key validity."""
        try:
            self.get_account()
            return True
        except Exception:
            return False
