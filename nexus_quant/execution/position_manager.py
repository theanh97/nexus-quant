"""
Position Manager — translates strategy target weights into executable orders.

Workflow each rebalance:
  1. Get current positions from Binance
  2. Get target weights from strategy ensemble
  3. Compute diffs (delta weights)
  4. Apply risk limits (max position, max turnover, max leverage)
  5. Generate orders (market orders for simplicity)
  6. Execute with retry
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .binance_client import BinanceFuturesClient, Position, OrderResult

log = logging.getLogger("nexus.execution")


@dataclass
class RebalanceOrder:
    """A single order to execute during rebalance."""
    symbol: str
    side: str           # "BUY" | "SELL"
    quantity: float     # absolute quantity (in base asset)
    notional_usd: float
    reason: str         # "increase_long" | "decrease_long" | "open_short" | etc.


@dataclass
class RebalanceResult:
    """Result of a full rebalance cycle."""
    timestamp: str
    balance_usd: float
    target_weights: Dict[str, float]
    current_weights: Dict[str, float]
    orders_planned: int
    orders_executed: int
    orders_failed: int
    total_turnover_usd: float
    fills: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


class PositionManager:
    """
    Manages the lifecycle of rebalancing:
    target weights → delta computation → order generation → execution.
    """

    def __init__(
        self,
        client: BinanceFuturesClient,
        symbols: List[str],
        max_gross_leverage: float = 0.70,
        max_position_pct: float = 0.30,
        max_turnover_pct: float = 0.80,
        min_order_usd: float = 10.0,
        log_dir: Optional[Path] = None,
    ):
        self.client = client
        self.symbols = list(symbols)
        self.max_gross_leverage = max_gross_leverage
        self.max_position_pct = max_position_pct
        self.max_turnover_pct = max_turnover_pct
        self.min_order_usd = min_order_usd
        self._log_dir = log_dir or Path("artifacts/execution")
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Current state
    # ------------------------------------------------------------------

    def get_current_weights(self) -> Dict[str, float]:
        """Get current portfolio weights (position notional / balance)."""
        balance = self.client.get_balance()
        if balance <= 0:
            return {s: 0.0 for s in self.symbols}

        positions = self.client.get_positions(self.symbols)
        pos_map = {p.symbol: p for p in positions}

        weights = {}
        for sym in self.symbols:
            p = pos_map.get(sym)
            if p:
                # Signed weight: positive = long, negative = short
                weights[sym] = (p.quantity * p.entry_price) / balance
            else:
                weights[sym] = 0.0
        return weights

    def get_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        raw = self.client.get_ticker_price()
        prices = {}
        for item in raw:
            if item["symbol"] in self.symbols:
                prices[item["symbol"]] = float(item["price"])
        return prices

    # ------------------------------------------------------------------
    # Order generation
    # ------------------------------------------------------------------

    def compute_orders(
        self,
        target_weights: Dict[str, float],
        balance: float,
        current_weights: Dict[str, float],
        prices: Dict[str, float],
    ) -> List[RebalanceOrder]:
        """
        Compute orders needed to move from current to target weights.

        Risk enforcement:
        1. Clip individual positions to max_position_pct
        2. Scale down if gross leverage exceeds max
        3. Skip tiny orders below min_order_usd
        4. Limit total turnover to max_turnover_pct
        """
        # Step 1: Clip individual positions
        clipped = {}
        for sym in self.symbols:
            tw = target_weights.get(sym, 0.0)
            clipped[sym] = max(-self.max_position_pct, min(self.max_position_pct, tw))

        # Step 2: Scale down if gross leverage exceeded
        gross = sum(abs(w) for w in clipped.values())
        if gross > self.max_gross_leverage:
            scale = self.max_gross_leverage / gross
            clipped = {s: w * scale for s, w in clipped.items()}

        # Step 3: Compute deltas
        deltas = {}
        for sym in self.symbols:
            deltas[sym] = clipped[sym] - current_weights.get(sym, 0.0)

        # Step 4: Turnover limit
        total_turnover = sum(abs(d) for d in deltas.values())
        if total_turnover > self.max_turnover_pct:
            scale = self.max_turnover_pct / total_turnover
            deltas = {s: d * scale for s, d in deltas.items()}

        # Step 5: Generate orders
        orders = []
        for sym in self.symbols:
            delta_w = deltas[sym]
            delta_usd = delta_w * balance
            price = prices.get(sym, 0)
            if price <= 0 or abs(delta_usd) < self.min_order_usd:
                continue

            qty = abs(delta_usd) / price
            qty = self.client.round_quantity(sym, qty)
            if qty <= 0:
                continue

            side = "BUY" if delta_w > 0 else "SELL"
            reason = "increase" if delta_w > 0 else "decrease"

            orders.append(RebalanceOrder(
                symbol=sym,
                side=side,
                quantity=qty,
                notional_usd=abs(delta_usd),
                reason=reason,
            ))

        # Sort by notional (largest first for better execution)
        orders.sort(key=lambda o: o.notional_usd, reverse=True)
        return orders

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_rebalance(
        self,
        target_weights: Dict[str, float],
        dry_run: bool = False,
    ) -> RebalanceResult:
        """
        Full rebalance cycle:
        1. Fetch balance + prices + positions
        2. Compute orders
        3. Execute (or simulate in dry_run mode)
        4. Log results
        """
        ts = datetime.now(timezone.utc).isoformat()
        errors: List[str] = []
        fills: List[Dict[str, Any]] = []

        # Fetch state
        try:
            balance = self.client.get_balance()
            prices = self.get_prices()
            current_weights = self.get_current_weights()
        except Exception as e:
            return RebalanceResult(
                timestamp=ts,
                balance_usd=0,
                target_weights=target_weights,
                current_weights={},
                orders_planned=0,
                orders_executed=0,
                orders_failed=0,
                total_turnover_usd=0,
                skipped=True,
                skip_reason=f"Failed to fetch state: {e}",
            )

        if balance < 10:
            return RebalanceResult(
                timestamp=ts,
                balance_usd=balance,
                target_weights=target_weights,
                current_weights=current_weights,
                orders_planned=0,
                orders_executed=0,
                orders_failed=0,
                total_turnover_usd=0,
                skipped=True,
                skip_reason=f"Balance too low: ${balance:.2f}",
            )

        # Compute orders
        orders = self.compute_orders(target_weights, balance, current_weights, prices)

        if not orders:
            return RebalanceResult(
                timestamp=ts,
                balance_usd=balance,
                target_weights=target_weights,
                current_weights=current_weights,
                orders_planned=0,
                orders_executed=0,
                orders_failed=0,
                total_turnover_usd=0,
            )

        total_turnover = sum(o.notional_usd for o in orders)
        executed = 0
        failed = 0

        for order in orders:
            if dry_run:
                fills.append({
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "notional_usd": round(order.notional_usd, 2),
                    "status": "DRY_RUN",
                })
                executed += 1
                continue

            try:
                result = self.client.place_order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    order_type="MARKET",
                )
                fills.append({
                    "symbol": result.symbol,
                    "side": result.side,
                    "quantity": result.quantity,
                    "price": result.price,
                    "status": result.status,
                    "order_id": result.order_id,
                })
                executed += 1
                log.info(
                    "Order filled: %s %s %.6f @ %.2f (notional $%.2f)",
                    result.side, result.symbol, result.quantity,
                    result.price, order.notional_usd,
                )
            except Exception as e:
                failed += 1
                err_msg = f"{order.symbol} {order.side} {order.quantity}: {e}"
                errors.append(err_msg)
                log.error("Order failed: %s", err_msg)

        result = RebalanceResult(
            timestamp=ts,
            balance_usd=balance,
            target_weights=target_weights,
            current_weights=current_weights,
            orders_planned=len(orders),
            orders_executed=executed,
            orders_failed=failed,
            total_turnover_usd=total_turnover,
            fills=fills,
            errors=errors,
        )

        self._log_rebalance(result)
        return result

    def _log_rebalance(self, result: RebalanceResult) -> None:
        """Append rebalance result to JSONL log."""
        try:
            log_file = self._log_dir / "rebalance_log.jsonl"
            record = {
                "timestamp": result.timestamp,
                "balance_usd": round(result.balance_usd, 2),
                "orders_planned": result.orders_planned,
                "orders_executed": result.orders_executed,
                "orders_failed": result.orders_failed,
                "turnover_usd": round(result.total_turnover_usd, 2),
                "fills": result.fills,
                "errors": result.errors,
                "skipped": result.skipped,
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass
