"""
Real-Time Signal Generator for P91b Champion Ensemble + Vol Tilt.

Fetches the latest N hours of data from Binance REST API,
runs the champion strategy at the most recent bar, and outputs
target portfolio weights. This is the foundation for paper/live trading.

Usage:
    from nexus_quant.live.signal_generator import SignalGenerator
    gen = SignalGenerator.from_production_config()
    signal = gen.generate()
    print(signal.target_weights)    # {"BTCUSDT": 0.05, "ETHUSDT": -0.03, ...}
    print(signal.vol_tilt_active)   # True/False
    print(signal.trades_needed)     # [{"symbol": "BTCUSDT", "delta": 0.02}, ...]
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..data.providers.registry import make_provider
from ..data.schema import MarketDataset
from ..strategies.registry import make_strategy
from ..backtest.engine import BacktestConfig, BacktestEngine
from ..backtest.costs import cost_model_from_config

PROJ_ROOT = Path(__file__).resolve().parents[2]
SIGNALS_LOG = PROJ_ROOT / "artifacts" / "live" / "signals_log.jsonl"
PAPER_STATE = PROJ_ROOT / "artifacts" / "live" / "paper_state.json"


@dataclass
class Signal:
    """A point-in-time signal from the strategy."""
    timestamp: str
    epoch: int
    target_weights: Dict[str, float]
    previous_weights: Dict[str, float]
    trades_needed: List[Dict[str, Any]]
    vol_tilt_active: bool
    vol_tilt_z_score: float
    gross_leverage: float
    net_exposure: float
    sub_signals: Dict[str, Dict[str, float]]
    prices: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "epoch": self.epoch,
            "target_weights": self.target_weights,
            "previous_weights": self.previous_weights,
            "trades_needed": self.trades_needed,
            "vol_tilt_active": self.vol_tilt_active,
            "vol_tilt_z_score": round(self.vol_tilt_z_score, 4),
            "gross_leverage": round(self.gross_leverage, 4),
            "net_exposure": round(self.net_exposure, 4),
            "sub_signals": self.sub_signals,
            "prices": self.prices,
            "meta": self.meta,
        }


@dataclass
class PaperState:
    """Paper trading state — tracks hypothetical positions."""
    weights: Dict[str, float]
    equity: float
    entry_prices: Dict[str, float]
    last_signal_epoch: int
    pnl_history: List[Dict[str, Any]]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "equity": round(self.equity, 4),
            "entry_prices": self.entry_prices,
            "last_signal_epoch": self.last_signal_epoch,
            "pnl_history": self.pnl_history[-100:],  # keep last 100 entries
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def load(cls) -> Optional["PaperState"]:
        if not PAPER_STATE.exists():
            return None
        try:
            with open(PAPER_STATE) as f:
                d = json.load(f)
            return cls(**d)
        except Exception:
            return None

    def save(self) -> None:
        PAPER_STATE.parent.mkdir(parents=True, exist_ok=True)
        with open(PAPER_STATE, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class SignalGenerator:
    """
    Generates real-time trading signals from the P91b champion strategy.

    Flow:
    1. Fetch latest 600h of data from Binance REST (covers longest lookback + buffer)
    2. Run each sub-strategy (V1, I460, I415, F144) independently
    3. Blend with P91b ensemble weights
    4. Compute volume momentum z-score for vol tilt
    5. Apply tilt if z > 0 (reduce leverage by tilt_ratio)
    6. Output target weights + trade deltas
    """

    def __init__(
        self,
        symbols: List[str],
        ensemble_weights: Dict[str, float],
        signal_configs: Dict[str, Dict[str, Any]],
        vol_tilt_lookback: int = 168,
        vol_tilt_ratio: float = 0.65,
        vol_tilt_enabled: bool = True,
        warmup_bars: int = 600,
        paper_equity: float = 100000.0,
    ) -> None:
        self.symbols = symbols
        self.ensemble_weights = ensemble_weights
        self.signal_configs = signal_configs
        self.vol_tilt_lookback = vol_tilt_lookback
        self.vol_tilt_ratio = vol_tilt_ratio
        self.vol_tilt_enabled = vol_tilt_enabled
        self.warmup_bars = warmup_bars
        self.paper_equity = paper_equity

        # Paper state
        self._paper = PaperState.load()

    @classmethod
    def from_production_config(cls, config_path: Optional[str] = None) -> "SignalGenerator":
        """Initialize from production config JSON."""
        if config_path is None:
            config_path = str(PROJ_ROOT / "configs" / "production_p91b_champion.json")
        with open(config_path) as f:
            cfg = json.load(f)

        symbols = cfg["data"]["symbols"]
        ens = cfg["ensemble"]
        vol = cfg.get("volume_tilt", {})

        return cls(
            symbols=symbols,
            ensemble_weights=ens["weights"],
            signal_configs=ens["signals"],
            vol_tilt_lookback=vol.get("lookback_bars", 168),
            vol_tilt_ratio=vol.get("tilt_ratio", 0.65),
            vol_tilt_enabled=vol.get("enabled", True),
            warmup_bars=cfg.get("operational", {}).get("warmup_bars_required", 600),
        )

    def _fetch_data(self) -> MarketDataset:
        """Fetch latest data from Binance (enough for strategy warmup)."""
        now_epoch = int(time.time())
        # Fetch warmup_bars hours back
        start_epoch = now_epoch - self.warmup_bars * 3600
        start_iso = datetime.fromtimestamp(start_epoch, tz=timezone.utc).strftime("%Y-%m-%d")
        end_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        provider = make_provider({
            "provider": "binance_rest_v1",
            "symbols": self.symbols,
            "bar_interval": "1h",
            "start": start_iso,
            "end": end_iso,
        }, seed=42)
        return provider.load()

    def _run_sub_strategy(
        self, dataset: MarketDataset, sig_key: str
    ) -> Dict[str, float]:
        """Run a single sub-strategy and return its target weights at the last bar."""
        cfg = self.signal_configs[sig_key]
        strat = make_strategy({"name": cfg["strategy"], "params": cfg["params"]})

        n = len(dataset.timeline)
        # Walk through all bars to let the strategy build state
        current_weights: Dict[str, float] = {s: 0.0 for s in dataset.symbols}
        for idx in range(1, n):
            if strat.should_rebalance(dataset, idx):
                current_weights = strat.target_weights(dataset, idx, current_weights)

        return current_weights

    def _compute_vol_z(self, dataset: MarketDataset) -> float:
        """Compute volume momentum z-score at the last bar."""
        total_vol = None
        for sym in self.symbols:
            vols = dataset.perp_volume.get(sym) if dataset.perp_volume else None
            if vols is None:
                continue
            arr = np.array(vols, dtype=np.float64)
            if total_vol is None:
                total_vol = arr.copy()
            else:
                min_l = min(len(total_vol), len(arr))
                total_vol = total_vol[:min_l] + arr[:min_l]

        if total_vol is None or len(total_vol) < self.vol_tilt_lookback * 2:
            return 0.0

        log_vol = np.log(np.maximum(total_vol, 1.0))
        lb = self.vol_tilt_lookback
        n = len(log_vol)

        # Momentum at last bar
        if n < lb + 1:
            return 0.0
        mom_last = log_vol[-1] - log_vol[-1 - lb]

        # Rolling stats for z-score (using last lb values of momentum)
        moms = []
        for i in range(max(lb, n - lb), n):
            if i >= lb:
                moms.append(log_vol[i] - log_vol[i - lb])
        if len(moms) < 10:
            return 0.0

        mu = float(np.mean(moms))
        sigma = float(np.std(moms))
        if sigma <= 0:
            return 0.0
        return float((mom_last - mu) / sigma)

    def generate(self) -> Signal:
        """
        Generate a signal at the current moment.

        Returns a Signal object with target weights and trade deltas.
        """
        # 1. Fetch data
        dataset = self._fetch_data()
        n = len(dataset.timeline)
        last_epoch = dataset.timeline[-1]

        # 2. Run each sub-strategy
        sub_weights: Dict[str, Dict[str, float]] = {}
        for sig_key in self.signal_configs:
            sub_weights[sig_key] = self._run_sub_strategy(dataset, sig_key)

        # 3. Blend with ensemble weights
        blended: Dict[str, float] = {s: 0.0 for s in self.symbols}
        for sig_key, bw in self.ensemble_weights.items():
            sw = sub_weights.get(sig_key, {})
            for s in self.symbols:
                blended[s] += bw * sw.get(s, 0.0)

        # 4. Compute vol tilt
        vol_z = self._compute_vol_z(dataset)
        vol_tilt_active = self.vol_tilt_enabled and vol_z > 0

        if vol_tilt_active:
            for s in self.symbols:
                blended[s] *= self.vol_tilt_ratio

        # 5. Compute metrics
        gross = sum(abs(w) for w in blended.values())
        net = sum(blended.values())

        # 6. Get current prices
        prices = {s: dataset.close(s, n - 1) for s in self.symbols}

        # 7. Compute trade deltas vs previous weights
        prev_weights = {}
        if self._paper:
            prev_weights = self._paper.weights
        else:
            prev_weights = {s: 0.0 for s in self.symbols}

        trades_needed = []
        for s in self.symbols:
            target = blended.get(s, 0.0)
            current = prev_weights.get(s, 0.0)
            delta = target - current
            if abs(delta) > 0.001:  # 0.1% threshold
                trades_needed.append({
                    "symbol": s,
                    "current_weight": round(current, 6),
                    "target_weight": round(target, 6),
                    "delta": round(delta, 6),
                    "direction": "BUY" if delta > 0 else "SELL",
                    "notional_usd": round(abs(delta) * self.paper_equity, 2),
                    "price": prices[s],
                })

        # Sort by absolute delta descending
        trades_needed.sort(key=lambda t: abs(t["delta"]), reverse=True)

        # 8. Build Signal
        now = datetime.now(timezone.utc)
        signal = Signal(
            timestamp=now.isoformat(),
            epoch=last_epoch,
            target_weights={s: round(w, 6) for s, w in blended.items()},
            previous_weights={s: round(prev_weights.get(s, 0.0), 6) for s in self.symbols},
            trades_needed=trades_needed,
            vol_tilt_active=vol_tilt_active,
            vol_tilt_z_score=vol_z,
            gross_leverage=gross,
            net_exposure=net,
            sub_signals={k: {s: round(v.get(s, 0.0), 6) for s in self.symbols} for k, v in sub_weights.items()},
            prices=prices,
            meta={
                "data_bars": n,
                "last_bar_epoch": last_epoch,
                "last_bar_utc": datetime.fromtimestamp(last_epoch, tz=timezone.utc).isoformat(),
            },
        )

        # 9. Update paper state
        self._update_paper(signal)

        # 10. Log signal
        self._log_signal(signal)

        return signal

    def _update_paper(self, signal: Signal) -> None:
        """Update paper trading state with new signal."""
        now_iso = datetime.now(timezone.utc).isoformat()

        if self._paper is None:
            self._paper = PaperState(
                weights={s: 0.0 for s in self.symbols},
                equity=self.paper_equity,
                entry_prices={s: signal.prices[s] for s in self.symbols},
                last_signal_epoch=signal.epoch,
                pnl_history=[],
                created_at=now_iso,
                updated_at=now_iso,
            )

        # Compute P&L since last signal (mark-to-market)
        if self._paper.last_signal_epoch > 0 and self._paper.entry_prices:
            pnl = 0.0
            for s in self.symbols:
                w = self._paper.weights.get(s, 0.0)
                if abs(w) > 1e-8:
                    entry_p = self._paper.entry_prices.get(s, 0.0)
                    curr_p = signal.prices.get(s, 0.0)
                    if entry_p > 0:
                        ret = (curr_p / entry_p) - 1.0
                        pnl += self._paper.equity * w * ret

            self._paper.equity += pnl
            self._paper.pnl_history.append({
                "epoch": signal.epoch,
                "timestamp": signal.timestamp,
                "pnl": round(pnl, 2),
                "equity": round(self._paper.equity, 2),
                "gross_leverage": round(signal.gross_leverage, 4),
            })

        # Update positions
        self._paper.weights = dict(signal.target_weights)
        self._paper.entry_prices = dict(signal.prices)
        self._paper.last_signal_epoch = signal.epoch
        self._paper.updated_at = now_iso
        self._paper.save()

    def _log_signal(self, signal: Signal) -> None:
        """Append signal to JSONL log."""
        SIGNALS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SIGNALS_LOG, "a") as f:
            f.write(json.dumps(signal.to_dict(), ensure_ascii=False) + "\n")


def generate_signal_cli(config_path: Optional[str] = None) -> Signal:
    """CLI helper: generate one signal and print it."""
    gen = SignalGenerator.from_production_config(config_path)
    print("[SIGNAL] Fetching latest data from Binance...", flush=True)
    signal = gen.generate()

    print(f"\n{'='*70}", flush=True)
    print(f"P91b CHAMPION — REAL-TIME SIGNAL @ {signal.timestamp}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Data bars: {signal.meta['data_bars']} | Last bar: {signal.meta['last_bar_utc']}", flush=True)
    print(f"  Vol tilt: {'ACTIVE (z={signal.vol_tilt_z_score:.2f})' if signal.vol_tilt_active else f'OFF (z={signal.vol_tilt_z_score:.2f})'}", flush=True)
    print(f"  Gross leverage: {signal.gross_leverage:.4f} | Net exposure: {signal.net_exposure:.4f}", flush=True)

    print(f"\n  TARGET WEIGHTS:", flush=True)
    for s in sorted(signal.target_weights.keys(), key=lambda x: abs(signal.target_weights[x]), reverse=True):
        w = signal.target_weights[s]
        if abs(w) > 0.001:
            direction = "LONG" if w > 0 else "SHORT"
            print(f"    {s:12s}  {w:+.4f}  ({direction})", flush=True)

    if signal.trades_needed:
        print(f"\n  TRADES NEEDED ({len(signal.trades_needed)}):", flush=True)
        for t in signal.trades_needed:
            print(f"    {t['direction']:4s} {t['symbol']:12s} Δ={t['delta']:+.4f} (~${t['notional_usd']:,.0f} @ ${t['price']:,.2f})", flush=True)
    else:
        print(f"\n  NO TRADES NEEDED (positions unchanged)", flush=True)

    # Paper P&L
    paper = PaperState.load()
    if paper:
        init_eq = 100000.0
        pnl_pct = (paper.equity / init_eq - 1) * 100
        print(f"\n  PAPER TRADING:", flush=True)
        print(f"    Equity: ${paper.equity:,.2f} ({pnl_pct:+.2f}%)", flush=True)
        print(f"    History: {len(paper.pnl_history)} signals tracked", flush=True)

    print(f"\n  Signal logged → {SIGNALS_LOG}", flush=True)
    return signal
