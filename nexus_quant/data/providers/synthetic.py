from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ...utils.hashing import sha256_text
from ...utils.time import parse_iso_utc
from ..schema import MarketDataset
from .base import DataProvider


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class _GenConfig:
    start: int
    end: int
    symbols: Tuple[str, ...]
    bar_interval_minutes: int
    funding_interval_hours: int


class SyntheticPerpV1Provider(DataProvider):
    """
    Deterministic synthetic dataset:
    - spot close series per symbol (correlated)
    - perp close = spot * (1 + basis)
    - funding every N hours, correlated with basis (plus noise)
    """

    def load(self) -> MarketDataset:
        start = parse_iso_utc(str(self.cfg["start"]))
        end = parse_iso_utc(str(self.cfg["end"]))
        symbols = tuple(str(s) for s in (self.cfg.get("symbols") or []))
        if not symbols:
            raise ValueError("synthetic_perp_v1: symbols is required")

        bar_interval_minutes = int(self.cfg.get("bar_interval_minutes") or 60)
        funding_interval_hours = int(self.cfg.get("funding_interval_hours") or 8)
        if bar_interval_minutes <= 0:
            raise ValueError("bar_interval_minutes must be > 0")
        if funding_interval_hours <= 0:
            raise ValueError("funding_interval_hours must be > 0")

        gen_cfg = _GenConfig(
            start=start,
            end=end,
            symbols=symbols,
            bar_interval_minutes=bar_interval_minutes,
            funding_interval_hours=funding_interval_hours,
        )
        fingerprint = self._fingerprint(gen_cfg)

        timeline = self._make_timeline(gen_cfg)
        spot_close, perp_close, funding = self._generate_series(gen_cfg, timeline)
        funding_times = {s: sorted(funding.get(s, {}).keys()) for s in symbols}

        return MarketDataset(
            provider="synthetic_perp_v1",
            timeline=timeline,
            symbols=list(symbols),
            perp_close=perp_close,
            spot_close=spot_close,
            funding=funding,
            fingerprint=fingerprint,
            _funding_times=funding_times,
        )

    def _fingerprint(self, gen_cfg: _GenConfig) -> str:
        payload = {
            "provider": "synthetic_perp_v1",
            "seed": self.seed,
            "start": gen_cfg.start,
            "end": gen_cfg.end,
            "symbols": list(gen_cfg.symbols),
            "bar_interval_minutes": gen_cfg.bar_interval_minutes,
            "funding_interval_hours": gen_cfg.funding_interval_hours,
        }
        return sha256_text(json.dumps(payload, sort_keys=True))

    def _make_timeline(self, gen_cfg: _GenConfig) -> List[int]:
        step = gen_cfg.bar_interval_minutes * 60
        if gen_cfg.end <= gen_cfg.start + step:
            raise ValueError("end must be after start by at least one bar")
        n = int((gen_cfg.end - gen_cfg.start) // step)
        return [gen_cfg.start + i * step for i in range(n)]

    def _generate_series(
        self, gen_cfg: _GenConfig, timeline: List[int]
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, Dict[int, float]]]:
        rng = random.Random(self.seed)

        # Return scale: keep it moderate (hourly). This is not meant to be "realistic market", only reproducible.
        market_sigma = 0.0012
        idio_sigma = 0.0016
        basis_phi = 0.97
        basis_sigma = 0.0008

        funding_sensitivity = 0.35  # maps basis to funding (roughly)
        funding_noise = 0.00005
        funding_cap = 0.0015  # cap per funding interval

        # Init prices per symbol (spot)
        spot0 = {}
        for s in gen_cfg.symbols:
            base = 100.0 + 50.0 * rng.random()
            if s.startswith("BTC"):
                base = 40000.0
            elif s.startswith("ETH"):
                base = 2500.0
            spot0[s] = base

        spot_close: Dict[str, List[float]] = {s: [spot0[s]] for s in gen_cfg.symbols}
        perp_close: Dict[str, List[float]] = {s: [] for s in gen_cfg.symbols}
        funding: Dict[str, Dict[int, float]] = {s: {} for s in gen_cfg.symbols}

        basis_state: Dict[str, float] = {s: 0.0 for s in gen_cfg.symbols}
        betas: Dict[str, float] = {s: 0.7 + 0.6 * rng.random() for s in gen_cfg.symbols}

        # First perp close uses initial spot and initial basis
        for s in gen_cfg.symbols:
            perp_close[s].append(spot_close[s][0] * (1.0 + basis_state[s]))

        funding_step_seconds = gen_cfg.funding_interval_hours * 3600

        for i in range(1, len(timeline)):
            # Shared market component
            m = rng.gauss(0.0, market_sigma)
            for s in gen_cfg.symbols:
                # Simple correlated log-return model
                r = betas[s] * m + rng.gauss(0.0, idio_sigma)
                spot_prev = spot_close[s][i - 1]
                spot_next = max(0.01, spot_prev * math.exp(r))
                spot_close[s].append(spot_next)

                # Basis AR(1)
                b = basis_state[s] * basis_phi + rng.gauss(0.0, basis_sigma)
                b = _clamp(b, -0.02, 0.02)  # keep basis small
                basis_state[s] = b
                perp_close[s].append(spot_next * (1.0 + b))

            ts = timeline[i]
            if (ts - timeline[0]) % funding_step_seconds == 0:
                for s in gen_cfg.symbols:
                    b = basis_state[s]
                    fr = b * funding_sensitivity + rng.gauss(0.0, funding_noise)
                    fr = _clamp(fr, -funding_cap, funding_cap)
                    funding[s][ts] = fr

        return spot_close, perp_close, funding

