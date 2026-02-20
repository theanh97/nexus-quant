"""
Volatility Surface Builder for Crypto Options

Builds a vol surface from a collection of (strike, expiry, IV) points
and extracts key features:
    - ATM IV (at-the-money)
    - 25-delta skew (put IV - call IV)
    - Butterfly spread (25d put + 25d call - 2*ATM)
    - Term structure slope (front vs back month)

Uses stdlib-only (no scipy/numpy). Linear interpolation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class OptionPoint:
    """A single option data point for surface construction."""
    strike: float           # Strike price (e.g., 50000.0 for BTC)
    expiry_ts: int          # Expiry unix timestamp (seconds)
    tte: float              # Time to expiry in years (computed from snapshot_ts)
    option_type: str        # "call" or "put"
    iv: float               # Implied vol (annualized, e.g., 0.65 = 65%)
    delta: float            # Option delta
    mark_price: float       # Mark price in USD
    underlying: float       # Underlying price at snapshot time


@dataclass
class VolSmile:
    """Vol smile for a single expiry (collection of strike→IV pairs)."""
    expiry_ts: int
    tte: float              # Time to expiry in years
    underlying: float       # Spot price at time of snapshot
    strikes: List[float] = field(default_factory=list)      # Sorted strikes
    ivs: List[float] = field(default_factory=list)           # Matching IVs
    deltas: List[float] = field(default_factory=list)        # Matching deltas
    option_types: List[str] = field(default_factory=list)    # "call"/"put"

    def atm_iv(self) -> Optional[float]:
        """Extract ATM IV by interpolation at moneyness=0 (strike==underlying)."""
        if not self.strikes or not self.ivs:
            return None
        target = self.underlying
        return _interp_sorted(self.strikes, self.ivs, target)

    def iv_at_moneyness(self, log_moneyness: float) -> Optional[float]:
        """Get IV at given log-moneyness log(K/S).
        log_moneyness=0 → ATM
        log_moneyness<0 → OTM put
        log_moneyness>0 → OTM call
        """
        if not self.strikes or not self.ivs:
            return None
        target_strike = self.underlying * math.exp(log_moneyness)
        return _interp_sorted(self.strikes, self.ivs, target_strike)

    def iv_at_delta(self, target_delta: float) -> Optional[float]:
        """Interpolate IV at a given delta (e.g., 0.25 for 25-delta call).
        Uses absolute delta values for simplicity.
        """
        if not self.deltas or not self.ivs:
            return None
        abs_deltas = [abs(d) for d in self.deltas]
        # Sort by delta for interpolation
        pairs = sorted(zip(abs_deltas, self.ivs))
        ds = [p[0] for p in pairs]
        ivs = [p[1] for p in pairs]
        return _interp_sorted(ds, ivs, abs(target_delta))


@dataclass
class VolSurface:
    """Full vol surface across strikes and expiries."""
    snapshot_ts: int                  # Unix timestamp of snapshot
    underlying_price: Dict[str, float]  # {symbol: price}
    smiles: Dict[str, List[VolSmile]]   # {symbol: [smile_front, smile_back, ...]}

    def get_smile(self, symbol: str, target_tte: float) -> Optional[VolSmile]:
        """Get smile nearest to target_tte (time to expiry in years)."""
        smiles = self.smiles.get(symbol, [])
        if not smiles:
            return None
        best = min(smiles, key=lambda s: abs(s.tte - target_tte))
        return best

    def term_structure(self, symbol: str) -> Optional[float]:
        """Front-month IV minus back-month IV.
        Positive = contango (front > back, normal)
        Negative = backwardation (front < back, stress)
        """
        smiles = sorted(self.smiles.get(symbol, []), key=lambda s: s.tte)
        if len(smiles) < 2:
            return None
        front_iv = smiles[0].atm_iv()
        back_iv = smiles[-1].atm_iv()
        if front_iv is None or back_iv is None:
            return None
        return front_iv - back_iv

    def extract_features(self, symbol: str) -> Dict[str, Optional[float]]:
        """Extract all key vol surface features for a symbol.

        Returns dict with:
            iv_atm: ATM IV (nearest expiry)
            iv_25d_put: 25-delta put IV
            iv_25d_call: 25-delta call IV
            skew_25d: put_iv - call_iv (positive = puts more expensive)
            butterfly_25d: 0.5*(put_iv + call_iv) - atm_iv
            term_spread: front_atm_iv - back_atm_iv
            rv_premium: IV - RV (set later by provider)
        """
        out: Dict[str, Optional[float]] = {
            "iv_atm": None,
            "iv_25d_put": None,
            "iv_25d_call": None,
            "skew_25d": None,
            "butterfly_25d": None,
            "term_spread": None,
        }

        smiles = sorted(self.smiles.get(symbol, []), key=lambda s: s.tte)
        if not smiles:
            return out

        # Nearest expiry smile
        front = smiles[0]
        atm = front.atm_iv()
        out["iv_atm"] = atm

        # 25-delta put (delta=-0.25) and call (delta=+0.25)
        iv_put_25 = front.iv_at_delta(-0.25)
        iv_call_25 = front.iv_at_delta(0.25)
        out["iv_25d_put"] = iv_put_25
        out["iv_25d_call"] = iv_call_25

        if iv_put_25 is not None and iv_call_25 is not None:
            out["skew_25d"] = iv_put_25 - iv_call_25
            if atm is not None:
                out["butterfly_25d"] = 0.5 * (iv_put_25 + iv_call_25) - atm

        # Term structure
        out["term_spread"] = self.term_structure(symbol)

        return out


# ── Interpolation helpers ─────────────────────────────────────────────────────

def _interp_sorted(xs: List[float], ys: List[float], target: float) -> Optional[float]:
    """Linear interpolation (or extrapolation clamp) on sorted xs."""
    if not xs or not ys or len(xs) != len(ys):
        return None
    if len(xs) == 1:
        return ys[0]

    # Clamp to range
    if target <= xs[0]:
        return ys[0]
    if target >= xs[-1]:
        return ys[-1]

    # Binary search for bracket
    lo, hi = 0, len(xs) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if xs[mid] <= target:
            lo = mid
        else:
            hi = mid

    # Linear interpolation
    t = (target - xs[lo]) / (xs[hi] - xs[lo]) if xs[hi] != xs[lo] else 0.0
    return ys[lo] + t * (ys[hi] - ys[lo])


# ── Surface builder ────────────────────────────────────────────────────────────

class VolSurfaceBuilder:
    """Build vol surfaces from raw Deribit option data."""

    def __init__(self, min_tte_days: float = 1.0, max_tte_days: float = 365.0):
        """
        Args:
            min_tte_days: ignore options expiring in less than N days
            max_tte_days: ignore options expiring beyond N days
        """
        self.min_tte = min_tte_days / 365.0
        self.max_tte = max_tte_days / 365.0

    def build(
        self,
        snapshot_ts: int,
        options: List[OptionPoint],
        underlying_prices: Dict[str, float],
    ) -> VolSurface:
        """Build a VolSurface from a list of OptionPoints.

        Args:
            snapshot_ts: timestamp for this snapshot
            options: list of OptionPoint records
            underlying_prices: {symbol: spot_price}

        Returns:
            VolSurface instance
        """
        # Group by symbol, then by expiry
        by_symbol: Dict[str, Dict[int, List[OptionPoint]]] = {}
        for opt in options:
            if opt.iv <= 0 or opt.underlying <= 0:
                continue
            if not (self.min_tte <= opt.tte <= self.max_tte):
                continue

            # Determine symbol from option name (e.g., "BTC-30DEC24-50000-C" → "BTC")
            sym = _infer_symbol(opt)
            if sym not in by_symbol:
                by_symbol[sym] = {}
            if opt.expiry_ts not in by_symbol[sym]:
                by_symbol[sym][opt.expiry_ts] = []
            by_symbol[sym][opt.expiry_ts].append(opt)

        # Build smiles
        smiles: Dict[str, List[VolSmile]] = {}
        for sym, by_expiry in by_symbol.items():
            sym_smiles = []
            for expiry_ts, pts in by_expiry.items():
                if not pts:
                    continue
                smile = self._build_smile(expiry_ts, pts)
                if smile is not None:
                    sym_smiles.append(smile)
            smiles[sym] = sorted(sym_smiles, key=lambda s: s.tte)

        return VolSurface(
            snapshot_ts=snapshot_ts,
            underlying_price=underlying_prices,
            smiles=smiles,
        )

    def _build_smile(self, expiry_ts: int, pts: List[OptionPoint]) -> Optional[VolSmile]:
        """Build a single smile for one expiry."""
        if not pts:
            return None

        underlying = pts[0].underlying
        tte = pts[0].tte

        # Separate puts and calls, deduplicate by combining nearby strikes
        # Sort by strike
        pairs: List[Tuple[float, float, float, str]] = []  # (strike, iv, delta, type)
        for p in pts:
            if p.iv > 0 and p.mark_price > 0:
                pairs.append((p.strike, p.iv, p.delta, p.option_type))

        if not pairs:
            return None

        # Sort by strike
        pairs.sort(key=lambda x: x[0])

        strikes = [p[0] for p in pairs]
        ivs = [p[1] for p in pairs]
        deltas = [p[2] for p in pairs]
        types = [p[3] for p in pairs]

        return VolSmile(
            expiry_ts=expiry_ts,
            tte=tte,
            underlying=underlying,
            strikes=strikes,
            ivs=ivs,
            deltas=deltas,
            option_types=types,
        )


def _infer_symbol(opt: OptionPoint) -> str:
    """Infer underlying symbol from option data. Default to BTC."""
    # In practice the DataProvider sets this; fallback here
    if opt.underlying > 10000:
        return "BTC"
    return "ETH"


# ── Rolling feature extractor ─────────────────────────────────────────────────

class RollingVolFeatures:
    """Maintains rolling history of vol surface features for signal generation.

    Maintains per-symbol, per-feature lists aligned with a timeline.
    Used to build the MarketDataset.features dict.
    """

    FEATURE_NAMES = [
        "iv_atm",
        "iv_25d_put",
        "iv_25d_call",
        "skew_25d",
        "butterfly_25d",
        "term_spread",
        "rv_realized",
    ]

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        # {feature_name: {symbol: [values_per_bar]}}
        self.data: Dict[str, Dict[str, List[Optional[float]]]] = {
            f: {s: [] for s in symbols} for f in self.FEATURE_NAMES
        }

    def append_bar(self, surface: VolSurface, rv_data: Dict[str, Optional[float]] = None):
        """Append one bar of features from a vol surface snapshot."""
        rv_data = rv_data or {}
        for sym in self.symbols:
            feats = surface.extract_features(sym)
            for fname in self.FEATURE_NAMES:
                if fname == "rv_realized":
                    val = rv_data.get(sym)
                else:
                    val = feats.get(fname)
                self.data[fname][sym].append(val)

    def to_features_dict(self) -> Dict[str, Dict[str, List[Optional[float]]]]:
        """Return the features dict for MarketDataset."""
        return {f: dict(v) for f, v in self.data.items()}

    def length(self) -> int:
        """Number of bars recorded."""
        for fname in self.FEATURE_NAMES:
            for sym in self.symbols:
                return len(self.data[fname][sym])
        return 0


# ── Realized vol calculator ────────────────────────────────────────────────────

def realized_vol(prices: List[float], lookback: int = 504) -> Optional[float]:
    """Compute close-to-close realized vol (annualized).

    Args:
        prices: list of closing prices (most recent last)
        lookback: number of bars (e.g., 504 = 21 days of hourly bars)

    Returns:
        Annualized realized vol or None if insufficient data
    """
    if len(prices) < lookback + 1:
        return None

    recent = prices[-(lookback + 1):]
    log_rets = [math.log(recent[i] / recent[i - 1]) for i in range(1, len(recent))]

    n = len(log_rets)
    if n < 2:
        return None

    mean = sum(log_rets) / n
    variance = sum((r - mean) ** 2 for r in log_rets) / (n - 1)

    # Annualize: hourly bars → multiply by sqrt(8760)
    annual_factor = math.sqrt(8760)
    return math.sqrt(variance) * annual_factor


def realized_vol_daily(prices: List[float], lookback_days: int = 21) -> Optional[float]:
    """Realized vol from daily prices (close-to-close, annualized).

    Args:
        prices: daily closing prices
        lookback_days: rolling window

    Returns:
        Annualized realized vol
    """
    if len(prices) < lookback_days + 1:
        return None

    recent = prices[-(lookback_days + 1):]
    log_rets = [math.log(recent[i] / recent[i - 1]) for i in range(1, len(recent))]

    n = len(log_rets)
    if n < 2:
        return None

    mean = sum(log_rets) / n
    variance = sum((r - mean) ** 2 for r in log_rets) / (n - 1)
    return math.sqrt(variance * 252)
