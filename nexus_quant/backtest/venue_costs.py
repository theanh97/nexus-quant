from __future__ import annotations

"""
Binance USDM perpetual VIP fee tier structure and helpers.

All rates are stored as fractions of notional (e.g. 0.0002 = 2 bps = 0.02%).
Stdlib-only; no third-party dependencies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VenueFeeTier:
    """
    Single fee tier for a venue/VIP level.

    Attributes
    ----------
    name:
        Human-readable label, e.g. ``"binance_usdm_vip0"``.
    vip_tier:
        Integer VIP level (0 = retail default, 9 = top tier).
    maker_rate:
        Maker rebate/fee as a fraction of notional.  Positive = fee paid,
        zero = zero fee (VIP 9 Binance USDM maker is free).
    taker_rate:
        Taker fee as a fraction of notional.  Always positive.
    volume_threshold_usd:
        Minimum 30-day traded volume in USD required for this tier.
        Zero for VIP 0 (no requirement).
    """

    name: str
    vip_tier: int
    maker_rate: float
    taker_rate: float
    volume_threshold_usd: float


# ---------------------------------------------------------------------------
# Binance USDM tier table
# Rates sourced from Binance USDM fee schedule (as of 2024).
# https://www.binance.com/en/fee/futureFee
#
# Format: (vip, maker_bps, taker_bps, volume_threshold_usd)
# ---------------------------------------------------------------------------

_BINANCE_USDM_RAW: List[Tuple[int, float, float, float]] = [
    (0, 2.00, 5.00, 0.0),                 # VIP 0  – retail default
    (1, 1.60, 4.00, 15_000_000.0),        # VIP 1  – 15 M USDT 30d vol
    (2, 1.40, 3.50, 150_000_000.0),       # VIP 2  – 150 M
    (3, 1.20, 3.00, 1_500_000_000.0),     # VIP 3  – 1.5 B
    (4, 1.00, 2.50, 15_000_000_000.0),    # VIP 4  – 15 B
    (5, 0.80, 2.00, 150_000_000_000.0),   # VIP 5  – 150 B
    (6, 0.60, 1.70, 500_000_000_000.0),   # VIP 6  – 500 B
    (7, 0.40, 1.50, 1_000_000_000_000.0), # VIP 7  – 1 T
    (8, 0.20, 1.20, 2_000_000_000_000.0), # VIP 8  – 2 T
    (9, 0.00, 1.00, 5_000_000_000_000.0), # VIP 9  – 5 T
]

# Public constant – full list of all 10 Binance USDM VIP tiers.
BINANCE_USDM_TIERS: List[VenueFeeTier] = [
    VenueFeeTier(
        name=f"binance_usdm_vip{vip}",
        vip_tier=vip,
        maker_rate=round(maker_bps / 10_000.0, 10),
        taker_rate=round(taker_bps / 10_000.0, 10),
        volume_threshold_usd=vol_usd,
    )
    for vip, maker_bps, taker_bps, vol_usd in _BINANCE_USDM_RAW
]

# Fast lookup index: vip_tier -> VenueFeeTier
_BINANCE_USDM_BY_VIP: Dict[int, VenueFeeTier] = {
    tier.vip_tier: tier for tier in BINANCE_USDM_TIERS
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_binance_tier(vip_tier: int) -> VenueFeeTier:
    """
    Return the ``VenueFeeTier`` for the given Binance USDM VIP level.

    Parameters
    ----------
    vip_tier:
        Integer between 0 and 9 inclusive.

    Returns
    -------
    VenueFeeTier
        The matching tier data.

    Raises
    ------
    ValueError
        If *vip_tier* is not in the range 0–9.
    """
    tier = _BINANCE_USDM_BY_VIP.get(int(vip_tier))
    if tier is None:
        valid = sorted(_BINANCE_USDM_BY_VIP.keys())
        raise ValueError(
            f"Unknown Binance USDM VIP tier: {vip_tier!r}. "
            f"Valid tiers are: {valid}"
        )
    return tier


def venue_fee_model_from_config(
    venue_cfg: Dict,
    costs_cfg: Dict,
) -> Tuple[float, float]:
    """
    Resolve ``(maker_rate, taker_rate)`` from a venue config dict and a costs
    config dict.

    Resolution order
    ----------------
    1. If ``venue_cfg["name"] == "binance_usdm"`` **and** ``venue_cfg``
       contains a ``"vip_tier"`` key, look up the corresponding
       ``VenueFeeTier`` from :data:`BINANCE_USDM_TIERS` and return its
       rates.
    2. Fall back to explicit ``costs_cfg["maker_fee_rate"]`` /
       ``costs_cfg["taker_fee_rate"]`` if present.
    3. Fall back to the legacy scalar ``costs_cfg["fee_rate"]`` applied to
       both maker and taker.
    4. Default to ``(0.0, 0.0005)`` (0 maker / 5 bps taker) – a conservative
       retail-ish default that avoids accidentally zero-cost simulations.

    Parameters
    ----------
    venue_cfg:
        Mapping that may contain ``"name"`` and ``"vip_tier"`` keys.
    costs_cfg:
        Mapping that may contain ``"maker_fee_rate"``, ``"taker_fee_rate"``,
        or the legacy ``"fee_rate"`` scalar.

    Returns
    -------
    Tuple[float, float]
        ``(maker_rate, taker_rate)`` as fractions of notional.
    """
    venue_cfg = venue_cfg or {}
    costs_cfg = costs_cfg or {}

    # --- 1. Binance USDM VIP tier lookup ---
    venue_name = str(venue_cfg.get("name") or "").strip().lower()
    if venue_name == "binance_usdm":
        raw_vip = venue_cfg.get("vip_tier")
        if raw_vip is not None:
            try:
                vip_int = int(raw_vip)
                tier = get_binance_tier(vip_int)
                return (tier.maker_rate, tier.taker_rate)
            except (ValueError, TypeError):
                # Malformed vip_tier – fall through to cost_cfg fallbacks.
                pass

    # --- 2. Explicit maker/taker rates in costs_cfg ---
    explicit_maker = costs_cfg.get("maker_fee_rate")
    explicit_taker = costs_cfg.get("taker_fee_rate")
    if explicit_maker is not None or explicit_taker is not None:
        maker = max(0.0, float(explicit_maker or 0.0))
        taker = max(0.0, float(explicit_taker or 0.0))
        return (maker, taker)

    # --- 3. Legacy scalar fee_rate ---
    legacy = costs_cfg.get("fee_rate")
    if legacy is not None:
        fr = max(0.0, float(legacy))
        return (fr, fr)

    # --- 4. Conservative default (VIP 0 taker, zero maker) ---
    default_tier = _BINANCE_USDM_BY_VIP[0]
    return (0.0, default_tier.taker_rate)
