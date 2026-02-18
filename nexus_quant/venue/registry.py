from __future__ import annotations

from typing import Dict, List

from .spec import VenueSpec


# NOTE: Fees can change by time / tier / region. Treat defaults as placeholders.
_VENUES: Dict[str, VenueSpec] = {
    # Crypto perps
    "binance_usdm": VenueSpec(name="binance_usdm", kind="crypto_perp", funding_interval_hours=8, notes="USDT-margined perpetuals"),
    "bybit_usdt": VenueSpec(name="bybit_usdt", kind="crypto_perp", funding_interval_hours=8, notes="USDT perpetuals"),
    "okx_swap": VenueSpec(name="okx_swap", kind="crypto_perp", funding_interval_hours=8, notes="Perpetual swaps"),
    "deribit_perp": VenueSpec(name="deribit_perp", kind="crypto_perp", funding_interval_hours=8, notes="Perpetual futures"),

    # Crypto spot
    "binance_spot": VenueSpec(name="binance_spot", kind="crypto_spot", notes="Spot exchange"),
    "coinbase_spot": VenueSpec(name="coinbase_spot", kind="crypto_spot", notes="Spot exchange"),

    # FX / CFDs (examples)
    "oanda_fx": VenueSpec(name="oanda_fx", kind="fx", notes="Retail FX broker (fees via spread/commission)."),
    "icmarkets_fx": VenueSpec(name="icmarkets_fx", kind="fx", notes="Retail FX broker (fees via spread/commission)."),
}


def list_venues() -> List[str]:
    return sorted(_VENUES.keys())


def get_venue(name: str) -> VenueSpec:
    key = (name or "").strip().lower()
    if key not in _VENUES:
        raise KeyError(f"Unknown venue: {name}. Known: {', '.join(list_venues())}")
    return _VENUES[key]

