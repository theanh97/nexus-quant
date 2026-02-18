from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VenueSpec:
    """
    Venue metadata (exchange/broker).

    This is used to make configs more explicit and auditable.
    We intentionally do NOT hardcode live fee tiers as "truth" here.
    """

    name: str
    kind: str  # crypto_perp | crypto_spot | fx | equities | options | ...
    funding_interval_hours: Optional[int] = None
    default_maker_fee_rate: Optional[float] = None
    default_taker_fee_rate: Optional[float] = None
    notes: str = ""

