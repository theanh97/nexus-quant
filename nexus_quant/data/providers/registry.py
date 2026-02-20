from __future__ import annotations

from typing import Any, Dict

from .base import DataProvider
from .synthetic import SyntheticPerpV1Provider
from .local_csv import LocalCSVProvider
from .binance_rest import BinanceRestProvider


def make_provider(data_cfg: Dict[str, Any], seed: int) -> DataProvider:
    name = str(data_cfg.get("provider") or "")
    if name == "synthetic_perp_v1":
        return SyntheticPerpV1Provider(data_cfg, seed=seed)
    if name == "local_csv_v1":
        return LocalCSVProvider(data_cfg, seed=seed)
    if name == "binance_rest_v1":
        return BinanceRestProvider(data_cfg, seed=seed)
    if name == "yahoo_futures_v1":
        from nexus_quant.projects.commodity_cta.providers.yahoo_futures import (
            YahooFuturesProvider,
        )
        return YahooFuturesProvider(data_cfg, seed=seed)
    raise ValueError(f"Unknown data provider: {name}")
