from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...utils.hashing import sha256_text, file_sha256
from ...utils.time import parse_iso_utc
from ..schema import MarketDataset
from .base import DataProvider


class LocalCSVProvider(DataProvider):
    """
    Load aligned OHLCV+funding from local CSV files (no pandas required).

    Expected layout:
    - cfg["csv_dir"] contains one file per symbol: {SYMBOL}.csv
    - columns (required): timestamp, close
    - columns (optional): spot_close, funding_rate
    - columns (optional, higher fidelity): volume, spot_volume, mark_close, index_close, bid_close, ask_close

    Notes:
    - funding_rate is treated as event at that timestamp.
    - data must be aligned across symbols (same timestamps); otherwise the provider takes intersection.
    """

    def load(self) -> MarketDataset:
        csv_dir = Path(str(self.cfg.get("csv_dir") or "")).expanduser()
        if not csv_dir.exists():
            raise ValueError(f"local_csv_v1: csv_dir not found: {csv_dir}")

        symbols = [str(s) for s in (self.cfg.get("symbols") or [])]
        if not symbols:
            raise ValueError("local_csv_v1: symbols is required")

        series = {}
        for sym in symbols:
            path = csv_dir / f"{sym}.csv"
            if not path.exists():
                raise ValueError(f"local_csv_v1: missing file: {path}")
            series[sym] = self._read_one(path)

        # Align by intersection of timestamps.
        common_ts = None
        for sym in symbols:
            ts_set = set(series[sym]["timeline"])
            common_ts = ts_set if common_ts is None else common_ts.intersection(ts_set)
        timeline = sorted(common_ts or [])
        if len(timeline) < 2:
            raise ValueError("local_csv_v1: not enough aligned bars across symbols")

        perp_close = {}
        spot_close = {}
        perp_volume = {}
        spot_volume = {}
        perp_mark = {}
        perp_index = {}
        bid_close = {}
        ask_close = {}
        funding = {}
        funding_times = {}
        for sym in symbols:
            idx = {t: i for i, t in enumerate(series[sym]["timeline"])}
            perp_close[sym] = [series[sym]["perp_close"][idx[t]] for t in timeline]
            if series[sym]["spot_close"] is not None:
                spot_close[sym] = [series[sym]["spot_close"][idx[t]] for t in timeline]
            if series[sym].get("perp_volume") is not None:
                perp_volume[sym] = [series[sym]["perp_volume"][idx[t]] for t in timeline]
            if series[sym].get("spot_volume") is not None:
                spot_volume[sym] = [series[sym]["spot_volume"][idx[t]] for t in timeline]
            if series[sym].get("perp_mark_close") is not None:
                perp_mark[sym] = [series[sym]["perp_mark_close"][idx[t]] for t in timeline]
            if series[sym].get("perp_index_close") is not None:
                perp_index[sym] = [series[sym]["perp_index_close"][idx[t]] for t in timeline]
            if series[sym].get("bid_close") is not None:
                bid_close[sym] = [series[sym]["bid_close"][idx[t]] for t in timeline]
            if series[sym].get("ask_close") is not None:
                ask_close[sym] = [series[sym]["ask_close"][idx[t]] for t in timeline]
            funding[sym] = {t: r for t, r in series[sym]["funding"].items() if t in set(timeline)}
            funding_times[sym] = sorted(funding[sym].keys())

        spot = spot_close if spot_close else None
        perp_vol = perp_volume if perp_volume else None
        spot_vol = spot_volume if spot_volume else None
        mark = perp_mark if perp_mark else None
        index = perp_index if perp_index else None
        bid = bid_close if bid_close else None
        ask = ask_close if ask_close else None

        fingerprint = self._fingerprint(csv_dir, symbols, series)
        return MarketDataset(
            provider="local_csv_v1",
            timeline=timeline,
            symbols=symbols,
            perp_close=perp_close,
            spot_close=spot,
            funding=funding,
            fingerprint=fingerprint,
            perp_volume=perp_vol,
            spot_volume=spot_vol,
            perp_mark_close=mark,
            perp_index_close=index,
            bid_close=bid,
            ask_close=ask,
            _funding_times=funding_times,
        )

    def _read_one(self, path: Path) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        has_spot = False
        has_perp_volume = False
        has_spot_volume = False
        has_mark = False
        has_index = False
        has_bidask = False
        funding: Dict[int, float] = {}
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                ts_raw = row.get("timestamp") or row.get("ts") or row.get("time")
                if not ts_raw:
                    continue
                ts = parse_iso_utc(str(ts_raw))
                c = row.get("close")
                if c is None:
                    continue
                sc = row.get("spot_close")
                if sc is not None and sc != "":
                    has_spot = True
                v = row.get("volume") or row.get("perp_volume")
                if v is not None and v != "":
                    has_perp_volume = True
                sv = row.get("spot_volume")
                if sv is not None and sv != "":
                    has_spot_volume = True
                mk = row.get("mark_close") or row.get("perp_mark_close")
                if mk is not None and mk != "":
                    has_mark = True
                ix = row.get("index_close") or row.get("perp_index_close")
                if ix is not None and ix != "":
                    has_index = True
                bid = row.get("bid_close") or row.get("bid")
                ask = row.get("ask_close") or row.get("ask")
                if (bid is not None and bid != "") or (ask is not None and ask != ""):
                    has_bidask = True
                fr = row.get("funding_rate")
                if fr is not None and fr != "":
                    funding[ts] = float(fr)
                rows.append(
                    {
                        "ts": ts,
                        "close": float(c),
                        "spot_close": float(sc) if (sc is not None and sc != "") else None,
                        "perp_volume": float(v) if (v is not None and v != "") else None,
                        "spot_volume": float(sv) if (sv is not None and sv != "") else None,
                        "perp_mark_close": float(mk) if (mk is not None and mk != "") else None,
                        "perp_index_close": float(ix) if (ix is not None and ix != "") else None,
                        "bid_close": float(bid) if (bid is not None and bid != "") else None,
                        "ask_close": float(ask) if (ask is not None and ask != "") else None,
                    }
                )

        rows.sort(key=lambda x: x["ts"])

        timeline: List[int] = []
        perp_close: List[float] = []
        spot_close: List[float] = []
        perp_volume: List[float] = []
        spot_volume: List[float] = []
        perp_mark_close: List[float] = []
        perp_index_close: List[float] = []
        bid_close: List[float] = []
        ask_close: List[float] = []
        seen = set()
        for x in rows:
            ts = int(x["ts"])
            if ts in seen:
                raise ValueError(f"local_csv_v1: duplicate timestamp in {path}: {ts}")
            seen.add(ts)
            timeline.append(ts)
            perp_close.append(float(x["close"]))
            if has_spot:
                spot_close.append(float(x["spot_close"]) if x["spot_close"] is not None else 0.0)
            if has_perp_volume:
                perp_volume.append(float(x["perp_volume"]) if x["perp_volume"] is not None else 0.0)
            if has_spot_volume:
                spot_volume.append(float(x["spot_volume"]) if x["spot_volume"] is not None else 0.0)
            if has_mark:
                perp_mark_close.append(float(x["perp_mark_close"]) if x["perp_mark_close"] is not None else 0.0)
            if has_index:
                perp_index_close.append(float(x["perp_index_close"]) if x["perp_index_close"] is not None else 0.0)
            if has_bidask:
                bid_close.append(float(x["bid_close"]) if x["bid_close"] is not None else 0.0)
                ask_close.append(float(x["ask_close"]) if x["ask_close"] is not None else 0.0)

        out = {
            "timeline": timeline,
            "perp_close": perp_close,
            "spot_close": spot_close if has_spot else None,
            "perp_volume": perp_volume if has_perp_volume else None,
            "spot_volume": spot_volume if has_spot_volume else None,
            "perp_mark_close": perp_mark_close if has_mark else None,
            "perp_index_close": perp_index_close if has_index else None,
            "bid_close": bid_close if has_bidask else None,
            "ask_close": ask_close if has_bidask else None,
            "funding": funding,
        }
        return out

    def _fingerprint(self, csv_dir: Path, symbols: List[str], series: Dict[str, Any]) -> str:
        file_hashes = {}
        for sym in symbols:
            file_hashes[sym] = file_sha256(csv_dir / f"{sym}.csv")
        payload = {
            "provider": "local_csv_v1",
            "seed": self.seed,
            "symbols": symbols,
            "files": file_hashes,
        }
        return sha256_text(json.dumps(payload, sort_keys=True))
