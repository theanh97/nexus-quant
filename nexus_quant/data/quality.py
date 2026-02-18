from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .schema import MarketDataset


def validate_dataset(dataset: MarketDataset, expected_step_seconds: Optional[int] = None) -> Dict[str, Any]:
    """
    Data quality gate focused on bias prevention and basic sanity.

    This is intentionally strict. If a dataset fails hard checks, backtests are not trustworthy.
    """
    timeline = dataset.timeline
    issues: List[Dict[str, Any]] = []

    def add(level: str, code: str, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        issues.append({"level": level, "code": code, "message": msg, "extra": extra or {}})

    # Timeline checks
    if len(timeline) < 2:
        add("error", "timeline_too_short", "Timeline must contain at least 2 timestamps.")
    if any(t2 <= t1 for t1, t2 in zip(timeline, timeline[1:])):
        add("error", "timeline_not_strictly_increasing", "Timeline must be strictly increasing (sorted, no duplicates).")

    timeline_set = set(timeline)

    if expected_step_seconds is not None and expected_step_seconds > 0 and len(timeline) >= 3:
        steps = [timeline[i] - timeline[i - 1] for i in range(1, len(timeline))]
        # Allow small irregularities but flag if too many.
        bad = sum(1 for s in steps if s != expected_step_seconds)
        frac_bad = bad / float(len(steps)) if steps else 0.0
        if frac_bad > 0.01:
            add(
                "warn",
                "timeline_step_irregular",
                "Timeline step is irregular for more than 1% of bars.",
                {"expected_step_seconds": expected_step_seconds, "fraction_bad": round(frac_bad, 6)},
            )

    # Series length checks
    n = len(timeline)
    for sym in dataset.symbols:
        closes = dataset.perp_close.get(sym)
        if closes is None or len(closes) != n:
            add("error", "perp_close_length_mismatch", f"perp_close[{sym}] length must match timeline.", {"len": len(closes or [])})
            continue
        if any((c is None) or (float(c) <= 0.0) for c in closes):
            add("error", "perp_close_non_positive", f"perp_close[{sym}] contains non-positive values.")

        if dataset.spot_close is not None:
            spots = dataset.spot_close.get(sym)
            if spots is None or len(spots) != n:
                add("warn", "spot_close_length_mismatch", f"spot_close[{sym}] length mismatch; basis may be unreliable.")
            else:
                # Spot may be missing (0), so only flag negative values.
                if any((x is None) or (float(x) < 0.0) for x in spots):
                    add("error", "spot_close_negative", f"spot_close[{sym}] contains negative values.")

        # Optional extended fields (warn on length mismatch, error on negative values).
        def _check_opt_series(
            series: Optional[Dict[str, List[float]]],
            *,
            code_len: str,
            code_neg: str,
            allow_zero: bool,
        ) -> None:
            if series is None:
                return
            xs = series.get(sym)
            if xs is None or len(xs) != n:
                add("warn", code_len, f"{code_len}[{sym}] length mismatch.", {"len": len(xs or [])})
                return
            if allow_zero:
                bad = any((x is None) or (float(x) < 0.0) for x in xs)
            else:
                bad = any((x is None) or (float(x) <= 0.0) for x in xs)
            if bad:
                add("error", code_neg, f"{code_neg}[{sym}] contains invalid (negative/zero) values.")

        _check_opt_series(dataset.perp_mark_close, code_len="perp_mark_length_mismatch", code_neg="perp_mark_invalid", allow_zero=True)
        _check_opt_series(dataset.perp_index_close, code_len="perp_index_length_mismatch", code_neg="perp_index_invalid", allow_zero=True)
        _check_opt_series(dataset.perp_volume, code_len="perp_volume_length_mismatch", code_neg="perp_volume_negative", allow_zero=True)
        _check_opt_series(dataset.spot_volume, code_len="spot_volume_length_mismatch", code_neg="spot_volume_negative", allow_zero=True)

        # Bid/ask sanity (if present).
        if dataset.bid_close is not None or dataset.ask_close is not None:
            bid = (dataset.bid_close or {}).get(sym)
            ask = (dataset.ask_close or {}).get(sym)
            if bid is None or ask is None or len(bid) != n or len(ask) != n:
                add("warn", "bidask_length_mismatch", f"bid/ask length mismatch for {sym}.")
            else:
                bad = 0
                for b, a in zip(bid, ask):
                    bb = float(b or 0.0)
                    aa = float(a or 0.0)
                    if bb < 0.0 or aa < 0.0:
                        bad += 1
                    # 0 indicates missing; only check ordering when both are non-zero.
                    if bb > 0.0 and aa > 0.0 and bb > aa:
                        bad += 1
                if bad > 0:
                    add("error", "bid_gt_ask", f"bid_close[{sym}] > ask_close[{sym}] (or negative) on {bad} bars.")

        # Funding times should be on timeline (event at bar close).
        f = dataset.funding.get(sym, {})
        if f:
            bad_ts = [t for t in f.keys() if t not in timeline_set]
            if bad_ts:
                add("warn", "funding_off_timeline", f"funding[{sym}] has events not on timeline; they will be ignored.", {"count": len(bad_ts)})

    ok = not any(x["level"] == "error" for x in issues)
    return {
        "ok": ok,
        "provider": dataset.provider,
        "fingerprint": dataset.fingerprint,
        "symbols": list(dataset.symbols),
        "bars": n,
        "expected_step_seconds": expected_step_seconds,
        "issues": issues,
    }
