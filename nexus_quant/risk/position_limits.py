from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PositionLimits:
    """
    Hard limits applied to a portfolio weight vector.

    Attributes
    ----------
    max_gross_leverage      : Maximum sum of |weights|.  A value of 2.0
                              means the portfolio can be up to 2x leveraged
                              on a gross basis.
    max_position_per_symbol : Maximum absolute weight for any single symbol.
    max_net_exposure        : Maximum absolute sum of all weights (directional
                              exposure cap).
    min_symbols_active      : Minimum number of symbols that must carry a
                              non-zero weight after enforcement.
    """
    max_gross_leverage:      float = 2.0
    max_position_per_symbol: float = 0.5
    max_net_exposure:        float = 0.2
    min_symbols_active:      int   = 2


# ---------------------------------------------------------------------------
# Enforcement logic
# ---------------------------------------------------------------------------

def enforce_position_limits(
    weights: Dict[str, float],
    limits: PositionLimits,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Clip a portfolio weight dict so it satisfies all PositionLimits.

    Enforcement order
    -----------------
    1. Per-symbol clip  – each |w| capped at max_position_per_symbol.
    2. Gross leverage   – if sum(|w|) > max_gross_leverage, scale all
                          weights down proportionally.
    3. Net exposure     – if |sum(w)| > max_net_exposure, shift the
                          smallest-magnitude weights toward zero until
                          the net cap is satisfied (proportional trim).

    Parameters
    ----------
    weights : Raw portfolio weights (symbol -> float).  May be modified;
              the original dict is not mutated.
    limits  : PositionLimits instance describing the constraints.

    Returns
    -------
    Tuple of:
        adjusted_weights – New dict with all limits satisfied.
        violations_fixed – List of human-readable strings describing
                           each constraint that was violated and corrected.
    """
    w: Dict[str, float] = dict(weights)    # defensive copy
    violations: List[str] = []

    # ------------------------------------------------------------------ 1
    # Per-symbol position cap
    # ------------------------------------------------------------------ 1
    cap = limits.max_position_per_symbol
    for sym in list(w.keys()):
        if abs(w[sym]) > cap:
            violations.append(
                f"Symbol '{sym}': |weight| {abs(w[sym]):.6f} > "
                f"max_position_per_symbol {cap:.6f}; clipped."
            )
            w[sym] = cap * (1.0 if w[sym] >= 0.0 else -1.0)

    # ------------------------------------------------------------------ 2
    # Gross leverage cap  (sum of |weights|)
    # ------------------------------------------------------------------ 2
    gross = sum(abs(v) for v in w.values())
    if gross > limits.max_gross_leverage and gross > 0.0:
        scale = limits.max_gross_leverage / gross
        violations.append(
            f"Gross leverage {gross:.6f} > max {limits.max_gross_leverage:.6f}; "
            f"scaling all weights by {scale:.6f}."
        )
        w = {sym: val * scale for sym, val in w.items()}

    # ------------------------------------------------------------------ 3
    # Net exposure cap  (|sum of weights|)
    # ------------------------------------------------------------------ 3
    net = sum(w.values())
    if abs(net) > limits.max_net_exposure:
        excess = abs(net) - limits.max_net_exposure
        violations.append(
            f"Net exposure {net:.6f} (|net|={abs(net):.6f}) > "
            f"max_net_exposure {limits.max_net_exposure:.6f}; trimming."
        )
        # Trim proportionally in the direction of the excess
        sign_net = 1.0 if net > 0.0 else -1.0
        # Identify symbols that contribute in the direction of the excess
        # and trim them proportionally until excess is absorbed.
        contributors = {s: v for s, v in w.items() if v * sign_net > 0.0}
        total_contrib = sum(abs(v) for v in contributors.values())
        if total_contrib > 0.0:
            for sym in contributors:
                fraction = abs(w[sym]) / total_contrib
                trim     = fraction * excess
                w[sym]  -= sign_net * trim

    # ------------------------------------------------------------------ 4
    # Minimum active symbols check (informational; we do not add positions)
    # ------------------------------------------------------------------ 4
    active = sum(1 for v in w.values() if v != 0.0)
    if active < limits.min_symbols_active:
        violations.append(
            f"Only {active} active symbol(s) after enforcement; "
            f"min_symbols_active={limits.min_symbols_active} is not met. "
            f"No automatic fix applied – caller must review."
        )

    return w, violations


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def position_limits_from_config(risk_cfg: Dict[str, Any]) -> PositionLimits:
    """
    Build a PositionLimits instance from a configuration dictionary.

    Supported keys (all optional; defaults from PositionLimits are used
    for missing keys):
        max_gross_leverage      (float)
        max_position_per_symbol (float)
        max_net_exposure        (float)
        min_symbols_active      (int)

    Parameters
    ----------
    risk_cfg : Configuration dict, e.g. from a TOML / JSON config file.

    Returns
    -------
    PositionLimits
    """
    defaults = PositionLimits()

    return PositionLimits(
        max_gross_leverage=float(
            risk_cfg.get("max_gross_leverage", defaults.max_gross_leverage)
        ),
        max_position_per_symbol=float(
            risk_cfg.get("max_position_per_symbol", defaults.max_position_per_symbol)
        ),
        max_net_exposure=float(
            risk_cfg.get("max_net_exposure", defaults.max_net_exposure)
        ),
        min_symbols_active=int(
            risk_cfg.get("min_symbols_active", defaults.min_symbols_active)
        ),
    )
