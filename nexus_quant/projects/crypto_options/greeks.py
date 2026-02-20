"""
Black-Scholes Greeks Calculator
stdlib-only (math module, no numpy/scipy)

Functions:
    bs_price(S, K, T, r, sigma, option_type) -> float
    bs_delta(S, K, T, r, sigma, option_type) -> float
    bs_gamma(S, K, T, r, sigma) -> float
    bs_vega(S, K, T, r, sigma) -> float
    bs_theta(S, K, T, r, sigma, option_type) -> float
    implied_vol(market_price, S, K, T, r, option_type) -> float

Uses math.erf for normal CDF (Python 3.2+).
"""
from __future__ import annotations

import math
from typing import Optional


# ── Normal distribution helpers ──────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ── d1, d2 helper ────────────────────────────────────────────────────────────

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    """Compute d1 and d2 for Black-Scholes.

    Args:
        S: underlying spot price
        K: strike price
        T: time to expiry in years (e.g., 30/365)
        r: risk-free rate (annualized, e.g., 0.05)
        sigma: implied volatility (annualized, e.g., 0.80)

    Returns:
        (d1, d2) tuple
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError(f"Invalid inputs: S={S}, K={K}, T={T}, sigma={sigma}")
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


# ── Option price ─────────────────────────────────────────────────────────────

def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes option price.

    Args:
        S: underlying price
        K: strike price
        T: time to expiry in years
        r: risk-free rate (annualized)
        sigma: implied vol (annualized)
        option_type: "call" or "put"

    Returns:
        Option price (same units as S)
    """
    if T <= 1e-10:
        # At expiry: intrinsic value only
        if option_type.lower() == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)

    if option_type.lower() == "call":
        return S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    elif option_type.lower() == "put":
        return K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


# ── Greeks ────────────────────────────────────────────────────────────────────

def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Delta: dV/dS. Range: [0,1] for call, [-1,0] for put."""
    if T <= 1e-10:
        if option_type.lower() == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def bs_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Gamma: d²V/dS². Same for call and put."""
    if T <= 1e-10:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Vega: dV/d(sigma). Same for call and put.
    Returns change per 1-unit (100%) change in vol.
    Divide by 100 to get per 1% change.
    """
    if T <= 1e-10:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T)


def bs_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Theta: dV/dt (time decay per year). Usually negative.
    Divide by 365 to get per-day theta.
    """
    if T <= 1e-10:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)
    term1 = -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if option_type.lower() == "call":
        term2 = -r * K * disc * _norm_cdf(d2)
        return term1 + term2
    elif option_type.lower() == "put":
        term2 = r * K * disc * _norm_cdf(-d2)
        return term1 + term2
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def bs_rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Rho: dV/dr. Change per 1-unit change in rate."""
    if T <= 1e-10:
        return 0.0
    _, d2 = _d1_d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)
    if option_type.lower() == "call":
        return K * T * disc * _norm_cdf(d2)
    return -K * T * disc * _norm_cdf(-d2)


# ── All Greeks bundle ─────────────────────────────────────────────────────────

def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    """Compute all Greeks at once.

    Returns:
        dict with keys: price, delta, gamma, vega, theta, rho
    """
    price = bs_price(S, K, T, r, sigma, option_type)
    delta = bs_delta(S, K, T, r, sigma, option_type)
    gamma = bs_gamma(S, K, T, r, sigma)
    vega = bs_vega(S, K, T, r, sigma)
    theta = bs_theta(S, K, T, r, sigma, option_type)
    rho = bs_rho(S, K, T, r, sigma, option_type)
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


# ── Implied volatility solver (Newton-Raphson) ────────────────────────────────

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    initial_guess: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Optional[float]:
    """Newton-Raphson solver for implied volatility.

    Args:
        market_price: observed option market price
        S: underlying spot price
        K: strike
        T: time to expiry in years
        r: risk-free rate
        option_type: "call" or "put"
        initial_guess: starting sigma (default 0.5 = 50% IV)
        max_iter: max iterations
        tol: convergence tolerance

    Returns:
        Implied vol (annualized) or None if not converged
    """
    if T <= 1e-10 or S <= 0 or K <= 0:
        return None

    # Check intrinsic value bounds
    disc = math.exp(-r * T)
    if option_type.lower() == "call":
        intrinsic = max(S - K * disc, 0.0)
    else:
        intrinsic = max(K * disc - S, 0.0)

    if market_price < intrinsic - 1e-6:
        return None  # Price below intrinsic — impossible

    sigma = max(initial_guess, 0.01)

    for _ in range(max_iter):
        try:
            price = bs_price(S, K, T, r, sigma, option_type)
            vega = bs_vega(S, K, T, r, sigma)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None

        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        if abs(vega) < 1e-12:
            # Vega too small — try bisection fallback
            break

        sigma = sigma - diff / vega
        # Clamp to reasonable range
        sigma = max(0.001, min(sigma, 20.0))

    # Bisection fallback for robustness
    return _implied_vol_bisection(market_price, S, K, T, r, option_type)


def _implied_vol_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    lo: float = 0.001,
    hi: float = 20.0,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> Optional[float]:
    """Bisection fallback for implied vol."""
    try:
        f_lo = bs_price(S, K, T, r, lo, option_type) - market_price
        f_hi = bs_price(S, K, T, r, hi, option_type) - market_price
    except Exception:
        return None

    if f_lo * f_hi > 0:
        return None  # Root not bracketed

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        try:
            f_mid = bs_price(S, K, T, r, mid, option_type) - market_price
        except Exception:
            return None

        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return 0.5 * (lo + hi)


# ── Delta-to-strike conversion ────────────────────────────────────────────────

def strike_from_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float,
    option_type: str = "put",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """Find strike K such that delta(S, K, T, r, sigma) == target_delta.

    Useful for finding 25-delta strikes.
    target_delta: e.g., -0.25 for 25-delta put, 0.25 for 25-delta call.
    """
    if T <= 0 or sigma <= 0:
        return None

    # Binary search on K
    lo, hi = S * 0.01, S * 10.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        try:
            d = bs_delta(S, mid, T, r, sigma, option_type)
        except Exception:
            return None

        diff = d - target_delta
        if abs(diff) < tol:
            return mid

        # delta decreases as K increases (for call and put)
        if option_type.lower() == "call":
            if d > target_delta:
                lo = mid
            else:
                hi = mid
        else:
            if d < target_delta:
                lo = mid
            else:
                hi = mid

        if hi - lo < 0.01:
            return 0.5 * (lo + hi)

    return 0.5 * (lo + hi)


# ── Self-test ─────────────────────────────────────────────────────────────────

def _selftest():
    """Quick sanity check."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")

    # Put-call parity: C - P = S - K*exp(-rT)
    parity = S - K * math.exp(-r * T)
    assert abs((call - put) - parity) < 1e-8, f"Put-call parity failed: {call-put} != {parity}"

    # Delta for ATM call should be ~0.5-0.6 range
    delta_c = bs_delta(S, K, T, r, sigma, "call")
    assert 0.5 < delta_c < 0.7, f"ATM call delta out of range: {delta_c}"

    # Implied vol round-trip
    iv = implied_vol(call, S, K, T, r, "call")
    assert iv is not None and abs(iv - sigma) < 1e-5, f"IV round-trip failed: {iv} != {sigma}"

    print(f"[greeks.py] Self-test passed. ATM call price={call:.4f}, delta={delta_c:.4f}, IV={iv:.4f}")


if __name__ == "__main__":
    _selftest()
