"""
Black-Scholes pricing engine, Greeks calculator, and IV/HV utilities.

All formulas follow the standard European option model:
  C = S·N(d1) - K·e^(-r·t)·N(d2)
  P = K·e^(-r·t)·N(-d2) - S·N(-d1)

where:
  d1 = [ln(S/K) + (r + σ²/2)·t] / (σ·√t)
  d2 = d1 - σ·√t
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

OptionType = Literal["call", "put"]


@dataclass
class OptionContract:
    """Lightweight container for a single option contract's key parameters."""

    underlying_price: float
    strike: float
    time_to_expiry: float  # years
    risk_free_rate: float  # annualised decimal (e.g. 0.053)
    volatility: float  # annualised decimal (e.g. 0.30 = 30%)
    option_type: OptionType = "call"


@dataclass
class Greeks:
    """Calculated Greeks for a single option contract."""

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0  # per calendar day
    vega: float = 0.0   # per 1-point move in IV (not per 1%)
    rho: float = 0.0
    iv: float = 0.0     # same as input volatility (echoed for convenience)
    theoretical_price: float = 0.0


# ---------------------------------------------------------------------------
# Core maths helpers
# ---------------------------------------------------------------------------

def _d1(S: float, K: float, r: float, sigma: float, t: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))


def _d2(d1_val: float, sigma: float, t: float) -> float:
    return d1_val - sigma * math.sqrt(t)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def black_scholes_price(contract: OptionContract) -> float:
    """Return the Black-Scholes theoretical price of a European option."""
    S = contract.underlying_price
    K = contract.strike
    r = contract.risk_free_rate
    sigma = contract.volatility
    t = contract.time_to_expiry

    if t <= 0 or sigma <= 0:
        # At or past expiry: intrinsic value only
        intrinsic = max(S - K, 0) if contract.option_type == "call" else max(K - S, 0)
        return float(intrinsic)

    d1 = _d1(S, K, r, sigma, t)
    d2 = _d2(d1, sigma, t)

    if contract.option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return max(float(price), 0.0)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def calculate_greeks(contract: OptionContract) -> Greeks:
    """Calculate all first-order Greeks plus Rho for the given contract."""
    S = contract.underlying_price
    K = contract.strike
    r = contract.risk_free_rate
    sigma = contract.volatility
    t = contract.time_to_expiry
    opt = contract.option_type

    price = black_scholes_price(contract)

    if t <= 0 or sigma <= 0:
        return Greeks(
            delta=1.0 if (opt == "call" and S > K) else (-1.0 if (opt == "put" and S < K) else 0.0),
            theoretical_price=price,
            iv=sigma,
        )

    d1 = _d1(S, K, r, sigma, t)
    d2 = _d2(d1, sigma, t)
    nd1 = norm.pdf(d1)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    sqrt_t = math.sqrt(t)

    # Delta
    if opt == "call":
        delta = float(Nd1)
    else:
        delta = float(Nd1 - 1)

    # Gamma (same for calls and puts)
    gamma = float(nd1 / (S * sigma * sqrt_t))

    # Theta (per calendar day = divide by 365)
    if opt == "call":
        theta = float(
            (-S * nd1 * sigma / (2 * sqrt_t) - r * K * math.exp(-r * t) * Nd2) / 365
        )
    else:
        theta = float(
            (-S * nd1 * sigma / (2 * sqrt_t) + r * K * math.exp(-r * t) * norm.cdf(-d2)) / 365
        )

    # Vega (per 1-point move in sigma, i.e. for Δσ = 1.0)
    vega = float(S * sqrt_t * nd1)

    # Rho
    if opt == "call":
        rho = float(K * t * math.exp(-r * t) * Nd2)
    else:
        rho = float(-K * t * math.exp(-r * t) * norm.cdf(-d2))

    return Greeks(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
        iv=sigma,
        theoretical_price=price,
    )


# ---------------------------------------------------------------------------
# Implied Volatility solver (Newton-Raphson with bisection fallback)
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    contract: OptionContract,
    tolerance: float = 1e-6,
    max_iter: int = 500,
) -> float | None:
    """
    Solve for the implied volatility that makes the BS price equal to *market_price*.

    Returns None if no solution is found within constraints.
    """
    S = contract.underlying_price
    K = contract.strike
    t = contract.time_to_expiry

    if t <= 0:
        return None

    # Intrinsic value sanity check
    if contract.option_type == "call":
        intrinsic = max(S - K * math.exp(-contract.risk_free_rate * t), 0)
    else:
        intrinsic = max(K * math.exp(-contract.risk_free_rate * t) - S, 0)

    if market_price < intrinsic - 0.01:
        return None

    # Bracketing bisection to find a robust starting point
    lo, hi = 1e-4, 10.0

    def price_at_sigma(sigma: float) -> float:
        c = OptionContract(
            underlying_price=S,
            strike=K,
            time_to_expiry=t,
            risk_free_rate=contract.risk_free_rate,
            volatility=sigma,
            option_type=contract.option_type,
        )
        return black_scholes_price(c)

    if price_at_sigma(lo) > market_price or price_at_sigma(hi) < market_price:
        return None

    sigma = 0.3  # initial guess

    for _ in range(max_iter):
        c_tmp = OptionContract(
            underlying_price=S,
            strike=K,
            time_to_expiry=t,
            risk_free_rate=contract.risk_free_rate,
            volatility=sigma,
            option_type=contract.option_type,
        )
        g = calculate_greeks(c_tmp)
        price_diff = g.theoretical_price - market_price
        vega = g.vega

        if abs(price_diff) < tolerance:
            return sigma

        if vega < 1e-10:
            # Fall back to bisection
            mid = (lo + hi) / 2.0
            if price_at_sigma(mid) < market_price:
                lo = mid
            else:
                hi = mid
            sigma = (lo + hi) / 2.0
        else:
            sigma -= price_diff / vega
            sigma = max(lo, min(hi, sigma))

    return sigma if abs(price_at_sigma(sigma) - market_price) < 0.01 else None


# ---------------------------------------------------------------------------
# Historical Volatility
# ---------------------------------------------------------------------------

def historical_volatility(close_prices: list[float] | np.ndarray, window: int = 30) -> float:
    """
    Compute annualised close-to-close historical volatility using log returns.

    Parameters
    ----------
    close_prices:
        Ordered list of closing prices (oldest first).
    window:
        Number of trading days to include in the calculation.

    Returns
    -------
    Annualised HV as a decimal (e.g. 0.25 = 25%).
    """
    prices = np.asarray(close_prices, dtype=float)
    if len(prices) < 2:
        return 0.0
    prices = prices[-window - 1:]  # keep only the required tail
    log_returns = np.diff(np.log(prices))
    if len(log_returns) == 0:
        return 0.0
    return float(np.std(log_returns, ddof=1) * math.sqrt(252))


# ---------------------------------------------------------------------------
# IV Rank and IV Percentile
# ---------------------------------------------------------------------------

def iv_rank(current_iv: float, iv_history: list[float] | np.ndarray) -> float:
    """
    IV Rank: where current IV sits between the 52-week low and high.

    Returns a value in [0, 1]:  0 = at 52-week low, 1 = at 52-week high.
    """
    arr = np.asarray(iv_history, dtype=float)
    iv_min = arr.min()
    iv_max = arr.max()
    if iv_max == iv_min:
        return 0.5
    return float((current_iv - iv_min) / (iv_max - iv_min))


def iv_percentile(current_iv: float, iv_history: list[float] | np.ndarray) -> float:
    """
    IV Percentile: fraction of past IV readings below current IV.

    Returns a value in [0, 1].
    """
    arr = np.asarray(iv_history, dtype=float)
    return float(np.mean(arr < current_iv))


# ---------------------------------------------------------------------------
# Probability of Profit helpers
# ---------------------------------------------------------------------------

def pop_long_call(delta: float) -> float:
    """
    Approximate probability of profit for a long call.

    POP ≈ 1 - |delta| (the option finishing ITM at expiry is roughly |delta|).
    This is a well-known simplification; for long options the trader profits
    only if the move is large enough to overcome extrinsic value, so the true
    PoP is lower than delta—but delta serves as an upper bound.
    """
    return float(max(0.0, min(1.0, 1.0 - abs(delta))))


def pop_long_put(delta: float) -> float:
    """Approximate probability of profit for a long put."""
    return float(max(0.0, min(1.0, abs(delta))))


# ---------------------------------------------------------------------------
# P&L payoff diagram data
# ---------------------------------------------------------------------------

def call_payoff_at_expiry(
    strike: float,
    premium_paid: float,
    underlying_prices: list[float] | np.ndarray,
) -> np.ndarray:
    """Return per-share P&L for a long call at expiry across a price range."""
    prices = np.asarray(underlying_prices, dtype=float)
    return np.maximum(prices - strike, 0) - premium_paid


def put_payoff_at_expiry(
    strike: float,
    premium_paid: float,
    underlying_prices: list[float] | np.ndarray,
) -> np.ndarray:
    """Return per-share P&L for a long put at expiry across a price range."""
    prices = np.asarray(underlying_prices, dtype=float)
    return np.maximum(strike - prices, 0) - premium_paid
