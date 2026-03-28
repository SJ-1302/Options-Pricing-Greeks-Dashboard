"""
Black-Scholes Option Pricing Model.

Implements the analytical Black-Scholes-Merton formula for European option
pricing, along with a Newton-Raphson implied volatility solver.

All functions are vectorized with NumPy for efficient computation over
entire option chains.

References
----------
Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate
Liabilities." Journal of Political Economy, 81(3), 637-654.
"""

import numpy as np
from scipy.stats import norm
from typing import Union

Numeric = Union[float, np.ndarray]

def _d1(S: Numeric, K: Numeric, T: Numeric, r: Numeric, sigma: Numeric) -> Numeric:
    """
    Compute d₁ in the Black-Scholes formula.

    d₁ = [ln(S/K) + (r + σ²/2) · T] / (σ · √T)

    Parameters
    ----------
    S : float or np.ndarray — Spot price
    K : float or np.ndarray — Strike price
    T : float or np.ndarray — Time to expiry in years
    r : float — Risk-free rate (annualized)
    sigma : float or np.ndarray — Volatility (annualized)

    Returns
    -------
    float or np.ndarray
    """
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def _d2(S: Numeric, K: Numeric, T: Numeric, r: Numeric, sigma: Numeric) -> Numeric:
    """
    Compute d₂ in the Black-Scholes formula.

    d₂ = d₁ - σ · √T
    """
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def black_scholes_price(
    S: Numeric,
    K: Numeric,
    T: Numeric,
    r: float,
    sigma: Numeric,
    option_type: str = "CE"
) -> Numeric:
    """
    Compute the Black-Scholes theoretical price for European options.

    Call: C = S · N(d₁) - K · e^(-rT) · N(d₂)
    Put:  P = K · e^(-rT) · N(-d₂) - S · N(-d₁)

    Parameters
    ----------
    S : float or np.ndarray
        Current spot / underlying price.
    K : float or np.ndarray
        Strike price(s).
    T : float or np.ndarray
        Time to expiry in years. Must be > 0.
    r : float
        Annualized risk-free interest rate (e.g. 0.065 for 6.5%).
    sigma : float or np.ndarray
        Annualized volatility (e.g. 0.20 for 20%).
    option_type : str
        'CE' for Call, 'PE' for Put (case-insensitive).

    Returns
    -------
    float or np.ndarray
        Theoretical option price(s).

    Examples
    --------
    >>> black_scholes_price(100, 100, 1, 0.05, 0.20, 'CE')
    10.4506...
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)

    discount = K * np.exp(-r * T)

    if option_type.upper() == "CE":
        price = S * norm.cdf(d1) - discount * norm.cdf(d2)
    elif option_type.upper() == "PE":
        price = discount * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'CE' or 'PE', got '{option_type}'")

    return float(price) if price.ndim == 0 else price

def put_call_parity_check(
    call_price: float, put_price: float,
    S: float, K: float, T: float, r: float
) -> dict:
    """
    Verify Put-Call Parity: C - P = S - K · e^(-rT).

    Returns
    -------
    dict
        Keys: 'lhs' (C - P), 'rhs' (S - K·e^{-rT}), 'difference', 'holds' (bool).
    """
    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)
    diff = abs(lhs - rhs)

    return {
        "lhs": lhs,
        "rhs": rhs,
        "difference": diff,
        "holds": diff < 0.01 * S,  # Within 1% of spot
    }

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "CE",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    initial_guess: float = 0.3
) -> float:
    """
    Compute implied volatility using Newton-Raphson method.

    Finds σ such that BS(S, K, T, r, σ) = market_price.

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    S, K, T, r : float
        Standard BS parameters.
    option_type : str
        'CE' or 'PE'.
    max_iterations : int
        Maximum solver iterations (default: 100).
    tolerance : float
        Convergence threshold (default: 1e-6).
    initial_guess : float
        Starting volatility estimate (default: 0.30).

    Returns
    -------
    float
        Implied volatility. Returns NaN if solver fails to converge.
    """
    if market_price <= 0:
        return np.nan

    intrinsic = max(S - K, 0) if option_type.upper() == "CE" else max(K - S, 0)
    if market_price < intrinsic * np.exp(-r * T) * 0.99:
        return np.nan

    sigma = initial_guess

    for _ in range(max_iterations):
        try:
            bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
            d1 = _d1(S, K, T, r, sigma)
            vega = S * norm.pdf(d1) * np.sqrt(T)

            if abs(vega) < 1e-12:
                return np.nan

            price_diff = bs_price - market_price
            sigma = sigma - price_diff / vega

            sigma = max(sigma, 0.001)
            sigma = min(sigma, 5.0)

            if abs(price_diff) < tolerance:
                return sigma

        except (ValueError, FloatingPointError, ZeroDivisionError):
            return np.nan

    return np.nan  # Did not converge

def implied_volatility_vectorized(
    market_prices: np.ndarray,
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    option_type: str = "CE"
) -> np.ndarray:
    """
    Compute implied volatility for an array of options (vectorized loop).

    Parameters
    ----------
    market_prices : np.ndarray
        Array of observed market prices.
    S : float
        Current spot price.
    K : np.ndarray
        Array of strike prices.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    option_type : str
        'CE' or 'PE'.

    Returns
    -------
    np.ndarray
        Array of implied volatilities.
    """
    ivs = np.full_like(market_prices, np.nan, dtype=np.float64)

    for i in range(len(market_prices)):
        if market_prices[i] > 0 and K[i] > 0:
            ivs[i] = implied_volatility(
                market_prices[i], S, float(K[i]), T, r, option_type
            )

    return ivs
