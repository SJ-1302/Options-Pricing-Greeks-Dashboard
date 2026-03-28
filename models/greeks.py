"""
Option Greeks Calculator.

Computes all five major Greeks (Delta, Gamma, Theta, Vega, Rho) using
analytical Black-Scholes closed-form solutions.

Scaling Conventions
-------------------
- Delta:  per ₹1 move in underlying (raw, -1 to +1)
- Gamma:  per ₹1 move in underlying
- Theta:  **per calendar day** (annual theta ÷ 365)
- Vega:   **per 1% change** in volatility (annual vega ÷ 100)
- Rho:    per 1% change in interest rate (annual rho ÷ 100)

All functions are vectorized with NumPy.
"""

import numpy as np
from scipy.stats import norm
from typing import Union, Dict
import pandas as pd

Numeric = Union[float, np.ndarray]

def _d1(S: Numeric, K: Numeric, T: Numeric, r: float, sigma: Numeric) -> Numeric:
    """d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)"""
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def _d2(d1_val: Numeric, sigma: Numeric, T: Numeric) -> Numeric:
    """d₂ = d₁ - σ·√T"""
    return d1_val - sigma * np.sqrt(T)

def delta(S: Numeric, K: Numeric, T: Numeric, r: float,
          sigma: Numeric, option_type: str = "CE") -> Numeric:
    """
    Option Delta — rate of change of option price w.r.t. underlying.

    Call Delta = N(d₁)        ∈ [0, 1]
    Put  Delta = N(d₁) - 1    ∈ [-1, 0]

    Returns
    -------
    float or np.ndarray
        Delta value(s).
    """
    d1 = _d1(S, K, T, r, sigma)

    if option_type.upper() == "CE":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0

def gamma(S: Numeric, K: Numeric, T: Numeric, r: float,
          sigma: Numeric) -> Numeric:
    """
    Option Gamma — rate of change of delta w.r.t. underlying.

    Gamma = n(d₁) / (S · σ · √T)

    Same for both calls and puts. Always positive.

    Returns
    -------
    float or np.ndarray
        Gamma value(s).
    """
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def theta(S: Numeric, K: Numeric, T: Numeric, r: float,
          sigma: Numeric, option_type: str = "CE") -> Numeric:
    """
    Option Theta — time decay, **scaled to per calendar day**.

    Call Θ = [-S·n(d₁)·σ/(2√T) - r·K·e^(-rT)·N(d₂)]  / 365
    Put  Θ = [-S·n(d₁)·σ/(2√T) + r·K·e^(-rT)·N(-d₂)] / 365

    Returns
    -------
    float or np.ndarray
        Theta per day (typically negative — options lose value over time).
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)

    time_decay = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))

    discount = K * np.exp(-r * T)

    if option_type.upper() == "CE":
        annual_theta = time_decay - r * discount * norm.cdf(d2)
    else:
        annual_theta = time_decay + r * discount * norm.cdf(-d2)

    return annual_theta / 365.0

def vega(S: Numeric, K: Numeric, T: Numeric, r: float,
         sigma: Numeric) -> Numeric:
    """
    Option Vega — sensitivity to volatility, **scaled per 1% change**.

    Vega = S · n(d₁) · √T  / 100

    Same for both calls and puts. Always positive.

    Returns
    -------
    float or np.ndarray
        Vega per 1% volatility change.
    """
    d1 = _d1(S, K, T, r, sigma)
    annual_vega = S * norm.pdf(d1) * np.sqrt(T)

    return annual_vega / 100.0

def rho(S: Numeric, K: Numeric, T: Numeric, r: float,
        sigma: Numeric, option_type: str = "CE") -> Numeric:
    """
    Option Rho — sensitivity to interest rate, **scaled per 1% change**.

    Call ρ = K · T · e^(-rT) · N(d₂)  / 100
    Put  ρ = -K · T · e^(-rT) · N(-d₂) / 100

    Returns
    -------
    float or np.ndarray
        Rho per 1% rate change.
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)

    discount = K * T * np.exp(-r * T)

    if option_type.upper() == "CE":
        annual_rho = discount * norm.cdf(d2)
    else:
        annual_rho = -discount * norm.cdf(-d2)

    return annual_rho / 100.0

def compute_greeks(
    S: float, K: float, T: float, r: float,
    sigma: float, option_type: str = "CE"
) -> Dict[str, float]:
    """
    Compute all five Greeks for a single option.

    Parameters
    ----------
    S     : Spot price
    K     : Strike price
    T     : Time to expiry (years)
    r     : Risk-free rate
    sigma : Volatility
    option_type : 'CE' or 'PE'

    Returns
    -------
    dict
        {'delta': ..., 'gamma': ..., 'theta': ..., 'vega': ..., 'rho': ...}
        Theta is per day, Vega and Rho are per 1% change.
    """
    return {
        "delta": float(delta(S, K, T, r, sigma, option_type)),
        "gamma": float(gamma(S, K, T, r, sigma)),
        "theta": float(theta(S, K, T, r, sigma, option_type)),
        "vega": float(vega(S, K, T, r, sigma)),
        "rho": float(rho(S, K, T, r, sigma, option_type)),
    }

def compute_all_greeks_for_chain(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    ce_sigmas: np.ndarray,
    pe_sigmas: np.ndarray
) -> pd.DataFrame:
    """
    Compute all Greeks for an entire option chain (vectorized).

    Parameters
    ----------
    S : float
        Current spot price.
    strikes : np.ndarray
        Array of strike prices.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    ce_sigmas : np.ndarray
        Implied volatilities for Call options.
    pe_sigmas : np.ndarray
        Implied volatilities for Put options.

    Returns
    -------
    pd.DataFrame
        Columns: strike, ce_delta, ce_gamma, ce_theta, ce_vega, ce_rho,
                 pe_delta, pe_gamma, pe_theta, pe_vega, pe_rho
    """
    K = np.asarray(strikes, dtype=np.float64)
    ce_iv = np.asarray(ce_sigmas, dtype=np.float64)
    pe_iv = np.asarray(pe_sigmas, dtype=np.float64)

    default_iv = 0.20
    ce_iv_safe = np.where((ce_iv > 0.001) & np.isfinite(ce_iv), ce_iv, default_iv)
    pe_iv_safe = np.where((pe_iv > 0.001) & np.isfinite(pe_iv), pe_iv, default_iv)

    result = pd.DataFrame({"strike": K})

    result["ce_delta"] = delta(S, K, T, r, ce_iv_safe, "CE")
    result["ce_gamma"] = gamma(S, K, T, r, ce_iv_safe)
    result["ce_theta"] = theta(S, K, T, r, ce_iv_safe, "CE")
    result["ce_vega"] = vega(S, K, T, r, ce_iv_safe)
    result["ce_rho"] = rho(S, K, T, r, ce_iv_safe, "CE")

    result["pe_delta"] = delta(S, K, T, r, pe_iv_safe, "PE")
    result["pe_gamma"] = gamma(S, K, T, r, pe_iv_safe)
    result["pe_theta"] = theta(S, K, T, r, pe_iv_safe, "PE")
    result["pe_vega"] = vega(S, K, T, r, pe_iv_safe)
    result["pe_rho"] = rho(S, K, T, r, pe_iv_safe, "PE")

    return result
