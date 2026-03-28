"""
Option Chain Data Processing & Filtering.

Provides functions to clean, validate, and filter raw option chain
data from the NSE fetcher or Kotak Neo API. Also computes derived
columns like mispricing metrics and moneyness classification.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from datetime import date, datetime

from models.black_scholes import black_scholes_price, implied_volatility_vectorized
from utils.helpers import time_to_expiry, SUPPORTED_INSTRUMENTS

def process_option_chain(
    df: pd.DataFrame,
    spot_price: float,
    risk_free_rate: float = 0.065
) -> pd.DataFrame:
    """
    Clean and enrich raw option chain data.

    Adds computed columns:
        - moneyness: 'ITM', 'ATM', 'OTM' for both CE and PE
        - bs_ce_price, bs_pe_price: theoretical Black-Scholes prices
        - ce_mispricing, pe_mispricing: market − BS price (positive = overpriced)
        - ce_mispricing_pct, pe_mispricing_pct: mispricing as % of BS price

    Parameters
    ----------
    df : pd.DataFrame
        Raw option chain with standard columns from nse_fetcher.
    spot_price : float
        Current spot/underlying price.
    risk_free_rate : float
        Risk-free rate for BS calculations.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with all original and computed columns.
    """
    df = df.copy()

    numeric_cols = [
        "strike", "ce_ltp", "ce_bid", "ce_ask", "ce_oi", "ce_volume", "ce_iv",
        "pe_ltp", "pe_bid", "pe_ask", "pe_oi", "pe_volume", "pe_iv",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    atm_threshold = 0.005  # Within 0.5% of spot = ATM

    df["moneyness_ratio"] = df["strike"] / spot_price

    df["ce_moneyness"] = np.where(
        np.abs(df["moneyness_ratio"] - 1) < atm_threshold, "ATM",
        np.where(df["strike"] < spot_price, "ITM", "OTM")
    )
    df["pe_moneyness"] = np.where(
        np.abs(df["moneyness_ratio"] - 1) < atm_threshold, "ATM",
        np.where(df["strike"] > spot_price, "ITM", "OTM")
    )

    for expiry_val in df["expiry"].unique():
        mask = df["expiry"] == expiry_val
        T = time_to_expiry(expiry_val)

        strikes = df.loc[mask, "strike"].values
        ce_ivs = df.loc[mask, "ce_iv"].values
        pe_ivs = df.loc[mask, "pe_iv"].values

        ce_iv_safe = np.where(ce_ivs > 0.01, ce_ivs, 0.15)
        pe_iv_safe = np.where(pe_ivs > 0.01, pe_ivs, 0.15)

        bs_ce = black_scholes_price(spot_price, strikes, T, risk_free_rate, ce_iv_safe, "CE")
        bs_pe = black_scholes_price(spot_price, strikes, T, risk_free_rate, pe_iv_safe, "PE")

        df.loc[mask, "bs_ce_price"] = bs_ce
        df.loc[mask, "bs_pe_price"] = bs_pe
        df.loc[mask, "T"] = T

    df["ce_mispricing"] = df["ce_ltp"] - df["bs_ce_price"]
    df["pe_mispricing"] = df["pe_ltp"] - df["bs_pe_price"]

    df["ce_mispricing_pct"] = np.where(
        df["bs_ce_price"] > 0.1,
        (df["ce_mispricing"] / df["bs_ce_price"]) * 100,
        0
    )
    df["pe_mispricing_pct"] = np.where(
        df["bs_pe_price"] > 0.1,
        (df["pe_mispricing"] / df["bs_pe_price"]) * 100,
        0
    )

    return df

def filter_by_expiry(df: pd.DataFrame, expiry_date: str) -> pd.DataFrame:
    """
    Filter option chain by a specific expiry date.

    Parameters
    ----------
    df : pd.DataFrame
        Full option chain data.
    expiry_date : str
        Expiry date string (must match format in the data).

    Returns
    -------
    pd.DataFrame
        Filtered rows for the given expiry.
    """
    return df[df["expiry"] == expiry_date].copy()

def filter_by_moneyness(
    df: pd.DataFrame,
    spot_price: float,
    range_pct: float = 10.0
) -> pd.DataFrame:
    """
    Filter strikes within a percentage range of the spot price.

    Parameters
    ----------
    df : pd.DataFrame
        Option chain data.
    spot_price : float
        Current spot price.
    range_pct : float
        Percentage range around spot (default: 10% → strikes from 90% to 110% of spot).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    lower = spot_price * (1 - range_pct / 100)
    upper = spot_price * (1 + range_pct / 100)
    return df[(df["strike"] >= lower) & (df["strike"] <= upper)].copy()

def filter_by_strike_range(
    df: pd.DataFrame,
    min_strike: float,
    max_strike: float
) -> pd.DataFrame:
    """Filter by explicit min/max strike price range."""
    return df[(df["strike"] >= min_strike) & (df["strike"] <= max_strike)].copy()

def get_available_expiries(df: pd.DataFrame) -> List[str]:
    """
    Get sorted list of unique expiry dates.

    Returns
    -------
    list[str]
        Unique expiry dates in chronological order.
    """
    expiries = df["expiry"].unique().tolist()

    def parse_expiry(exp_str):
        for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%B-%Y"):
            try:
                return datetime.strptime(exp_str, fmt)
            except ValueError:
                continue
        return datetime.max

    return sorted(expiries, key=parse_expiry)

def get_atm_strike(df: pd.DataFrame, spot_price: float) -> float:
    """
    Find the ATM (at-the-money) strike closest to the spot price.

    Returns
    -------
    float
        The strike price nearest to spot.
    """
    strikes = df["strike"].unique()
    return float(strikes[np.argmin(np.abs(strikes - spot_price))])
