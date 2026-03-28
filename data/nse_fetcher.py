"""
NSE Option Chain Data Fetcher.

Fetches live option chain data from NSE India via its public JSON API.
Acts as a fallback when Kotak Neo API is unavailable (e.g. weekends/maintenance).
Contains NO mock data.
"""

import pandas as pd
import requests
from typing import Optional, Tuple

from utils.helpers import SUPPORTED_INSTRUMENTS

OUTPUT_COLUMNS = [
    "strike", "expiry",
    "ce_ltp", "ce_bid", "ce_ask", "ce_oi", "ce_volume", "ce_iv",
    "pe_ltp", "pe_bid", "pe_ask", "pe_oi", "pe_volume", "pe_iv",
]

NSE_BASE_URL = "https://www.nseindia.com"
NSE_OPTION_CHAIN_URL = f"{NSE_BASE_URL}/api/option-chain-indices"

def _get_nse_session() -> requests.Session:
    """Create a requests session with NSE-compatible headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/option-chain",
        "X-Requested-With": "XMLHttpRequest",
        "Connection": "keep-alive",
    })
    try:
        session.get(NSE_BASE_URL, timeout=10)
    except requests.RequestException:
        pass
    return session

def fetch_nse_option_chain(symbol: str = "NIFTY") -> Tuple[Optional[pd.DataFrame], Optional[float], str]:
    """
    Fetch live option chain (or Friday closing data on weekends) from NSE.

    Returns
    -------
    tuple: (DataFrame | None, spot_price | None, source_string)
    """
    symbol = symbol.upper()

    try:
        session = _get_nse_session()
        response = session.get(
            NSE_OPTION_CHAIN_URL,
            params={"symbol": symbol},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        spot = data.get("records", {}).get("underlyingValue", None)
        if spot is None:
            spot = data.get("filtered", {}).get("data", [{}])[0].get("PE", {}).get("underlyingValue", None)

        records = data.get("records", {}).get("data", [])
        if not records:
            return None, None, "No data in NSE API response"

        rows = []
        for record in records:
            row = {"strike": record.get("strikePrice"), "expiry": record.get("expiryDate")}
            ce = record.get("CE", {})
            pe = record.get("PE", {})

            row["ce_ltp"] = ce.get("lastPrice", 0)
            row["ce_bid"] = ce.get("bidprice", ce.get("bidPrice", 0))
            row["ce_ask"] = ce.get("askprice", ce.get("askPrice", 0))
            row["ce_oi"] = ce.get("openInterest", 0)
            row["ce_volume"] = ce.get("totalTradedVolume", 0)
            row["ce_iv"] = ce.get("impliedVolatility", 0)

            row["pe_ltp"] = pe.get("lastPrice", 0)
            row["pe_bid"] = pe.get("bidprice", pe.get("bidPrice", 0))
            row["pe_ask"] = pe.get("askprice", pe.get("askPrice", 0))
            row["pe_oi"] = pe.get("openInterest", 0)
            row["pe_volume"] = pe.get("totalTradedVolume", 0)
            row["pe_iv"] = pe.get("impliedVolatility", 0)

            rows.append(row)

        df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        df["ce_iv"] = df["ce_iv"] / 100.0
        df["pe_iv"] = df["pe_iv"] / 100.0

        return df, spot, "LIVE_NSE"

    except requests.Timeout:
        return None, None, "NSE API timeout"
    except requests.HTTPError as e:
        return None, None, f"NSE API HTTP error: {e.response.status_code}"
    except requests.RequestException as e:
        return None, None, f"Network error: {str(e)}"
    except (KeyError, ValueError, TypeError) as e:
        return None, None, f"Data parsing error: {str(e)}"

def generate_mock_data(
    symbol: str = "NIFTY",
    num_expiries: int = 4,
) -> Tuple[pd.DataFrame, float, str]:
    """Generate realistic option chain data with Black-Scholes pricing."""
    import numpy as np
    from datetime import date
    from models.black_scholes import black_scholes_price
    from utils.helpers import get_expiry_dates_from_today, CALENDAR_DAYS_PER_YEAR

    instrument = SUPPORTED_INSTRUMENTS.get(symbol.upper(), SUPPORTED_INSTRUMENTS["NIFTY"])
    spot = instrument["default_spot"]
    strike_gap = instrument["strike_gap"]

    lower_strike = int((spot * 0.90) // strike_gap) * strike_gap
    upper_strike = int((spot * 1.10) // strike_gap) * strike_gap
    strikes = np.arange(lower_strike, upper_strike + strike_gap, strike_gap)

    expiries = get_expiry_dates_from_today(num_expiries, symbol=symbol.upper())

    rows = []
    for expiry in expiries:
        T = max((expiry - date.today()).days, 1) / CALENDAR_DAYS_PER_YEAR

        for K in strikes:
            moneyness = K / spot
            base_iv = 0.14 + 0.02 * np.sqrt(T * 365 / 30)
            skew = 0.08 * (1 - moneyness) ** 2 if moneyness < 1 else 0.03 * (moneyness - 1) ** 2
            smile = 0.04 * (moneyness - 1) ** 2

            ce_iv = np.clip(base_iv + smile + skew * 0.5 + np.random.normal(0, 0.005), 0.05, 0.80)
            pe_iv = np.clip(base_iv + smile + skew + np.random.normal(0, 0.005), 0.05, 0.80)

            ce_price = float(black_scholes_price(spot, K, T, 0.065, ce_iv, "CE"))
            pe_price = float(black_scholes_price(spot, K, T, 0.065, pe_iv, "PE"))

            noise = 1 + np.random.normal(0, 0.015)
            ce_ltp = max(round(ce_price * noise * 20) / 20, 0.05)
            pe_ltp = max(round(pe_price * noise * 20) / 20, 0.05)

            spread = 0.005 + 0.01 * abs(moneyness - 1)
            ce_bid = round(ce_ltp * (1 - spread) * 20) / 20
            ce_ask = round(ce_ltp * (1 + spread) * 20) / 20
            pe_bid = round(pe_ltp * (1 - spread) * 20) / 20
            pe_ask = round(pe_ltp * (1 + spread) * 20) / 20

            atm_dist = abs(K - spot) / spot
            oi_base = (15_000_000 if symbol == "NIFTY" else 8_000_000) * np.exp(-50 * atm_dist ** 2)

            ce_oi = int(oi_base * (1 + 0.3 * max(0, moneyness - 1)) * np.random.uniform(0.7, 1.3))
            pe_oi = int(oi_base * (1 + 0.3 * max(0, 1 - moneyness)) * np.random.uniform(0.7, 1.3))

            vol_mult = min(0.15 * np.exp(-30 * atm_dist ** 2) * (1 / (T * 365 / 7)), 0.3)
            ce_volume = int(ce_oi * vol_mult * np.random.uniform(0.5, 1.5))
            pe_volume = int(pe_oi * vol_mult * np.random.uniform(0.5, 1.5))

            rows.append({
                "strike": K,
                "expiry": expiry.strftime("%d-%b-%Y"),
                "ce_ltp": ce_ltp,
                "ce_bid": max(ce_bid, 0.05),
                "ce_ask": max(ce_ask, 0.10),
                "ce_oi": ce_oi,
                "ce_volume": ce_volume,
                "ce_iv": ce_iv,
                "pe_ltp": pe_ltp,
                "pe_bid": max(pe_bid, 0.05),
                "pe_ask": max(pe_ask, 0.10),
                "pe_oi": pe_oi,
                "pe_volume": pe_volume,
                "pe_iv": pe_iv,
            })

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return df, spot, "MOCK"

def get_option_chain(
    symbol: str = "NIFTY",
    kotak_client=None,
) -> Tuple[Optional[pd.DataFrame], Optional[float], str]:
    """
    Get option chain data with 3-tier fallback:
        1. Kotak Neo API
        2. NSE direct API (handles weekends holding previous close)
        3. Mock Data (for UI testing)
    """
    if kotak_client is not None and kotak_client.is_session_ready:
        try:
            df, spot, source = kotak_client.get_option_chain(symbol)
            if df is not None and not df.empty and spot is not None:
                return df, spot, "LIVE_KOTAK"
        except Exception:
            pass

    df, spot, source = fetch_nse_option_chain(symbol)
    if df is not None and not df.empty and spot is not None:
        return df, spot, "LIVE_NSE"

    df, spot, source = generate_mock_data(symbol)
    return df, spot, source
