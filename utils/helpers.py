"""
Utility functions and constants for the Options Pricing & Greeks Dashboard.
"""

from datetime import datetime, date, timedelta
from typing import Union, List
import numpy as np
import calendar

RISK_FREE_RATE = 0.065
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365

SUPPORTED_INSTRUMENTS = {
    "NIFTY": {
        "lot_size": 65,
        "tick_size": 0.05,
        "strike_gap": 50,
        "display_name": "NIFTY 50",
        "default_spot": 22800.0,
        "expiry_type": "weekly",
        "expiry_day_of_week": 1,  # Tuesday
    },
    "BANKNIFTY": {
        "lot_size": 15,
        "tick_size": 0.05,
        "strike_gap": 100,
        "display_name": "Bank NIFTY",
        "default_spot": 48500.0,
        "expiry_type": "monthly",
        "expiry_day_of_week": 1,  # Last Tuesday
    },
}

NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 10),   # Maha Shivaratri
    date(2026, 3, 17),   # Holi
    date(2026, 3, 31),   # Id-ul-Fitr (Eid) / Bank holiday
    date(2026, 4, 2),    # Ram Navami
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 25),   # Buddha Purnima
    date(2026, 6, 7),    # Bakri Id
    date(2026, 7, 7),    # Muharram
    date(2026, 8, 15),   # Independence Day
    date(2026, 9, 5),    # Milad un-Nabi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 9),   # Diwali (Laxmi Pujan)
    date(2026, 11, 10),  # Diwali Balipratipada
    date(2026, 11, 19),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
}

def is_trading_day(d: date) -> bool:
    """Check if a given date is a trading day (not weekend, not holiday)."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if d in NSE_HOLIDAYS_2026:
        return False
    return True

def adjust_expiry_for_holidays(expiry: date) -> date:
    """
    If expiry falls on a holiday/weekend, shift to the previous trading day.
    This is how NSE handles holidays — expiry moves to the day before.
    """
    while not is_trading_day(expiry):
        expiry -= timedelta(days=1)
    return expiry

def time_to_expiry(expiry_date: Union[str, date, datetime]) -> float:
    """
    Calculate time to expiry in years (calendar-day basis).
    Clamped to minimum 1/365 to avoid division-by-zero.
    """
    if isinstance(expiry_date, str):
        for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y"):
            try:
                expiry_date = datetime.strptime(expiry_date, fmt).date()
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Cannot parse expiry date: {expiry_date}")

    if isinstance(expiry_date, datetime):
        expiry_date = expiry_date.date()

    days_remaining = max((expiry_date - date.today()).days, 1)
    return days_remaining / CALENDAR_DAYS_PER_YEAR

def get_expiry_dates_from_today(num_expiries: int = 4, symbol: str = "NIFTY") -> List[date]:
    """
    Generate upcoming expiry dates with holiday adjustment.

    NIFTY: Weekly Tuesdays (shifted earlier if Tuesday is a holiday)
    BANKNIFTY: Monthly (last Tuesday of month, shifted if holiday)
    """
    instrument = SUPPORTED_INSTRUMENTS.get(symbol.upper(), SUPPORTED_INSTRUMENTS["NIFTY"])
    expiry_type = instrument.get("expiry_type", "weekly")
    day_of_week = instrument.get("expiry_day_of_week", 1)

    today = date.today()
    expiries = []

    if expiry_type == "weekly":
        days_ahead = (day_of_week - today.weekday()) % 7
        if days_ahead == 0:
            adjusted = adjust_expiry_for_holidays(today)
            expiries.append(adjusted)
            days_ahead = 7

        next_expiry = today + timedelta(days=days_ahead)
        while len(expiries) < num_expiries:
            adjusted = adjust_expiry_for_holidays(next_expiry)
            if adjusted not in expiries and adjusted >= today:
                expiries.append(adjusted)
            next_expiry += timedelta(weeks=1)

    elif expiry_type == "monthly":
        year, month = today.year, today.month
        while len(expiries) < num_expiries:
            last_day = calendar.monthrange(year, month)[1]
            last_date = date(year, month, last_day)
            offset = (last_date.weekday() - day_of_week) % 7
            monthly_expiry = last_date - timedelta(days=offset)
            adjusted = adjust_expiry_for_holidays(monthly_expiry)

            if adjusted >= today and adjusted not in expiries:
                expiries.append(adjusted)

            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

    return expiries

def format_number(value: float, decimals: int = 2) -> str:
    """Format number with commas. Returns '—' for NaN/Inf."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "—"
    return f"{value:,.{decimals}f}"

def format_indian(value: float) -> str:
    """Format number in Indian numbering system (lakhs, crores)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    value = int(round(value))
    is_negative = value < 0
    value = abs(value)
    s = str(value)
    if len(s) <= 3:
        result = s
    else:
        result = s[-3:]
        s = s[:-3]
        while s:
            result = s[-2:] + "," + result
            s = s[:-2]
    return ("-" + result) if is_negative else result

def get_color_for_value(value: float, metric: str = "delta") -> str:
    """Return hex color based on value and Greek type for heatmaps."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "#2d2d2d"

    color_maps = {
        "delta": ("#ef4444", "#6b7280", "#22c55e"),
        "gamma": ("#1e1b4b", "#7c3aed", "#fbbf24"),
        "theta": ("#22c55e", "#6b7280", "#ef4444"),
        "vega": ("#1e1b4b", "#3b82f6", "#06b6d4"),
        "rho": ("#ef4444", "#6b7280", "#22c55e"),
        "mispricing": ("#ef4444", "#6b7280", "#22c55e"),
    }
    low_color, mid_color, high_color = color_maps.get(metric, ("#ef4444", "#6b7280", "#22c55e"))

    if metric == "delta":
        normalized = np.clip(value, -1, 1)
    elif metric == "gamma":
        normalized = np.clip(value * 100, 0, 1)
    elif metric == "theta":
        normalized = np.clip(value / 50, -1, 0)
    elif metric == "vega":
        normalized = np.clip(value / 50, 0, 1)
    elif metric == "mispricing":
        normalized = np.clip(value / 20, -1, 1)
    else:
        normalized = np.clip(value, -1, 1)

    if normalized >= 0:
        return _interpolate_color(mid_color, high_color, abs(normalized))
    else:
        return _interpolate_color(mid_color, low_color, abs(normalized))

def _interpolate_color(color1: str, color2: str, t: float) -> str:
    """Linearly interpolate between two hex colors."""
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"

def validate_bs_inputs(S: float, K: float, T: float, r: float, sigma: float) -> bool:
    """Validate Black-Scholes input parameters."""
    if S <= 0:
        raise ValueError(f"Spot price must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"Time to expiry must be positive, got {T}")
    if sigma <= 0:
        raise ValueError(f"Volatility must be positive, got {sigma}")
    return True
