"""
Kotak Neo Trade API v2 — Live Option Chain Fetcher.

Direct REST API implementation (no SDK dependency).
Based on official docs: https://github.com/Kotak-Neo/Kotak-neo-api-v2

Auth Flow:
  Step 1: POST /login/1.0/tradeApiLogin   → TOTP login → get view token + SID
  Step 2: POST /login/1.0/tradeApiValidate → MPIN       → get trade token + SID + baseUrl

Then use baseUrl for all data APIs (quotes, scrip search, scrip master).
"""

import io
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, List, Dict
import logging
import json

from utils.helpers import SUPPORTED_INSTRUMENTS

logger = logging.getLogger(__name__)

LOGIN_URL = "https://mis.kotaksecurities.com/login/1.0/tradeApiLogin"
VALIDATE_URL = "https://mis.kotaksecurities.com/login/1.0/tradeApiValidate"
NEO_FIN_KEY = "neotradeapi"

OUTPUT_COLUMNS = [
    "strike", "expiry",
    "ce_ltp", "ce_bid", "ce_ask", "ce_oi", "ce_volume", "ce_iv",
    "pe_ltp", "pe_bid", "pe_ask", "pe_oi", "pe_volume", "pe_iv",
]

class KotakNeoClient:
    """
    Direct REST API client for Kotak Neo Trade API v2.

    No SDK dependency — uses plain requests library.

    Usage:
        client = KotakNeoClient(consumer_key="your_token")
        result = client.totp_login(mobile="+917042425323", ucc="YNNCI", totp="123456")
        result = client.validate_mpin(mpin="1234")
        df, spot, source = client.get_option_chain("NIFTY")
    """

    def __init__(self, consumer_key: str):
        self._consumer_key = consumer_key
        self._view_token = None
        self._view_sid = None
        self._trade_token = None
        self._trade_sid = None
        self._base_url = None
        self._server_id = None
        self._totp_done = False
        self._session_ready = False
        self._login_error = None
        self._scrip_cache = {}  # exchange_segment → DataFrame

    @property
    def is_totp_done(self) -> bool:
        return self._totp_done

    @property
    def is_session_ready(self) -> bool:
        return self._session_ready

    @property
    def login_error(self) -> Optional[str]:
        return self._login_error

    def initialize(self) -> bool:
        """No initialization needed for direct REST — always returns True."""
        self._login_error = None
        return True

    def totp_login(self, mobile: str, ucc: str, totp: str) -> dict:
        """
        Step 1: Login with TOTP.

        POST https://mis.kotaksecurities.com/login/1.0/tradeApiLogin
        Headers: Authorization, neo-fin-key
        Body: {mobileNumber, ucc, totp}
        """
        try:
            headers = {
                "Authorization": self._consumer_key,
                "neo-fin-key": NEO_FIN_KEY,
                "Content-Type": "application/json",
            }
            payload = {
                "mobileNumber": mobile,
                "ucc": ucc.upper(),
                "totp": totp,
            }

            logger.info(f"Kotak Neo: TOTP login for UCC={ucc}")
            resp = requests.post(LOGIN_URL, headers=headers, json=payload, timeout=30)

            if resp.status_code != 200:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                self._login_error = error_msg
                return {"error": error_msg}

            data = resp.json()
            logger.info(f"Kotak Neo TOTP response: {json.dumps(data, indent=2)[:500]}")

            inner = data.get("data", data)

            if isinstance(inner, dict) and inner.get("token"):
                self._view_token = inner["token"]
                self._view_sid = inner.get("sid", "")
                self._server_id = inner.get("hsServerId", "")
                self._totp_done = True
                self._login_error = None
                logger.info("Kotak Neo: TOTP login successful — enter MPIN next")
                return {"status": "ok", "message": "TOTP verified — enter MPIN"}
            else:
                error_detail = inner.get("message", "") or inner.get("error", "") or str(data)
                self._login_error = f"TOTP login failed: {error_detail}"
                return {"error": self._login_error}

        except requests.exceptions.RequestException as e:
            self._login_error = f"Network error: {str(e)}"
            return {"error": self._login_error}
        except Exception as e:
            self._login_error = f"TOTP login error: {str(e)}"
            return {"error": self._login_error}

    def validate_mpin(self, mpin: str) -> dict:
        """
        Step 2: Validate MPIN to get full trading access.

        POST https://mis.kotaksecurities.com/login/1.0/tradeApiValidate
        Headers: Authorization, neo-fin-key, sid, Auth
        Body: {mpin}
        """
        if not self._totp_done:
            return {"error": "Must call totp_login() first"}

        try:
            headers = {
                "Authorization": self._consumer_key,
                "neo-fin-key": NEO_FIN_KEY,
                "sid": self._view_sid,
                "Auth": self._view_token,
                "Content-Type": "application/json",
            }
            payload = {"mpin": mpin}

            logger.info("Kotak Neo: Validating MPIN...")
            resp = requests.post(VALIDATE_URL, headers=headers, json=payload, timeout=30)

            if resp.status_code != 200:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                self._login_error = error_msg
                return {"error": error_msg}

            data = resp.json()
            logger.info(f"Kotak Neo MPIN response: {json.dumps(data, indent=2)[:500]}")

            inner = data.get("data", data)

            if isinstance(inner, dict) and inner.get("token"):
                self._trade_token = inner["token"]
                self._trade_sid = inner.get("sid", "")
                self._base_url = inner.get("baseUrl", "https://cis.kotaksecurities.com")
                self._session_ready = True
                self._login_error = None
                logger.info(f"Kotak Neo: Session active! Base URL: {self._base_url}")
                return {"status": "ok", "message": "Session active — live data ready"}
            else:
                error_detail = inner.get("message", "") or inner.get("error", "") or str(data)
                self._login_error = f"MPIN validation failed: {error_detail}"
                return {"error": self._login_error}

        except requests.exceptions.RequestException as e:
            self._login_error = f"Network error: {str(e)}"
            return {"error": self._login_error}
        except Exception as e:
            self._login_error = f"MPIN validation error: {str(e)}"
            return {"error": self._login_error}

    def _auth_headers(self) -> dict:
        """Standard auth headers for all trading/data API calls."""
        return {
            "Authorization": self._consumer_key,
            "Auth": self._trade_token,
            "Sid": self._trade_sid,
            "neo-fin-key": NEO_FIN_KEY,
            "Content-Type": "application/json",
        }

    def _load_scrip_master(self, exchange_segment: str = "nse_fo") -> Optional[pd.DataFrame]:
        """
        Download scrip master CSV for the exchange segment.
        GET {baseUrl}/script-details/1.0/masterscrip/file-paths
        """
        if exchange_segment in self._scrip_cache:
            return self._scrip_cache[exchange_segment]

        try:
            url = f"{self._base_url}/script-details/1.0/masterscrip/file-paths"
            resp = requests.get(url, headers=self._auth_headers(), timeout=30)

            if resp.status_code != 200:
                logger.warning(f"Scrip master paths failed: {resp.status_code}")
                return None

            data = resp.json()
            file_paths = data.get("filesPaths", data.get("data", {}).get("filesPaths", []))

            if not file_paths:
                logger.warning(f"No file paths in scrip master response: {str(data)[:300]}")
                return None

            segment_map = {
                "nse_fo": "nse_fo",
                "nse_cm": "nse_cm",
                "bse_fo": "bse_fo",
            }
            target = segment_map.get(exchange_segment, exchange_segment)

            csv_url = None
            for fp in file_paths:
                if isinstance(fp, str) and target in fp.lower():
                    csv_url = fp
                    break
                elif isinstance(fp, dict):
                    seg = fp.get("exchange_segment", "") or fp.get("exchangeSegment", "")
                    if target in seg.lower():
                        csv_url = fp.get("url", fp.get("filePath", ""))
                        break

            if not csv_url:
                if file_paths:
                    csv_url = file_paths[0] if isinstance(file_paths[0], str) else file_paths[0].get("url", "")

            if csv_url:
                csv_resp = requests.get(csv_url, timeout=60)
                if csv_resp.status_code == 200:
                    df = pd.read_csv(io.StringIO(csv_resp.text))
                    self._scrip_cache[exchange_segment] = df
                    logger.info(f"Loaded scrip master for {exchange_segment}: {len(df)} rows")
                    return df

            return None

        except Exception as e:
            logger.warning(f"Scrip master load error: {e}")
            return None

    def _search_scrip(self, exchange_segment: str, symbol: str,
                      expiry: str = None, option_type: str = None,
                      strike_price: str = None) -> list:
        """
        Search for scrips using the search API.
        """
        try:
            url = f"{self._base_url}/script-details/1.0/scrip-search"
            params = {
                "exchange_segment": exchange_segment,
                "symbol": symbol.lower(),
            }
            if expiry:
                params["expiry"] = expiry
            if option_type:
                params["option_type"] = option_type
            if strike_price:
                params["strike_price"] = strike_price

            resp = requests.get(url, headers=self._auth_headers(), params=params, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("data", data.get("message", []))
                if isinstance(results, list):
                    return results
                elif isinstance(results, dict):
                    return [results]
            return []

        except Exception as e:
            logger.warning(f"Scrip search error: {e}")
            return []

    def _get_quotes(self, instrument_tokens: List[Dict],
                    quote_type: str = "all") -> list:
        """
        Get quotes for instrument tokens.
        POST {baseUrl}/quick/user/quote
        """
        all_quotes = []
        batch_size = 20

        for i in range(0, len(instrument_tokens), batch_size):
            batch = instrument_tokens[i:i + batch_size]
            try:
                url = f"{self._base_url}/quick/user/quote"
                payload = {
                    "instrument_tokens": batch,
                    "quote_type": quote_type,
                }
                resp = requests.post(
                    url, headers=self._auth_headers(),
                    json=payload, timeout=30
                )

                if resp.status_code == 200:
                    data = resp.json()
                    quotes = data.get("data", data.get("message", []))
                    if isinstance(quotes, list):
                        all_quotes.extend(quotes)
                    elif isinstance(quotes, dict):
                        all_quotes.append(quotes)

            except Exception as e:
                logger.warning(f"Quotes batch {i} error: {e}")

        return all_quotes

    def _get_index_ltp(self, symbol: str) -> Optional[float]:
        """Get current spot price of index."""
        try:
            index_token = "Nifty 50" if symbol.upper() == "NIFTY" else "Nifty Bank"
            quotes = self._get_quotes(
                instrument_tokens=[{
                    "instrument_token": index_token,
                    "exchange_segment": "nse_cm",
                }],
                quote_type="ltp",
            )

            for q in quotes:
                for key in ["last_traded_price", "ltp", "lastTradedPrice"]:
                    if key in q:
                        val = float(q[key])
                        return val / 100 if val > 100000 else val
            return None

        except Exception as e:
            logger.warning(f"Index LTP error: {e}")
            return None

    def get_option_chain(
        self,
        symbol: str = "NIFTY",
        expiry_date: Optional[str] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[float], str]:
        """
        Build full option chain from Kotak Neo REST API.

        Returns: (DataFrame, spot_price, 'LIVE_KOTAK') or (None, None, error)
        """
        if not self._session_ready:
            return None, None, "Kotak Neo session not ready — complete login first"

        try:
            instrument = SUPPORTED_INSTRUMENTS.get(
                symbol.upper(), SUPPORTED_INSTRUMENTS["NIFTY"]
            )
            spot = self._get_index_ltp(symbol)
            if spot is None:
                spot = instrument["default_spot"]
                logger.warning(f"Using default spot for {symbol}: {spot}")

            if expiry_date is None:
                from utils.helpers import get_expiry_dates_from_today
                expiries = get_expiry_dates_from_today(1, symbol=symbol)
                if expiries:
                    expiry_date = expiries[0].strftime("%d-%b-%Y")
                else:
                    return None, None, "Could not determine expiry date"

            all_scrips = []
            for opt_type in ["CE", "PE"]:
                scrips = self._search_scrip(
                    exchange_segment="nse_fo",
                    symbol=symbol,
                    expiry=expiry_date,
                    option_type=opt_type,
                )
                all_scrips.extend(scrips)

            if not all_scrips:
                logger.warning(f"No scrips found for {symbol} {expiry_date}, trying scrip master")
                return self._build_chain_from_scrip_master(symbol, expiry_date, spot)

            token_list = []
            scrip_map = {}

            for scrip in all_scrips:
                token = (
                    scrip.get("pSymbol")
                    or scrip.get("instrument_token")
                    or scrip.get("pSymbolName")
                    or ""
                )
                if token:
                    token_list.append({
                        "instrument_token": token,
                        "exchange_segment": "nse_fo",
                    })
                    scrip_map[token] = scrip

            if not token_list:
                return None, None, "No valid instrument tokens found"

            quotes = self._get_quotes(token_list, quote_type="all")

            chain = self._assemble_chain(quotes, scrip_map, expiry_date, spot)

            if not chain:
                return None, None, "No option data in quotes response"

            df = pd.DataFrame(
                sorted(chain.values(), key=lambda x: x["strike"]),
                columns=OUTPUT_COLUMNS,
            )

            logger.info(f"Kotak Neo: fetched {len(df)} strikes for {symbol}")
            return df, spot, "LIVE_KOTAK"

        except Exception as e:
            error_msg = f"Kotak Neo API error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, None, error_msg

    def _build_chain_from_scrip_master(
        self, symbol: str, expiry_date: str, spot: float
    ) -> Tuple[Optional[pd.DataFrame], Optional[float], str]:
        """Fallback: build option chain from scrip master CSV + quotes."""
        master = self._load_scrip_master("nse_fo")
        if master is None or master.empty:
            return None, None, "Could not load scrip master"

        sym_col = None
        for col in ["pSymbolName", "symbol", "pSymbol", "trading_symbol"]:
            if col in master.columns:
                sym_col = col
                break

        if sym_col is None:
            return None, None, f"Unknown scrip master format: {list(master.columns)[:10]}"

        mask = master[sym_col].str.upper().str.contains(symbol.upper(), na=False)
        filtered = master[mask]

        if filtered.empty:
            return None, None, f"No {symbol} options in scrip master"

        token_list = []
        scrip_map = {}

        for _, row in filtered.head(200).iterrows():
            token = str(row.get("pSymbol", row.get("instrument_token", "")))
            if token:
                token_list.append({
                    "instrument_token": token,
                    "exchange_segment": "nse_fo",
                })
                scrip_map[token] = row.to_dict()

        if not token_list:
            return None, None, "No tokens from scrip master"

        quotes = self._get_quotes(token_list, quote_type="all")
        chain = self._assemble_chain(quotes, scrip_map, expiry_date, spot)

        if not chain:
            return None, None, "No quotes data from scrip master tokens"

        df = pd.DataFrame(
            sorted(chain.values(), key=lambda x: x["strike"]),
            columns=OUTPUT_COLUMNS,
        )
        return df, spot, "LIVE_KOTAK"

    def _assemble_chain(self, quotes: list, scrip_map: dict,
                        expiry_date: str, spot: float) -> dict:
        """Assemble option chain dict from quotes + scrip info."""
        chain = {}

        for quote in quotes:
            token = quote.get("instrument_token", "")
            scrip = scrip_map.get(token, {})

            strike = 0.0
            for key in ["strike_price", "strikePrice", "pStrikePrice"]:
                val = scrip.get(key) or quote.get(key)
                if val:
                    try:
                        strike = float(val)
                        break
                    except (ValueError, TypeError):
                        pass

            opt_type = ""
            for key in ["option_type", "optionType", "pOptionType"]:
                val = scrip.get(key) or quote.get(key)
                if val:
                    opt_type = str(val).upper()
                    break

            if strike <= 0 or opt_type not in ("CE", "PE"):
                continue

            if strike not in chain:
                chain[strike] = {
                    "strike": strike, "expiry": expiry_date,
                    "ce_ltp": 0, "ce_bid": 0, "ce_ask": 0,
                    "ce_oi": 0, "ce_volume": 0, "ce_iv": 0,
                    "pe_ltp": 0, "pe_bid": 0, "pe_ask": 0,
                    "pe_oi": 0, "pe_volume": 0, "pe_iv": 0,
                }

            prefix = "ce" if opt_type == "CE" else "pe"

            ltp = self._safe_float(quote, ["last_traded_price", "ltp", "lastTradedPrice"])
            if ltp > spot * 2:
                ltp /= 100
            chain[strike][f"{prefix}_ltp"] = ltp

            chain[strike][f"{prefix}_bid"] = self._safe_float(
                quote, ["best_bid_price", "bidPrice"]
            )
            chain[strike][f"{prefix}_ask"] = self._safe_float(
                quote, ["best_ask_price", "askPrice"]
            )

            chain[strike][f"{prefix}_oi"] = int(
                self._safe_float(quote, ["open_interest", "openInterest", "oi"])
            )

            chain[strike][f"{prefix}_volume"] = int(
                self._safe_float(quote, ["volume", "total_quantity_traded", "totalQtyTraded"])
            )

            iv = self._safe_float(quote, ["implied_volatility", "iv"])
            chain[strike][f"{prefix}_iv"] = iv / 100 if iv > 1 else iv

        return chain

    @staticmethod
    def _safe_float(data: dict, keys: list) -> float:
        """Extract first non-None float value from dict using multiple key names."""
        for key in keys:
            val = data.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return 0.0

    def logout(self):
        """Terminate the session."""
        self._totp_done = False
        self._session_ready = False
        self._view_token = None
        self._trade_token = None
        self._trade_sid = None
        self._scrip_cache.clear()
        logger.info("Kotak Neo: Session terminated")
