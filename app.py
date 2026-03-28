"""
Options Pricing & Greeks Dashboard
===================================
A production-quality Streamlit dashboard for real-time NSE options analysis
using the Black-Scholes pricing model.

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots

from data.nse_fetcher import get_option_chain
from data.kotak_neo_fetcher import KotakNeoClient
from data.option_chain import (
    process_option_chain, filter_by_expiry, filter_by_moneyness,
    get_available_expiries, get_atm_strike,
)
from models.black_scholes import black_scholes_price
from models.greeks import compute_all_greeks_for_chain, compute_greeks
from utils.helpers import (
    SUPPORTED_INSTRUMENTS, RISK_FREE_RATE,
    format_number, format_indian, time_to_expiry,
)

st.set_page_config(
    page_title="Options Pricing & Greeks Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: #0e1117;
        border-right: 1px solid #2d2d30;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.8rem !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: #1a1c24;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-weight: 700;
        font-size: 1.6rem;
        color: #ffffff;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #111217;
        padding: 4px;
        border-bottom: 2px solid #2d2d30;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 500;
        font-size: 0.85rem;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #ffffff !important;
        border-bottom: 2px solid #ffffff !important;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(139, 92, 246, 0.12);
    }

    .dashboard-header {
        background: #1a1c24;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    .dashboard-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    .dashboard-header p {
        color: #8b949e;
        font-size: 0.95rem;
        margin: 0;
    }

    .source-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .source-live {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .source-mock {
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c4b5fd;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(139, 92, 246, 0.15);
    }

    .overpriced { color: #f87171; font-weight: 600; }
    .underpriced { color: #4ade80; font-weight: 600; }
    .fairpriced { color: #94a3b8; }

    footer {visibility: hidden;}
    /* Restored visibility of Streamlit header to show sidebar toggle */
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("##  Dashboard Controls")
    st.markdown("---")
    symbol = st.selectbox(
        " Instrument",
        list(SUPPORTED_INSTRUMENTS.keys()),
        format_func=lambda x: SUPPORTED_INSTRUMENTS[x]["display_name"],
        help="Select NIFTY or BANKNIFTY index options",
    )

    risk_free_rate = st.slider(
        " Risk-Free Rate (%)",
        min_value=1.0, max_value=15.0,
        value=RISK_FREE_RATE * 100,
        step=0.25,
        help="RBI repo rate, used in Black-Scholes pricing",
    ) / 100.0

    strike_range_pct = st.slider(
        " Strike Range (% of Spot)",
        min_value=2.0, max_value=20.0,
        value=8.0, step=1.0,
        help="Filter strikes within this range of the current spot price",
    )

    option_filter = st.radio(
        "Option Type",
        ["Both", "CE (Calls)", "PE (Puts)"],
        horizontal=True,
    )

    auto_refresh = st.toggle(
        " Auto-Refresh",
        value=False,
        help="Automatically refresh data every 30 seconds (useful during market hours)",
    )

    if auto_refresh:
        refresh_interval = st.select_slider(
            "Refresh interval",
            options=[15, 30, 60, 120],
            value=30,
            format_func=lambda x: f"{x}s",
        )
    else:
        refresh_interval = None

    st.write("")
    refresh = st.button("🔄 Refresh Data", use_container_width=True, type="primary")

    st.markdown("---")

    with st.expander(" Kotak Neo — Live Data", expanded=False):
        st.caption(
            "Connect your Kotak Neo account for real-time market data. "
            "Free — no API charges."
        )

        if "kotak_client" not in st.session_state:
            st.session_state.kotak_client = None
        if "kotak_step" not in st.session_state:
            st.session_state.kotak_step = "credentials"  # credentials → mpin → connected

        kotak_client = st.session_state.kotak_client

        if st.session_state.kotak_step == "connected" and kotak_client and kotak_client.is_session_ready:
            st.success("✅ Connected to Kotak Neo — Live data active")
            if st.button("🔓 Disconnect", key="kotak_disconnect"):
                kotak_client.logout()
                st.session_state.kotak_client = None
                st.session_state.kotak_step = "credentials"
                st.rerun()
        else:
            if st.session_state.kotak_step == "credentials":
                st.caption(
                    "📋 **Setup**: Register for TOTP at "
                    "[kotaksecurities.com](https://www.kotaksecurities.com/platform/kotak-neo-trade-api/) "
                    "→ scan QR with Google Authenticator."
                )
                consumer_key = st.text_input(
                    "Consumer Key (Token)",
                    type="password",
                    help="From neo.kotaksecurities.com → Trade API → Default Application",
                    key="kotak_key",
                )
                mobile = st.text_input(
                    "Mobile Number",
                    placeholder="+917042425323",
                    help="Registered with Kotak Securities (with +91 prefix)",
                    key="kotak_mobile",
                )
                ucc = st.text_input(
                    "UCC (Unique Client Code)",
                    placeholder="YNNCI",
                    help="Found in Kotak Neo → Profile → Account details",
                    key="kotak_ucc",
                )
                totp = st.text_input(
                    "TOTP (from Authenticator App)",
                    type="password",
                    max_chars=6,
                    help="6-digit code from Google Authenticator",
                    key="kotak_totp",
                )

                if st.button(" Login with TOTP", key="kotak_login", use_container_width=True):
                    if consumer_key and mobile and ucc and totp:
                        client = KotakNeoClient(consumer_key=consumer_key)
                        if client.initialize():
                            result = client.totp_login(
                                mobile=mobile, ucc=ucc, totp=totp
                            )
                            if isinstance(result, dict) and "error" in result:
                                st.error(f"Login failed: {result['error']}")
                            else:
                                st.session_state.kotak_client = client
                                st.session_state.kotak_step = "mpin"
                                st.rerun()
                        else:
                            st.error(client.login_error)
                    else:
                        st.warning("Please fill in all 4 fields")

            elif st.session_state.kotak_step == "mpin":
                st.info("🔑 TOTP verified — now enter your MPIN")
                mpin = st.text_input(
                    "MPIN",
                    type="password",
                    max_chars=6,
                    help="Your Kotak Neo MPIN (4-6 digits)",
                    key="kotak_mpin",
                )
                col_verify, col_cancel = st.columns(2)
                with col_verify:
                    if st.button("✅ Validate", key="kotak_validate", use_container_width=True):
                        if mpin and kotak_client:
                            result = kotak_client.validate_mpin(mpin=mpin)
                            if isinstance(result, dict) and "error" in result:
                                st.error(f"MPIN failed: {result['error']}")
                            else:
                                st.session_state.kotak_step = "connected"
                                st.rerun()
                with col_cancel:
                    if st.button("❌ Cancel", key="kotak_cancel", use_container_width=True):
                        st.session_state.kotak_client = None
                        st.session_state.kotak_step = "credentials"
                        st.rerun()

_kotak_client = st.session_state.get("kotak_client", None)
_use_live = _kotak_client is not None and _kotak_client.is_session_ready

def load_data_live(sym: str, kotak_client):
    """Fetch live data directly from Kotak (no caching)."""
    return get_option_chain(sym, kotak_client=kotak_client)

@st.cache_data(ttl=300, show_spinner=False)
def load_data_cached(sym: str):
    """Fetch data with caching (NSE fallback for weekends)."""
    return get_option_chain(sym)

if refresh:
    load_data_cached.clear()

with st.spinner("Fetching option chain data..."):
    if _use_live:
        raw_df, spot_price, data_source = load_data_live(symbol, _kotak_client)
    else:
        raw_df, spot_price, data_source = load_data_cached(symbol)

if raw_df is None or raw_df.empty or spot_price is None:
    st.title("Options Pricing & Greeks Dashboard")
    st.caption("No live data available - Offline")

    st.error(f"**Failed to fetch data:** {data_source or 'No live or fallback data available'}")
    st.warning("""
    **To get real-time data:**
    1. **Kotak Neo (recommended):** Open the sidebar →  Kotak Neo → Login with TOTP + MPIN
    2. **NSE Direct Fallback:** Shows previous day's closing data on weekends.
    """)
    st.stop()

if auto_refresh and refresh_interval:
    import time, threading
    st.toast(f" Auto-refresh in {refresh_interval}s", icon="")
    def _schedule_rerun():
        time.sleep(refresh_interval)
    _t = threading.Thread(target=_schedule_rerun, daemon=True)
    _t.start()

chain_df = process_option_chain(raw_df, spot_price, risk_free_rate)

expiries = get_available_expiries(chain_df)

with st.sidebar:
    selected_expiry = st.selectbox(
        " Expiry Date",
        expiries,
        help="Select option expiry date",
    )

df = filter_by_expiry(chain_df, selected_expiry)

df = filter_by_moneyness(df, spot_price, strike_range_pct)

df = df.sort_values("strike").reset_index(drop=True)

T = time_to_expiry(selected_expiry)

greeks_df = compute_all_greeks_for_chain(
    S=spot_price,
    strikes=df["strike"].values,
    T=T,
    r=risk_free_rate,
    ce_sigmas=df["ce_iv"].values,
    pe_sigmas=df["pe_iv"].values,
)

df = df.merge(greeks_df, on="strike", how="left")

atm_strike = get_atm_strike(df, spot_price)

lot_size = SUPPORTED_INSTRUMENTS[symbol]["lot_size"]

badge_color = "warning" if "MOCK" in data_source else "success"
source_label = "SIMULATED" if "MOCK" in data_source else ("LIVE" if "KOTAK" in data_source else "LIVE (NSE)")

st.title("Options Pricing & Greeks Dashboard")
st.caption(f"{symbol} · Expiry: {selected_expiry} · {len(df)} strikes loaded · **Status:** :{badge_color}[{source_label}]")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Spot Price", f"₹{format_number(spot_price)}")
with col2:
    st.metric("ATM Strike", f"₹{format_number(atm_strike)}")
with col3:
    st.metric("Days to Expiry", f"{max(int(T * 365), 1)}")
with col4:
    atm_row = df.loc[(df["strike"] - atm_strike).abs().idxmin()]
    atm_iv = atm_row["ce_iv"] * 100
    st.metric("ATM IV (CE)", f"{atm_iv:.1f}%")
with col5:
    st.metric("Lot Size", f"{lot_size}")

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.1)", zeroline=False),
    margin=dict(l=60, r=30, t=50, b=60),
    hoverlabel=dict(
        bgcolor="rgba(15,10,30,0.9)",
        bordercolor="rgba(139,92,246,0.3)",
        font_size=12,
    ),
    legend=dict(
        bgcolor="rgba(15,10,30,0.5)",
        bordercolor="rgba(139,92,246,0.15)",
        borderwidth=1,
    ),
)

CE_COLOR = "#a78bfa"       # Purple for calls
PE_COLOR = "#60a5fa"       # Blue for puts
CE_FILL = "rgba(167,139,250,0.1)"
PE_FILL = "rgba(96,165,250,0.1)"
ACCENT = "#fbbf24"         # Gold accent
ATM_COLOR = "rgba(251,191,36,0.4)"

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Option Chain",
    " Greeks Analysis",
    "Greeks Heatmap",
    " Pricing & IV",
    "Payoff Diagram",
])

with tab1:
    st.subheader("Option Chain with Mispricing Analysis", divider="gray")

    show_ce = option_filter in ["Both", "CE (Calls)"]
    show_pe = option_filter in ["Both", "PE (Puts)"]

    display_cols = ["strike"]

    if show_ce:
        display_cols += ["ce_ltp", "bs_ce_price", "ce_mispricing", "ce_mispricing_pct", "ce_iv", "ce_oi", "ce_volume"]
    if show_pe:
        display_cols += ["pe_ltp", "bs_pe_price", "pe_mispricing", "pe_mispricing_pct", "pe_iv", "pe_oi", "pe_volume"]

    display_df = df[display_cols].copy()

    if show_ce:
        display_df["ce_iv"] = (display_df["ce_iv"] * 100).round(2)
    if show_pe:
        display_df["pe_iv"] = (display_df["pe_iv"] * 100).round(2)

    round_cols = [c for c in display_df.columns if c != "strike"]
    for col in round_cols:
        if "oi" in col or "volume" in col:
            display_df[col] = display_df[col].astype(int)
        else:
            display_df[col] = display_df[col].round(2)

    rename_map = {
        "strike": "Strike",
        "ce_ltp": "CE LTP", "bs_ce_price": "CE BS Price",
        "ce_mispricing": "CE Misprice (₹)", "ce_mispricing_pct": "CE Misprice (%)",
        "ce_iv": "CE IV (%)", "ce_oi": "CE OI", "ce_volume": "CE Vol",
        "pe_ltp": "PE LTP", "bs_pe_price": "PE BS Price",
        "pe_mispricing": "PE Misprice (₹)", "pe_mispricing_pct": "PE Misprice (%)",
        "pe_iv": "PE IV (%)", "pe_oi": "PE OI", "pe_volume": "PE Vol",
    }
    display_df = display_df.rename(columns=rename_map)

    def style_mispricing(val):
        if isinstance(val, (int, float)):
            if val > 2:
                return "color: #f87171; font-weight: 600"   # Overpriced → red
            elif val < -2:
                return "color: #4ade80; font-weight: 600"   # Underpriced → green
        return "color: #94a3b8"

    styled = display_df.style
    mispricing_cols = [c for c in display_df.columns if "Misprice" in c]
    for col in mispricing_cols:
        styled = styled.map(style_mispricing, subset=[col])

    def highlight_atm(row):
        if row["Strike"] == atm_strike:
            return ["background-color: rgba(251,191,36,0.08)"] * len(row)
        return [""] * len(row)

    styled = styled.apply(highlight_atm, axis=1)
    styled = styled.format(precision=2, na_rep="—")

    st.dataframe(styled, use_container_width=True, height=500)

    st.subheader("Mispricing Summary", divider="gray")

    col_a, col_b = st.columns(2)

    if show_ce:
        with col_a:
            overpriced_ce = df[df["ce_mispricing_pct"] > 3].nlargest(5, "ce_mispricing_pct")
            underpriced_ce = df[df["ce_mispricing_pct"] < -3].nsmallest(5, "ce_mispricing_pct")

            st.markdown("**Most Overpriced Calls**")
            if not overpriced_ce.empty:
                for _, row in overpriced_ce.iterrows():
                    st.markdown(
                        f"Strike **{format_number(row['strike'])}**: "
                        f"Market ₹{format_number(row['ce_ltp'])} vs "
                        f"BS ₹{format_number(row['bs_ce_price'])} "
                        f"(**+{row['ce_mispricing_pct']:.1f}%**)"
                    )
            else:
                st.caption("No significantly overpriced calls found.")

            st.markdown("**Most Underpriced Calls**")
            if not underpriced_ce.empty:
                for _, row in underpriced_ce.iterrows():
                    st.markdown(
                        f"Strike **{format_number(row['strike'])}**: "
                        f"Market ₹{format_number(row['ce_ltp'])} vs "
                        f"BS ₹{format_number(row['bs_ce_price'])} "
                        f"(**{row['ce_mispricing_pct']:.1f}%**)"
                    )
            else:
                st.caption("No significantly underpriced calls found.")

    if show_pe:
        with col_b:
            overpriced_pe = df[df["pe_mispricing_pct"] > 3].nlargest(5, "pe_mispricing_pct")
            underpriced_pe = df[df["pe_mispricing_pct"] < -3].nsmallest(5, "pe_mispricing_pct")

            st.markdown("**Most Overpriced Puts**")
            if not overpriced_pe.empty:
                for _, row in overpriced_pe.iterrows():
                    st.markdown(
                        f"Strike **{format_number(row['strike'])}**: "
                        f"Market ₹{format_number(row['pe_ltp'])} vs "
                        f"BS ₹{format_number(row['bs_pe_price'])} "
                        f"(**+{row['pe_mispricing_pct']:.1f}%**)"
                    )
            else:
                st.caption("No significantly overpriced puts found.")

            st.markdown("**Most Underpriced Puts**")
            if not underpriced_pe.empty:
                for _, row in underpriced_pe.iterrows():
                    st.markdown(
                        f"Strike **{format_number(row['strike'])}**: "
                        f"Market ₹{format_number(row['pe_ltp'])} vs "
                        f"BS ₹{format_number(row['bs_pe_price'])} "
                        f"(**{row['pe_mispricing_pct']:.1f}%**)"
                    )
            else:
                st.caption("No significantly underpriced puts found.")

with tab2:
    st.subheader("Option Greeks vs Strike Price", divider="gray")

    greek_names = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
    selected_greeks = st.multiselect(
        "Select Greeks to display",
        greek_names,
        default=["Delta", "Gamma", "Theta"],
    )

    if not selected_greeks:
        st.info("Select at least one Greek to display.")
    else:
        for greek_name in selected_greeks:
            g_lower = greek_name.lower()
            ce_col = f"ce_{g_lower}"
            pe_col = f"pe_{g_lower}"

            units = {
                "Delta": "per ₹1 move",
                "Gamma": "per ₹1 move",
                "Theta": "per day (₹)",
                "Vega": "per 1% IV change (₹)",
                "Rho": "per 1% rate change (₹)",
            }

            fig = go.Figure()

            if show_ce:
                fig.add_trace(go.Scatter(
                    x=df["strike"], y=df[ce_col],
                    name=f"CE {greek_name}",
                    mode="lines+markers",
                    line=dict(color=CE_COLOR, width=2.5),
                    marker=dict(size=5, color=CE_COLOR),
                    fill="tozeroy",
                    fillcolor=CE_FILL,
                    hovertemplate=f"Strike: %{{x}}<br>CE {greek_name}: %{{y:.4f}}<extra></extra>",
                ))

            if show_pe:
                fig.add_trace(go.Scatter(
                    x=df["strike"], y=df[pe_col],
                    name=f"PE {greek_name}",
                    mode="lines+markers",
                    line=dict(color=PE_COLOR, width=2.5),
                    marker=dict(size=5, color=PE_COLOR),
                    fill="tozeroy",
                    fillcolor=PE_FILL,
                    hovertemplate=f"Strike: %{{x}}<br>PE {greek_name}: %{{y:.4f}}<extra></extra>",
                ))

            fig.add_vline(
                x=atm_strike,
                line=dict(color=ATM_COLOR, width=1, dash="dot"),
                annotation_text="ATM",
                annotation_font_color=ACCENT,
            )

            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(
                    text=f"{greek_name} <span style='font-size:0.7em;color:#94a3b8'>({units[greek_name]})</span>",
                    font_size=16,
                ),
                xaxis_title="Strike Price",
                yaxis_title=greek_name,
                height=380,
            )

            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Greeks Heatmap across Strikes", divider="gray")

    heatmap_type = st.radio(
        "Option Type for Heatmap",
        ["CE (Calls)", "PE (Puts)"],
        horizontal=True,
        key="heatmap_type",
    )
    prefix = "ce" if "CE" in heatmap_type else "pe"

    greek_list = ["delta", "gamma", "theta", "vega", "rho"]
    strikes_arr = df["strike"].values

    heatmap_data = []
    for g in greek_list:
        col_name = f"{prefix}_{g}"
        heatmap_data.append(df[col_name].values)

    heatmap_matrix = np.array(heatmap_data)

    norm_matrix = np.zeros_like(heatmap_matrix)
    for i in range(len(greek_list)):
        row = heatmap_matrix[i]
        rmin, rmax = np.nanmin(row), np.nanmax(row)
        if rmax - rmin > 1e-10:
            norm_matrix[i] = (row - rmin) / (rmax - rmin)
        else:
            norm_matrix[i] = 0.5

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_matrix,
        x=strikes_arr,
        y=[g.capitalize() for g in greek_list],
        colorscale=[
            [0, "#1e1b4b"],
            [0.25, "#5b21b6"],
            [0.5, "#7c3aed"],
            [0.75, "#a78bfa"],
            [1, "#fbbf24"],
        ],
        text=np.round(heatmap_matrix, 4),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Strike: %{x}<br>Greek: %{y}<br>Value: %{z:.4f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Value", font=dict(color="#c4b5fd")),
            tickfont=dict(color="#94a3b8"),
        ),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Greeks Heatmap — {heatmap_type}",
        xaxis_title="Strike Price",
        yaxis_title="Greek",
        height=400,
    )

    fig.add_vline(x=atm_strike, line=dict(color=ATM_COLOR, width=2, dash="dot"))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ATM Greeks Summary", divider="gray")
    atm_data = df.loc[(df["strike"] - atm_strike).abs().idxmin()]

    g_cols = st.columns(5)
    for i, g in enumerate(greek_list):
        with g_cols[i]:
            ce_val = atm_data.get(f"ce_{g}", 0)
            pe_val = atm_data.get(f"pe_{g}", 0)
            st.metric(
                f"{g.capitalize()}",
                f"CE: {ce_val:.4f}",
                delta=f"PE: {pe_val:.4f}",
                delta_color="off",
            )

with tab4:
    st.subheader("Market Price vs Black-Scholes Theoretical Price", divider="gray")

    pricing_col1, pricing_col2 = st.columns(2)

    with pricing_col1:
        fig_ce = go.Figure()

        fig_ce.add_trace(go.Bar(
            x=df["strike"], y=df["ce_ltp"],
            name="Market Price",
            marker_color=CE_COLOR,
            opacity=0.7,
            hovertemplate="Strike: %{x}<br>Market: ₹%{y:.2f}<extra></extra>",
        ))
        fig_ce.add_trace(go.Scatter(
            x=df["strike"], y=df["bs_ce_price"],
            name="BS Price",
            mode="lines+markers",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=5),
            hovertemplate="Strike: %{x}<br>BS: ₹%{y:.2f}<extra></extra>",
        ))
        fig_ce.add_vline(x=atm_strike, line=dict(color=ATM_COLOR, width=1, dash="dot"))

        fig_ce.update_layout(
            **PLOTLY_LAYOUT,
            title="Call Options (CE)",
            xaxis_title="Strike", yaxis_title="Price (₹)",
            height=380, barmode="overlay",
        )
        st.plotly_chart(fig_ce, use_container_width=True)

    with pricing_col2:
        fig_pe = go.Figure()

        fig_pe.add_trace(go.Bar(
            x=df["strike"], y=df["pe_ltp"],
            name="Market Price",
            marker_color=PE_COLOR,
            opacity=0.7,
            hovertemplate="Strike: %{x}<br>Market: ₹%{y:.2f}<extra></extra>",
        ))
        fig_pe.add_trace(go.Scatter(
            x=df["strike"], y=df["bs_pe_price"],
            name="BS Price",
            mode="lines+markers",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=5),
            hovertemplate="Strike: %{x}<br>BS: ₹%{y:.2f}<extra></extra>",
        ))
        fig_pe.add_vline(x=atm_strike, line=dict(color=ATM_COLOR, width=1, dash="dot"))

        fig_pe.update_layout(
            **PLOTLY_LAYOUT,
            title="Put Options (PE)",
            xaxis_title="Strike", yaxis_title="Price (₹)",
            height=380, barmode="overlay",
        )
        st.plotly_chart(fig_pe, use_container_width=True)

    st.subheader("Implied Volatility Smile / Skew", divider="gray")

    fig_iv = go.Figure()

    valid_ce = df[df["ce_iv"] > 0.01]
    valid_pe = df[df["pe_iv"] > 0.01]

    fig_iv.add_trace(go.Scatter(
        x=valid_ce["strike"], y=valid_ce["ce_iv"] * 100,
        name="CE IV",
        mode="lines+markers",
        line=dict(color=CE_COLOR, width=3, shape="spline"),
        marker=dict(size=6, color=CE_COLOR, line=dict(width=1, color="#1e1b4b")),
        fill="tozeroy",
        fillcolor="rgba(167,139,250,0.06)",
        hovertemplate="Strike: %{x}<br>CE IV: %{y:.2f}%<extra></extra>",
    ))

    fig_iv.add_trace(go.Scatter(
        x=valid_pe["strike"], y=valid_pe["pe_iv"] * 100,
        name="PE IV",
        mode="lines+markers",
        line=dict(color=PE_COLOR, width=3, shape="spline"),
        marker=dict(size=6, color=PE_COLOR, line=dict(width=1, color="#1e1b4b")),
        fill="tozeroy",
        fillcolor="rgba(96,165,250,0.06)",
        hovertemplate="Strike: %{x}<br>PE IV: %{y:.2f}%<extra></extra>",
    ))

    fig_iv.add_vline(
        x=atm_strike,
        line=dict(color=ATM_COLOR, width=1.5, dash="dot"),
        annotation_text="ATM",
        annotation_font_color=ACCENT,
    )

    fig_iv.update_layout(
        **PLOTLY_LAYOUT,
        title="IV Smile — Implied Volatility vs Strike",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility (%)",
        height=420,
    )

    st.plotly_chart(fig_iv, use_container_width=True)

    st.subheader("Mispricing Scatter — Market vs BS Price Deviation", divider="gray")

    fig_mis = make_subplots(rows=1, cols=2, subplot_titles=("Calls (CE)", "Puts (PE)"))

    ce_colors = np.where(df["ce_mispricing_pct"] > 0, "#f87171", "#4ade80")
    fig_mis.add_trace(go.Bar(
        x=df["strike"], y=df["ce_mispricing_pct"],
        marker_color=ce_colors,
        name="CE Misprice %",
        hovertemplate="Strike: %{x}<br>Mispricing: %{y:.2f}%<extra></extra>",
    ), row=1, col=1)

    pe_colors = np.where(df["pe_mispricing_pct"] > 0, "#f87171", "#4ade80")
    fig_mis.add_trace(go.Bar(
        x=df["strike"], y=df["pe_mispricing_pct"],
        marker_color=pe_colors,
        name="PE Misprice %",
        hovertemplate="Strike: %{x}<br>Mispricing: %{y:.2f}%<extra></extra>",
    ), row=1, col=2)

    fig_mis.update_layout(
        **PLOTLY_LAYOUT,
        title="Mispricing: Market − BS Price (% deviation)",
        height=380,
        showlegend=False,
    )
    fig_mis.update_xaxes(title_text="Strike", gridcolor="rgba(139,92,246,0.08)")
    fig_mis.update_yaxes(title_text="Mispricing (%)", gridcolor="rgba(139,92,246,0.08)")

    st.plotly_chart(fig_mis, use_container_width=True)

    st.caption(" Positive = Market price > BS (overpriced) ·  Negative = Market price < BS (underpriced)")

with tab5:
    st.subheader("Interactive Strategy Builder", divider="gray")

    if "strategy_legs" not in st.session_state:
        st.session_state.strategy_legs = [{
            "id": 1, "action": "Buy", "type": "CE", "strike": atm_strike, "lots": 1, "premium": 100.0
        }]
    
    available_strikes = sorted(df["strike"].unique())

    def add_leg():
        new_id = 1 if not st.session_state.strategy_legs else max(leg["id"] for leg in st.session_state.strategy_legs) + 1
        st.session_state.strategy_legs.append({
            "id": new_id, "action": "Buy", "type": "CE", "strike": atm_strike, "lots": 1, "premium": 100.0
        })

    def remove_leg(leg_id):
        st.session_state.strategy_legs = [leg for leg in st.session_state.strategy_legs if leg["id"] != leg_id]

    st.markdown("**Configure Legs**")
    
    updated_legs = []
    
    headers = st.columns([1.5, 1.5, 2, 2, 2, 1])
    headers[0].caption("Action")
    headers[1].caption("Type")
    headers[2].caption("Strike Price")
    headers[3].caption("Lots")
    headers[4].caption("Premium (₹)")
    
    for leg in st.session_state.strategy_legs:
        c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1.5, 2, 2, 2, 1])
        with c1:
            action = st.selectbox("Action", ["Buy", "Sell"], index=0 if leg["action"] == "Buy" else 1, key=f"action_{leg['id']}", label_visibility="collapsed")
        with c2:
            op_type = st.selectbox("Type", ["CE", "PE"], index=0 if leg["type"] == "CE" else 1, key=f"type_{leg['id']}", label_visibility="collapsed")
        with c3:
            s_idx = available_strikes.index(leg["strike"]) if leg["strike"] in available_strikes else 0
            strike = st.selectbox("Strike", available_strikes, index=s_idx, key=f"strike_{leg['id']}", label_visibility="collapsed")
        
        default_prem = 100.0
        strike_row = df[df["strike"] == strike]
        if not strike_row.empty:
            default_prem = float(strike_row["ce_ltp"].values[0]) if op_type == "CE" else float(strike_row["pe_ltp"].values[0])

        with c4:
            lots = st.number_input("Lots", min_value=1, value=leg["lots"], key=f"lots_{leg['id']}", label_visibility="collapsed")
        with c5:
            bound_key_prem = f"prem_{leg['id']}"
            if bound_key_prem not in st.session_state:
                st.session_state[bound_key_prem] = default_prem
            premium = st.number_input("Premium", min_value=0.01, step=0.5, key=bound_key_prem, label_visibility="collapsed")
        with c6:
            st.button("❌", key=f"rm_{leg['id']}", on_click=remove_leg, args=(leg['id'],))
        
        updated_legs.append({
            "id": leg["id"],
            "action": action,
            "type": op_type,
            "strike": strike,
            "lots": lots,
            "premium": premium
        })
    
    st.session_state.strategy_legs = updated_legs

    col_btn1, col_btn2 = st.columns([2, 10])
    with col_btn1:
        st.button("+ Add Leg", on_click=add_leg)

    st.markdown("---")

    if not st.session_state.strategy_legs:
        st.info("Please add at least one leg to view the payoff diagram.")
    else:
        strikes = [l["strike"] for l in st.session_state.strategy_legs]
        min_strike, max_strike = min(strikes), max(strikes)
        
        price_range = np.linspace(min_strike * 0.85, max_strike * 1.15, 300)
        net_payoff = np.zeros_like(price_range)
        net_premium_paid = 0.0
        
        for leg in st.session_state.strategy_legs:
            is_call = leg["type"] == "CE"
            is_long = leg["action"] == "Buy"
            k = leg["strike"]
            p = leg["premium"]
            qty = leg["lots"] * lot_size
            
            if is_call:
                intrinsic = np.maximum(price_range - k, 0)
            else:
                intrinsic = np.maximum(k - price_range, 0)
            
            if is_long:
                payoff = (intrinsic - p) * qty
                net_premium_paid += p * qty
            else:
                payoff = (p - intrinsic) * qty
                net_premium_paid -= p * qty
            
            net_payoff += payoff

        max_profit_val = np.max(net_payoff)
        max_loss_val = np.min(net_payoff)
        
        slope_start = net_payoff[1] - net_payoff[0]
        slope_end = net_payoff[-1] - net_payoff[-2]
        
        max_profit = f"₹{format_number(max_profit_val)}"
        if slope_start < -0.01 or slope_end > 0.01:
            max_profit = "Unlimited"
            
        max_loss = f"₹{format_number(max_loss_val)}"
        if slope_start > 0.01 or slope_end < -0.01:
            max_loss = "Unlimited"

        crossings = np.where(np.diff(np.signbit(net_payoff)))[0]
        breakevens = [price_range[i] for i in crossings]

        fig_payoff = go.Figure()

        profit_mask = net_payoff >= 0
        loss_mask = net_payoff < 0

        for i, leg in enumerate(st.session_state.strategy_legs):
            leg_name = f"{leg['action']} {leg['lots']}x {leg['strike']} {leg['type']}"
            is_call = leg["type"] == "CE"
            is_long = leg["action"] == "Buy"
            k = leg["strike"]
            p = leg["premium"]
            qty = leg["lots"] * lot_size
            
            intrinsic = np.maximum(price_range - k, 0) if is_call else np.maximum(k - price_range, 0)
            leg_payoff = (intrinsic - p) * qty if is_long else (p - intrinsic) * qty
            
            fig_payoff.add_trace(go.Scatter(
                x=price_range, y=leg_payoff,
                mode="lines",
                name=leg_name,
                line=dict(color="rgba(148,163,184,0.4)", width=1, dash="dash"),
                hovertemplate=f"{leg_name}: ₹%{{y:,.0f}}<extra></extra>",
            ))

        fig_payoff.add_trace(go.Scatter(
            x=price_range, y=net_payoff,
            mode="lines",
            name="Net P&L",
            line=dict(color="#60a5fa", width=3),
            hovertemplate="Spot: ₹%{x:.0f} | Net P&L: ₹%{y:,.0f}<extra></extra>",
        ))

        fig_payoff.add_trace(go.Scatter(
            x=price_range[profit_mask], y=net_payoff[profit_mask],
            fill="tozeroy", fillcolor="rgba(34,197,94,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        
        fig_payoff.add_trace(go.Scatter(
            x=price_range[loss_mask], y=net_payoff[loss_mask],
            fill="tozeroy", fillcolor="rgba(239,68,68,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))

        fig_payoff.add_hline(y=0, line=dict(color="rgba(148,163,184,0.3)", width=1))
        fig_payoff.add_vline(x=spot_price, line=dict(color="#06b6d4", width=1.5, dash="dot"), annotation_text=f"Spot: ₹{spot_price:,.0f}", annotation_font_color="#06b6d4")

        for be in breakevens:
            fig_payoff.add_vline(x=be, line=dict(color="#f59e0b", width=1.5, dash="dash"), annotation_text=f"BE: ₹{be:,.0f}", annotation_font_color="#f59e0b", annotation_font_size=11)

        fig_payoff.update_layout(
            **PLOTLY_LAYOUT,
            title="Combined Strategy Net Payoff",
            xaxis_title="Spot Price at Expiry (₹)",
            yaxis_title=f"Net Profit / Loss (₹)",
            height=480,
        )

        st.plotly_chart(fig_payoff, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            be_str = ", ".join([f"₹{b:,.0f}" for b in breakevens][:3]) if breakevens else "None"
            help_str = ", ".join([f"₹{b:,.2f}" for b in breakevens]) if breakevens else None
            st.metric("Breakevens", be_str, help=help_str)
        with m2:
            st.metric("Max Profit", max_profit)
        with m3:
            st.metric("Max Loss", max_loss)
        with m4:
            direction = "Debit" if net_premium_paid > 0 else "Credit"
            st.metric(f"Net Premium ({direction})", f"₹{format_number(abs(net_premium_paid))}")

st.markdown("---")
