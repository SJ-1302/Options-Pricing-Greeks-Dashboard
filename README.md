# Options Pricing & Greeks Dashboard

Built a real-time options pricing dashboard for NIFTY and BANKNIFTY contracts using live NSE option chain data, implementing the Black–Scholes model with Greeks calculation to enable options trading strategy analysis and risk management for retail traders.

## Features

### Black-Scholes Pricing Engine
- Analytical closed-form European option pricing
- Put-Call Parity verification
- Newton-Raphson Implied Volatility solver

### All 5 Greeks (Properly Scaled)
| Greek | Description | Scaling |
|-------|-------------|---------|
| **Delta** | Price sensitivity to underlying | Per ₹1 move |
| **Gamma** | Rate of change of Delta | Per ₹1 move |
| **Theta** | Time decay | **Per calendar day** |
| **Vega** | Volatility sensitivity | **Per 1% IV change** |
| **Rho** | Interest rate sensitivity | **Per 1% rate change** |

### Interactive Dashboard (5 Tabs)
1. **Option Chain** — Full chain table with mispricing analysis (market vs BS price)
2. **Greeks Analysis** — Individual Greek plots (Delta, Gamma, Theta, Vega, Rho) vs strike
3. **Greeks Heatmap** — 2D heatmap across Greeks and strikes
4. **Pricing & IV** — Market vs BS price comparison + IV Smile/Skew curve
5. **Payoff Diagram** — Interactive P&L chart with configurable position

# Live Project Demo

## Dashboard Overview

<p align="center">
  <img src="images/Screen Recording 2026-03-28 151126 - frame at 0m51s.jpg" width="800"/>
</p>

---

## Key Functionalities

### Option Chain Data

<p align="center">
  <img src="images/Screen Recording 2026-03-28 151126 - frame at 0m55s.jpg" width="700"/>
</p>

---

### Greeks Calculation

<p align="center">
  <img src="images/Screen Recording 2026-03-28 151126 - frame at 1m1s.jpg" width="700"/>
</p>

---

### Strategy Analysis

<p align="center">
  <img src="images/Screen Recording 2026-03-28 151126 - frame at 1m13s.jpg" width="700"/>
</p>

---

### Risk Visualization

<p align="center">
  <img src="images/Screen Recording 2026-03-28 151126 - frame at 1m16s.jpg" width="700"/>
</p>

---

## Additional Views

<p align="center">
  <img src="images/Screen Recording 2026-03-28 151126 - frame at 1m18s.jpg" width="45%"/>
  <img src="images/Screen Recording 2026-03-28 151126 - frame at 1m22s.jpg" width="45%"/>
</p>

<p align="center">
  <img src="images/Screen Recording 2026-03-28 151126 - frame at 1m25s.jpg" width="45%"/>
  <img src="images/Screen Recording 2026-03-28 151126 - frame at 1m27s.jpg" width="45%"/>
</p>

<p align="center">
  <img src="images/Screen Recording 2026-03-28 151126 - frame at 1m38s.jpg" width="45%"/>
  <img src="images/Screen Recording 2026-03-28 151126 - frame at 2m13s.jpg" width="45%"/>
</p>

---


### Data Pipeline
- **Live API**: Fetches from NSE option chain endpoint
- **Mock Fallback**: Realistic simulated data with proper IV smile, OI distribution, and volume
- **Seamless Switch**: Same schema for both — downstream code is source-agnostic
- **Caching**: 5-minute Streamlit cache for performance

### Instruments Supported
- **NIFTY 50** (Lot size: 25, Strike gap: ₹50)
- **Bank NIFTY** (Lot size: 15, Strike gap: ₹100)

---

## Architecture

```
Options Pricing & Greeks Dashboard/
├── data/
│   ├── __init__.py            # Package exports
│   ├── nse_fetcher.py         # NSE API + mock data fallback
│   └── option_chain.py        # Data cleaning, filtering, mispricing
├── models/
│   ├── __init__.py            # Package exports
│   ├── black_scholes.py       # BS pricing + IV solver
│   └── greeks.py              # All 5 Greeks (vectorized)
├── utils/
│   ├── __init__.py            # Package exports
│   └── helpers.py             # Constants, formatting, validation
├── app.py                     # Main Streamlit dashboard
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

### Data Flow
```
NSE API / Mock Generator
        ↓
  nse_fetcher.py (raw DataFrame)
        ↓
  option_chain.py (enriched + mispricing)
        ↓
  ┌─────┴──────┐
  │  models/   │
  │ BS + Greeks │
  └─────┬──────┘
        ↓
     app.py (Streamlit + Plotly)
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
python -m streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

### 3. Use the Controls
- **Instrument**: Switch between NIFTY and BANKNIFTY
- **Risk-Free Rate**: Adjust from 1% to 15% (default: 6.5%)
- **Strike Range**: Filter strikes within ±2% to ±20% of spot
- **Option Type**: View CE, PE, or both
- **Expiry Date**: Select from available expiry dates
- **Refresh**: Re-fetch data and clear cache

---

## Black-Scholes Formulas

### Option Pricing
```
Call: C = S·N(d₁) - K·e^(-rT)·N(d₂)
Put:  P = K·e^(-rT)·N(-d₂) - S·N(-d₁)

Where:
  d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
  d₂ = d₁ - σ·√T
  N(·) = Standard normal CDF
```

### Greeks
```
Delta (CE) = N(d₁)           Delta (PE) = N(d₁) - 1
Gamma      = n(d₁) / (S·σ·√T)
Theta (CE) = [-S·n(d₁)·σ/(2√T) - r·K·e^(-rT)·N(d₂)]  / 365
Theta (PE) = [-S·n(d₁)·σ/(2√T) + r·K·e^(-rT)·N(-d₂)] / 365
Vega       = S·n(d₁)·√T / 100
Rho (CE)   = K·T·e^(-rT)·N(d₂)  / 100
Rho (PE)   = -K·T·e^(-rT)·N(-d₂) / 100
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| UI Framework | Streamlit |
| Charting | Plotly |
| Numerics | NumPy, SciPy |
| Data | Pandas |
| HTTP | Requests |

---

## Key Design Decisions

1. **Vectorized computation**: All BS and Greeks calculations use NumPy arrays for chain-wide computation in a single call
2. **Modular structure**: Each module is independently testable and reusable
3. **Seamless data fallback**: Live/mock data share the same DataFrame schema
4. **Proper Greek scaling**: Theta per day (not per year), Vega per 1% (not per 100%), matching industry convention
5. **Mispricing metric**: Market price vs BS theoretical price, both absolute and percentage, to identify over/underpriced options
6. **IV smile visualization**: CE and PE implied volatility plotted against strike price to reveal skew patterns

---

