# Credit VaR Engine

> A professional-grade **Value-at-Risk calculator** for a credit portfolio of bonds and CDS —
> built to mirror the toolset used by Market Risk Analytics teams at firms like Morgan Stanley.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mugeniai05-credit-var-engine.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Live Dashboard

** [Click here to open the live app](https://credit-var-engine-8nlxf4appjqnvurtsuiahud.streamlit.app/)**

No installation required — runs entirely in your browser.

---

## Results

### KPI Dashboard — $100M Portfolio at 99% Confidence, 10-Day Horizon
| Metric | Value | Notes |
|---|---|---|
| Historical VaR | **$5.44M** | 5.44% of NAV |
| Monte Carlo VaR | **$8.58M** | 8.58% of NAV — Cholesky correlated shocks |
| Stressed VaR | **$4.90M** | Worst 3-month window recalibration |
| IRC Proxy | **$27.33M** | 99.9% 1-year jump-to-default charge |
| Exceedances | **0 / 250** | 0.0% — Basel Green Zone ✅ |
| Basel Zone | **Green** | Model acceptable — no capital add-on |

### Spread Return Correlation Matrix
| | IG_OAS | HY_OAS | IG_CDS | HY_CDS |
|---|---|---|---|---|
| **IG_OAS** | 1.00 | 0.43 | 0.92 | 0.41 |
| **HY_OAS** | 0.43 | 1.00 | 0.38 | 0.96 |
| **IG_CDS** | 0.92 | 0.38 | 1.00 | 0.36 |
| **HY_CDS** | 0.41 | 0.96 | 0.36 | 1.00 |

> IG_OAS↔IG_CDS ~0.92 reflects CDS-bond basis risk. Cross-asset IG↔HY ~0.43 captures the divergence between investment-grade and high-yield credit cycles.

---

## How to Use the Dashboard

1. **Open the app** at the live link above (or run locally — see Setup below)
2. **Configure your model** in the left sidebar:
   - Set **Confidence Level** (95%, 99%, or 99.9%)
   - Set **Holding Period** (1-day for internal monitoring, 10-day for regulatory capital)
   - Adjust **Historical Lookback** — more history captures more stress events
   - Choose **MC Simulations** — higher = more precise tail estimates
3. **Set portfolio weights** — allocate across IG Bonds, HY Bonds, IG CDS, HY CDS (must sum to 100%)
4. **Set portfolio value** in $M
5. Click **▶ Run VaR Engine**
6. Explore the output panels:
   - **KPI row** — key risk metrics at a glance
   - **P&L distributions** — historical and MC histograms with VaR thresholds marked
   - **Credit spread history** — IG/HY OAS and CDS spreads over time
   - **Backtesting chart** — daily P&L vs VaR threshold with exceedance markers
   - **VaR term structure** — how VaR scales across holding periods (1→20 days)
   - **Confidence sensitivity** — VaR across 90%→99.9% confidence levels
   - **Correlation heatmap** — spread return correlations used in the MC model
   - **Backtesting detail table** — expandable daily P&L log

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/MugeniAI05/credit-var-engine.git
cd credit-var-engine

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch dashboard
streamlit run app.py
```

### Optional: Real FRED Data
By default the engine uses realistic synthetic data. To use live ICE BofA spread data:

1. Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html
2. Set the environment variable:
```bash
export FRED_API_KEY="your_key_here"   # macOS/Linux
set FRED_API_KEY=your_key_here        # Windows
```

---

## Project Structure

```
credit-var-engine/
├── app.py                    # Streamlit dashboard (main entry point)
├── requirements.txt
├── data/
│   └── fetch_data.py         # FRED fetcher + synthetic fallback (OU process)
└── models/
    ├── portfolio.py          # DV01-based P&L model (bonds + CDS)
    ├── historical_var.py     # Historical Simulation + Stressed VaR + IRC proxy
    ├── monte_carlo_var.py    # MC VaR via Cholesky decomposition + FRTB ES
    └── backtesting.py        # Walk-forward backtest + Basel traffic light
```

---

## Methodology

### Instruments
| Instrument | Risk Driver | Duration | Position |
|---|---|---|---|
| IG Corporate Bond | IG OAS (BAMLC0A0CM) | 7.0yr | Long credit risk |
| HY Corporate Bond | HY OAS (BAMLH0A0HYM2) | 4.5yr | Long credit risk |
| IG CDS (seller) | IG CDS spread | 5.0yr | Long credit risk |
| HY CDS (seller) | HY CDS spread | 4.0yr | Long credit risk |

### P&L Approximation
```
ΔP ≈ −Modified Duration × ΔSpread (bps) × (1/10,000) × Notional
```

### Models
**Historical Simulation VaR** — applies each day's realized spread shock to today's portfolio. VaR(h) = VaR(1) × √h.

**Monte Carlo VaR** — estimates the covariance matrix of spread returns, Cholesky-decomposes it to generate N correlated scenarios, prices each scenario via DV01. Also computes FRTB Expected Shortfall.

**Stressed VaR** — identifies the worst rolling 63-day window and recalibrates VaR on that sub-sample, per Basel 2.5 requirements.

**IRC Proxy** — 99.9% VaR scaled to a 1-year horizon, approximating jump-to-default and rating migration risk.

**Backtesting** — walk-forward expanding-window backtest. Counts exceedances over the last 250 trading days and classifies into Basel traffic-light zones per BCBS 352.

---

## Relevance to Market Risk Analytics

| This Project | Industry Requirement |
|---|---|
| Historical Simulation & MC VaR | VaR, Stressed VaR models |
| Credit spread P&L via DV01 | Bonds, CDS credit products |
| Walk-forward backtesting | Model performance monitoring |
| Basel traffic-light (BCBS 352) | Regulatory compliance |
| Stressed VaR / IRC proxy | Basel 2.5 capital requirements |
| Cholesky correlation modelling | Monte Carlo risk engine |
| FRTB Expected Shortfall | FRTB IMA requirements |
| Python + Streamlit | Production-ready implementation |

---

## Data Sources
- **FRED** (Federal Reserve Economic Data): `BAMLC0A0CM` (IG OAS), `BAMLH0A0HYM2` (HY OAS)
- CDS spreads derived via basis adjustment + idiosyncratic noise from cash bond OAS
