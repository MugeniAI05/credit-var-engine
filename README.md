# 📊 Credit VaR Engine

A professional-grade Value-at-Risk calculator for a credit portfolio of bonds and CDS.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Optional: Real FRED Data
Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html
Then set: `export FRED_API_KEY="your_key_here"`
Without a key, realistic synthetic data is used automatically.

## Project Structure
```
credit_var_engine/
├── app.py                    # Streamlit dashboard
├── requirements.txt
├── data/
│   └── fetch_data.py         # FRED data fetcher + synthetic fallback
└── models/
    ├── portfolio.py          # DV01-based P&L model
    ├── historical_var.py     # Historical Simulation + Stressed VaR + IRC
    ├── monte_carlo_var.py    # MC VaR via Cholesky decomposition
    └── backtesting.py        # Walk-forward backtest + Basel traffic light
```

## What's Modelled
- IG/HY Corporate Bonds (OAS-driven)
- IG/HY CDS protection seller positions
- Historical Simulation VaR (configurable lookback)
- Monte Carlo VaR with correlated spread shocks
- Stressed VaR (worst 3-month window)
- IRC Proxy (99.9% 1-year VaR)
- Basel Traffic Light backtesting (BCBS 352)
- FRTB Expected Shortfall calculation
