"""
backtesting.py
──────────────
VaR Backtesting – Basel Committee Traffic Light Approach (Basel 2.5 / BCBS 352).

The backtest compares each day's actual P&L against the prior day's VaR estimate.
An "exceedance" occurs when the actual loss exceeds the VaR threshold.

Basel Traffic Light Zones (for 250-day backtest at 99% confidence):
  Green  Zone: 0–4  exceedances  → model acceptable
  Yellow Zone: 5–9  exceedances  → increased scrutiny / capital multiplier
  Red    Zone: 10+  exceedances  → model likely invalid, regulatory action

Implementation:
  - Rolling VaR is estimated using a walk-forward expanding window
  - Exceedance = actual_pnl < −VaR
"""

import numpy as np
import pandas as pd
from .portfolio import CreditPortfolio
from .historical_var import HistoricalVaR


def run_backtest(portfolio: CreditPortfolio,
                 confidence: float = 0.99,
                 hp: int = 1,
                 min_window: int = 250) -> dict:
    """
    Walk-forward backtesting.

    Parameters
    ----------
    portfolio   : CreditPortfolio
    confidence  : VaR confidence level
    hp          : holding period (days); default 1 for daily backtest
    min_window  : minimum history before generating VaR estimates

    Returns
    -------
    dict with keys:
        exceedances, total_obs, exc_pct, zone, zone_desc, detail_df
    """
    pnl = portfolio.pnl_series

    if len(pnl) < min_window + 20:
        min_window = max(50, len(pnl) // 3)

    results = []
    indices = []

    for i in range(min_window, len(pnl)):
        window_pnl = pnl.iloc[:i].values

        # 1-day VaR using historical simulation on expanding window
        var_1d = -np.percentile(window_pnl, (1 - confidence) * 100)

        actual_pnl  = pnl.iloc[i]
        exceedance  = actual_pnl < -var_1d

        results.append({
            "pnl":        actual_pnl,
            "var":        var_1d,
            "exceedance": exceedance,
        })
        indices.append(pnl.index[i])

    detail_df = pd.DataFrame(results, index=indices)

    # Use last 250 obs for Basel zone (or all if fewer)
    backtest_window = detail_df.tail(250)
    n_exc  = backtest_window["exceedance"].sum()
    n_obs  = len(backtest_window)
    exc_pct = n_exc / n_obs * 100 if n_obs else 0

    zone, zone_desc = _basel_zone(n_exc)

    return {
        "exceedances": int(n_exc),
        "total_obs":   n_obs,
        "exc_pct":     exc_pct,
        "zone":        zone,
        "zone_desc":   zone_desc,
        "detail_df":   detail_df,
    }


def _basel_zone(n_exc: int) -> tuple[str, str]:
    """Returns Basel traffic-light zone and description."""
    if n_exc <= 4:
        return "Green",  "Model acceptable — no capital add-on"
    elif n_exc <= 9:
        return "Yellow", f"Increased scrutiny — capital multiplier may apply (+{n_exc - 4} add-on)"
    else:
        return "Red",    "Model likely invalid — regulatory review required"
