"""
historical_var.py
─────────────────
Historical Simulation Value-at-Risk for the credit portfolio.

Method:
  1. Use the last N days of observed spread changes (the "historical window")
  2. Re-apply each day's spread shock to today's portfolio (full revaluation)
  3. Sort the resulting P&L vector; VaR = −percentile at (1−confidence)
  4. Scale 1-day VaR to h-day: VaR(h) = VaR(1) × √h  (square-root-of-time rule)

Stressed VaR:
  Re-run the same method but using only the stress window (2008 GFC-like episode
  identified as the 10th-percentile rolling-average-spread period).

IRC Proxy:
  Incremental Risk Charge proxy using 99.9% VaR scaled to 1-year with
  √(252/hp) adjustment.
"""

import numpy as np
import pandas as pd
from .portfolio import CreditPortfolio


class HistoricalVaR:
    def __init__(self,
                 portfolio: CreditPortfolio,
                 confidence: float = 0.99,
                 hp: int = 10):
        self.port       = portfolio
        self.confidence = confidence
        self.hp         = hp

    def compute(self) -> tuple[float, np.ndarray]:
        """
        Returns
        -------
        var_value : float  – VaR in dollars (positive number = loss)
        pnl       : ndarray – 1-day P&L scenarios ($)
        """
        pnl_1d = self.port.pnl_series.values
        var_1d = np.percentile(pnl_1d, (1 - self.confidence) * 100)
        var_scaled = -var_1d * np.sqrt(self.hp)
        return max(var_scaled, 0), pnl_1d

    def stressed_var(self) -> float:
        """
        Identifies the worst 3-month rolling window in the historical data
        and recalculates VaR using only that sub-sample.
        """
        pnl = self.port.pnl_series

        # Identify worst window by cumulative P&L
        window = 63  # ~3 months
        if len(pnl) <= window:
            stressed_pnl = pnl.values
        else:
            rolling_sum = pnl.rolling(window).sum()
            worst_end   = rolling_sum.idxmin()
            worst_end_idx = pnl.index.get_loc(worst_end)
            start_idx = max(0, worst_end_idx - window)
            stressed_pnl = pnl.iloc[start_idx:worst_end_idx + 1].values

        var_1d = np.percentile(stressed_pnl, (1 - self.confidence) * 100)
        return max(-var_1d * np.sqrt(self.hp), 0)

    def irc_proxy(self) -> float:
        """
        IRC Proxy: 99.9% 1-year VaR.
        Uses the HS distribution scaled to 252 days.
        """
        pnl_1d = self.port.pnl_series.values
        var_1d = np.percentile(pnl_1d, 0.1)   # 99.9th percentile loss
        return max(-var_1d * np.sqrt(252), 0)
