"""
monte_carlo_var.py
──────────────────
Monte Carlo Value-at-Risk for the credit portfolio.

Method:
  1. Estimate the covariance matrix of daily spread returns from history
  2. Cholesky-decompose to generate correlated normal spread shocks
  3. Simulate N joint spread scenarios
  4. Compute portfolio P&L for each scenario via DV01 approximation
  5. VaR = negative of (1−confidence) percentile of P&L distribution

The Cholesky method correctly captures cross-spread correlations
(e.g., IG/HY co-movement) which is critical for credit books.
"""

import numpy as np
import pandas as pd
from .portfolio import CreditPortfolio


class MonteCarloVaR:
    def __init__(self,
                 portfolio: CreditPortfolio,
                 confidence: float = 0.99,
                 hp: int = 10,
                 n_sims: int = 10_000,
                 random_seed: int = 42):
        self.port       = portfolio
        self.confidence = confidence
        self.hp         = hp
        self.n_sims     = n_sims
        self.rng        = np.random.default_rng(random_seed)

    def compute(self) -> tuple[float, np.ndarray]:
        """
        Returns
        -------
        var_value : float   – VaR in dollars (positive = loss)
        pnl       : ndarray – simulated 1-day P&L scenarios
        """
        returns = self.port.spread_returns[["IG_OAS", "HY_OAS", "IG_CDS", "HY_CDS"]].values

        mu  = returns.mean(axis=0)               # (4,)
        cov = np.cov(returns.T)                  # (4×4)

        # Cholesky decomposition: cov = L @ L.T
        # Regularise by adding small diagonal to ensure positive-definiteness
        cov_reg = cov + np.eye(4) * 1e-6
        L = np.linalg.cholesky(cov_reg)

        # Simulate correlated spread shocks
        z = self.rng.standard_normal((self.n_sims, 4))   # uncorrelated
        spread_shocks = z @ L.T + mu                      # correlated (bps)

        # DV01 vector: $ loss per 1bp widening per instrument
        weights  = self.port.weights
        notional = self.port.notional
        notionals = weights * notional

        durations = np.array([7.0, 4.5, 5.0, 4.0])
        dv01 = -durations * (1 / 10_000) * notionals    # (4,)

        # Portfolio P&L for each simulation
        pnl_1d = spread_shocks @ dv01                   # (N,)

        # Scale to holding period
        pnl_hp = pnl_1d * np.sqrt(self.hp)

        var_value = -np.percentile(pnl_hp, (1 - self.confidence) * 100)
        return max(var_value, 0), pnl_hp

    def expected_shortfall(self) -> float:
        """
        Expected Shortfall (CVaR / ES) – average loss beyond VaR threshold.
        Required under FRTB as the primary risk measure (replaces VaR).
        """
        _, pnl = self.compute()
        threshold = np.percentile(pnl, (1 - self.confidence) * 100)
        tail_losses = pnl[pnl <= threshold]
        return -tail_losses.mean() if len(tail_losses) else 0.0
