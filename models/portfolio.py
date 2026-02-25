"""
portfolio.py
────────────
Builds a synthetic credit portfolio from spread data.

Instruments:
  1. IG Corporate Bond   – DV01-based P&L from IG OAS changes
  2. HY Corporate Bond   – DV01-based P&L from HY OAS changes
  3. IG CDS (Protection) – spread tightening is a LOSS for protection buyer
  4. HY CDS (Protection) – same direction as bonds (positive carry for seller)

P&L approximation for bond:
  ΔP ≈ −Modified_Duration × ΔSpread × Notional
  (spread in decimal, e.g. 1 bps = 0.0001)

P&L approximation for CDS protection seller (long credit risk):
  ΔP ≈ −CS01 × ΔSpread × Notional
  CS01 ≈ duration × 0.0001  (per 1bp)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CreditPortfolio:
    spread_df:      pd.DataFrame       # raw spreads (bps)
    spread_returns: pd.DataFrame       # daily spread changes (bps)
    weights:        np.ndarray         # allocation weights [ig_bond, hy_bond, ig_cds, hy_cds]
    notional:       float              # total portfolio notional ($)
    pnl_series:     pd.Series = field(default=None, repr=False)
    instrument_names: list = field(default_factory=list)

    # Market risk sensitivities
    DURATIONS = {
        "IG_OAS": 7.0,   # ~7yr IG bond modified duration
        "HY_OAS": 4.5,   # ~4.5yr HY bond (shorter maturities)
        "IG_CDS": 5.0,   # IG 5yr CDS CS01
        "HY_CDS": 4.0,   # HY CDS CS01
    }


def build_portfolio(spread_df: pd.DataFrame,
                    weights:   np.ndarray,
                    notional:  float) -> CreditPortfolio:
    """
    Constructs a CreditPortfolio.

    Parameters
    ----------
    spread_df : DataFrame with columns [IG_OAS, HY_OAS, IG_CDS, HY_CDS] in bps
    weights   : array-like of 4 weights (must sum to ~1)
    notional  : total portfolio value in dollars
    """
    col_map = {
        0: "IG_OAS",
        1: "HY_OAS",
        2: "IG_CDS",
        3: "HY_CDS",
    }
    instrument_names = ["IG Corp Bond", "HY Corp Bond", "IG CDS", "HY CDS"]

    # Daily spread changes in bps
    spread_changes = spread_df.diff().dropna()

    # Notional per instrument
    notionals = weights * notional  # [$]

    durations = np.array([
        CreditPortfolio.DURATIONS["IG_OAS"],
        CreditPortfolio.DURATIONS["HY_OAS"],
        CreditPortfolio.DURATIONS["IG_CDS"],
        CreditPortfolio.DURATIONS["HY_CDS"],
    ])

    # ΔP&L per bps per dollar = −duration × (1/10000) × notional
    # CDS protection SELLER has same sign as bond holder (long credit risk)
    dv01 = -durations * (1 / 10_000) * notionals  # $ per 1bp adverse move

    # Daily P&L: matrix multiply (T×4) × (4,) → T-length series
    spread_change_arr = spread_changes[["IG_OAS", "HY_OAS", "IG_CDS", "HY_CDS"]].values
    pnl_matrix = spread_change_arr * dv01  # elementwise, then sum
    portfolio_pnl = pd.Series(pnl_matrix.sum(axis=1), index=spread_changes.index)

    # Align spread_df to changes index
    spread_df_aligned = spread_df.loc[spread_changes.index]

    return CreditPortfolio(
        spread_df=spread_df_aligned,
        spread_returns=spread_changes,
        weights=weights,
        notional=notional,
        pnl_series=portfolio_pnl,
        instrument_names=instrument_names,
    )
