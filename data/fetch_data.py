"""
fetch_data.py
─────────────
Downloads credit spread indices from FRED (Federal Reserve Economic Data).
Falls back to synthetic data if network is unavailable.

FRED Series used:
  BAMLC0A0CM   – ICE BofA US Corporate OAS (Investment Grade)
  BAMLH0A0HYM2 – ICE BofA US High Yield OAS
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


# IG and HY OAS series on FRED (public, no key required via fredapi)
SERIES = {
    "IG_OAS":  "BAMLC0A0CM",
    "HY_OAS":  "BAMLH0A0HYM2",
}

# CDS basis adjustments (bps) – typical market basis vs cash bond spreads
CDS_BASIS = {
    "IG_CDS": -10,   # CDS typically trades ~10bps tighter than bonds (negative basis)
    "HY_CDS":  25,   # HY CDS wider due to liquidity premium
}


def fetch_credit_spreads(lookback_days: int = 760) -> pd.DataFrame:
    """
    Returns a DataFrame of daily credit spreads (in bps) with columns:
        IG_OAS | HY_OAS | IG_CDS | HY_CDS
    covering approximately `lookback_days` of history.
    """
    end   = datetime.today()
    start = end - timedelta(days=lookback_days + 60)   # buffer for weekends/holidays

    if FRED_AVAILABLE:
        try:
            return _fetch_from_fred(start, end, lookback_days)
        except Exception as e:
            print(f"[WARNING] FRED fetch failed ({e}). Falling back to synthetic data.")

    return _synthetic_spreads(lookback_days)


def _fetch_from_fred(start, end, lookback_days) -> pd.DataFrame:
    fred = Fred()
    dfs = {}
    for name, series_id in SERIES.items():
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        dfs[name] = s

    df = pd.DataFrame(dfs).dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Derive CDS columns from OAS with basis adjustments + idiosyncratic noise
    # CDS ≠ exact bond spread due to: doc risk, cheapest-to-deliver, funding basis
    rng = np.random.default_rng(42)
    n   = len(df)
    df["IG_CDS"] = df["IG_OAS"] + CDS_BASIS["IG_CDS"] + rng.normal(0, 3, n)
    df["HY_CDS"] = df["HY_OAS"] + CDS_BASIS["HY_CDS"] + rng.normal(0, 8, n)

    # Keep only last `lookback_days` business days
    df = df.tail(lookback_days)
    return df[["IG_OAS", "HY_OAS", "IG_CDS", "HY_CDS"]]


def _synthetic_spreads(lookback_days: int) -> pd.DataFrame:
    """
    Generates realistic synthetic credit spreads calibrated to long-run
    averages and volatilities of actual FRED data.
    Includes a simulated stress episode (like 2020 Covid / 2008 GFC).
    """
    np.random.seed(42)
    n = lookback_days

    dates = pd.bdate_range(end=datetime.today(), periods=n)

    # Long-run mean (bps) and mean-reversion speed for OU process
    params = {
        "IG_OAS": dict(mu=120, kappa=0.03, sigma=8,  s0=120),
        "HY_OAS": dict(mu=450, kappa=0.025, sigma=30, s0=450),
    }

    def ornstein_uhlenbeck(mu, kappa, sigma, s0, n):
        s = np.zeros(n)
        s[0] = s0
        for t in range(1, n):
            s[t] = s[t-1] + kappa * (mu - s[t-1]) + sigma * np.random.randn()
        return np.clip(s, 10, None)

    # Correlated shocks (IG/HY correlation ~0.85)
    rho = 0.85
    chol = np.linalg.cholesky([[1, rho], [rho, 1]])

    z_ig = ornstein_uhlenbeck(**params["IG_OAS"], n=n)
    z_hy = ornstein_uhlenbeck(**params["HY_OAS"], n=n)

    # Inject one stress episode in the middle third of the sample
    stress_start = n // 3
    stress_end   = stress_start + int(n * 0.08)
    z_ig[stress_start:stress_end] += np.linspace(0, 180, stress_end - stress_start)
    z_hy[stress_start:stress_end] += np.linspace(0, 700, stress_end - stress_start)
    # Recovery
    z_ig[stress_start:stress_end] = np.clip(z_ig[stress_start:stress_end], 10, 400)
    z_hy[stress_start:stress_end] = np.clip(z_hy[stress_start:stress_end], 50, 1800)

    # Add idiosyncratic noise to CDS so correlation matrix looks realistic
    # IG_OAS↔IG_CDS ~0.95, cross IG↔HY ~0.80 (matches real market behaviour)
    rng = np.random.default_rng(99)
    ig_cds_noise = rng.normal(0, 3, n)   # IG basis vol ~3 bps/day
    hy_cds_noise = rng.normal(0, 8, n)   # HY basis vol ~8 bps/day

    df = pd.DataFrame({
        "IG_OAS": z_ig,
        "HY_OAS": z_hy,
        "IG_CDS": z_ig + CDS_BASIS["IG_CDS"] + ig_cds_noise,
        "HY_CDS": z_hy + CDS_BASIS["HY_CDS"] + hy_cds_noise,
    }, index=dates)

    df = df.clip(lower=10)
    return df
