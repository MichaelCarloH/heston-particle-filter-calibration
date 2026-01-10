"""
Utility functions for data loading and preprocessing.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_market_data(
    ticker: str = "^GSPC",
    start: str = "2007-01-01",
    end: str = "2025-12-31",
    risk_free_ticker: str = "^IRX"
) -> Tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    """
    Load market data from Yahoo Finance.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (default: "^GSPC" for S&P 500)
    start : str
        Start date in YYYY-MM-DD format
    end : str
        End date in YYYY-MM-DD format
    risk_free_ticker : str
        Ticker for risk-free rate proxy (default: "^IRX" for 3-month Treasury)
    
    Returns
    -------
    log_returns : pd.Series
        Log returns of the asset
    risk_free_rate : pd.Series
        Risk-free rate time series (annualized, as decimal)
    dates : pd.DatetimeIndex
        Date index aligned with log_returns
    
    Examples
    --------
    >>> log_returns, risk_free_rate, dates = load_market_data()
    >>> print(f"Loaded {len(log_returns)} observations")
    """
    # Download price data
    logger.info(f"Downloading {ticker} data from {start} to {end}")
    price_data = yf.download(ticker, start=start, end=end, progress=False)
    
    # Handle MultiIndex columns
    if isinstance(price_data.columns, pd.MultiIndex):
        price_data.columns = price_data.columns.droplevel(1)
    
    # Compute log returns
    log_returns = np.log(price_data["Close"] / price_data["Close"].shift(1)).dropna()
    log_returns.name = "log_returns"
    
    logger.info(f"Computed {len(log_returns)} log returns")
    
    # Download risk-free rate
    try:
        logger.info(f"Downloading risk-free rate from {risk_free_ticker}")
        treasury = yf.download(risk_free_ticker, start=start, end=end, progress=False)
        
        if isinstance(treasury.columns, pd.MultiIndex):
            treasury.columns = treasury.columns.droplevel(1)
        
        # Convert from percentage to decimal and align with log_returns
        risk_free_rate = (treasury["Close"] / 100.0).reindex(
            log_returns.index, method='ffill'
        )
        
        logger.info(f"Risk-free rate statistics:")
        logger.info(f"  Mean: {risk_free_rate.mean():.4f}")
        logger.info(f"  Std: {risk_free_rate.std():.4f}")
        logger.info(f"  Min: {risk_free_rate.min():.4f}, Max: {risk_free_rate.max():.4f}")
        
    except Exception as e:
        logger.warning(f"Could not download risk-free rate: {e}")
        logger.warning("Using constant r=0.04")
        risk_free_rate = pd.Series(0.04, index=log_returns.index)
    
    dates = log_returns.index
    
    return log_returns, risk_free_rate, dates


def compute_realized_volatility(
    log_returns: pd.Series,
    window: int = 21,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Compute realized volatility using rolling window.
    
    Parameters
    ----------
    log_returns : pd.Series
        Log returns time series
    window : int
        Rolling window length in days (default: 21)
    annualization_factor : int
        Factor to annualize volatility (default: 252 for daily data)
    
    Returns
    -------
    pd.Series
        Annualized realized volatility
    """
    realized_vol = (
        (log_returns**2)
        .rolling(window)
        .mean()
        .apply(np.sqrt)
        * np.sqrt(annualization_factor)
    )
    realized_vol.name = f"RealizedVol_{window}d"
    return realized_vol


def load_vix_data(
    start: str,
    end: str
) -> pd.Series:
    """
    Load VIX implied volatility data.
    
    Parameters
    ----------
    start : str
        Start date in YYYY-MM-DD format
    end : str
        End date in YYYY-MM-DD format
    
    Returns
    -------
    pd.Series
        VIX implied volatility (annualized, as decimal)
    """
    logger.info("Downloading VIX data")
    vix = yf.download("^VIX", start=start, end=end, progress=False)
    
    # Fix MultiIndex columns if needed
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)
    
    # Convert from percentage to decimal
    vix_series = (vix["Close"] / 100.0).rename("VIX_ImpliedVol")
    
    return vix_series

