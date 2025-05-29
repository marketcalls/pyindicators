"""MACD (Moving Average Convergence Divergence) implementation."""

import numpy as np
from numba import njit
from typing import Optional, Tuple

from pyindicators.trend.moving_averages import ema


@njit(cache=True)
def macd(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    min_periods: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of prices.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
    fast_period : int, default 12
        Fast EMA period
    slow_period : int, default 26
        Slow EMA period
    signal_period : int, default 9
        Signal line EMA period
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    tuple of np.ndarray
        MACD line, signal line, and histogram
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import macd
    >>> close = np.random.randn(100) + 100
    >>> macd_line, signal, histogram = macd(close)
    """
    if fast_period >= slow_period:
        raise ValueError("Fast period must be less than slow period")
    
    # Calculate EMAs
    fast_ema = ema(close, fast_period, min_periods)
    slow_ema = ema(close, slow_period, min_periods)
    
    # MACD line
    macd_line = fast_ema - slow_ema
    
    # Signal line (EMA of MACD)
    signal_line = ema(macd_line, signal_period, min_periods)
    
    # MACD histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


@njit(cache=True)
def macd_signal(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate only the MACD signal line.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
    fast_period : int, default 12
        Fast EMA period
    slow_period : int, default 26
        Slow EMA period
    signal_period : int, default 9
        Signal line EMA period
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        Signal line values
    """
    _, signal, _ = macd(close, fast_period, slow_period, signal_period, min_periods)
    return signal


@njit(cache=True)
def macd_histogram(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate only the MACD histogram.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
    fast_period : int, default 12
        Fast EMA period
    slow_period : int, default 26
        Slow EMA period
    signal_period : int, default 9
        Signal line EMA period
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        Histogram values
    """
    _, _, histogram = macd(close, fast_period, slow_period, signal_period, min_periods)
    return histogram