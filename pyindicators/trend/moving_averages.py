"""Moving average implementations optimized with Numba."""

import numpy as np
from numba import njit, prange
from typing import Optional

from pyindicators.core.utils import validate_input, ewma_vectorized


@njit(cache=True, parallel=True)
def sma(
    close: np.ndarray,
    period: int = 20,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA).
    
    SMA is the unweighted mean of the previous n data points.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices or values to average
    period : int, default 20
        Number of periods for the moving average
    min_periods : int, optional
        Minimum periods required for calculation
        
    Returns
    -------
    np.ndarray
        SMA values
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import sma
    >>> close = np.random.randn(100) + 100
    >>> sma_values = sma(close, period=20)
    """
    close, min_periods = validate_input(close, period, min_periods)
    n = len(close)
    
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # Use cumulative sum for efficiency
    cumsum = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + close[i]
    
    # Calculate SMA using parallel processing
    for i in prange(period - 1, n):
        result[i] = (cumsum[i + 1] - cumsum[i - period + 1]) / period
    
    return result


@njit(cache=True)
def ema(
    close: np.ndarray,
    period: int = 20,
    min_periods: Optional[int] = None,
    adjust: bool = True
) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).
    
    EMA gives more weight to recent prices, making it more responsive
    to new information compared to SMA.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices or values to average
    period : int, default 20
        Number of periods for the moving average
    min_periods : int, optional
        Minimum periods required for calculation
    adjust : bool, default True
        If True, use adjust method (divide by decaying adjustment factor)
        If False, use recursive method
        
    Returns
    -------
    np.ndarray
        EMA values
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import ema
    >>> close = np.random.randn(100) + 100
    >>> ema_values = ema(close, period=20)
    """
    close, min_periods = validate_input(close, period, min_periods)
    
    # Calculate smoothing factor (alpha)
    alpha = 2.0 / (period + 1.0)
    
    return ewma_vectorized(close, alpha, min_periods)


@njit(cache=True)
def wma(
    close: np.ndarray,
    period: int = 20,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Weighted Moving Average (WMA).
    
    WMA assigns linearly decreasing weights to older data points.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices or values to average
    period : int, default 20
        Number of periods for the moving average
    min_periods : int, optional
        Minimum periods required for calculation
        
    Returns
    -------
    np.ndarray
        WMA values
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import wma
    >>> close = np.random.randn(100) + 100
    >>> wma_values = wma(close, period=20)
    """
    close, min_periods = validate_input(close, period, min_periods)
    n = len(close)
    
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # Calculate weights
    weights = np.arange(1, period + 1, dtype=np.float64)
    weight_sum = weights.sum()
    
    # Calculate WMA
    for i in range(period - 1, n):
        weighted_sum = 0.0
        for j in range(period):
            weighted_sum += close[i - period + 1 + j] * weights[j]
        result[i] = weighted_sum / weight_sum
    
    return result


@njit(cache=True)
def dema(
    close: np.ndarray,
    period: int = 20,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Double Exponential Moving Average (DEMA).
    
    DEMA is a smoother and more responsive moving average that
    reduces the lag associated with traditional moving averages.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
    period : int, default 20
        Number of periods
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        DEMA values
    """
    close, min_periods = validate_input(close, period, min_periods)
    
    # Calculate EMA
    ema1 = ema(close, period, min_periods)
    
    # Calculate EMA of EMA
    ema2 = ema(ema1, period, min_periods)
    
    # DEMA = 2 * EMA - EMA(EMA)
    result = 2 * ema1 - ema2
    
    return result


@njit(cache=True)
def tema(
    close: np.ndarray,
    period: int = 20,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Triple Exponential Moving Average (TEMA).
    
    TEMA further reduces lag by using triple exponential smoothing.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
    period : int, default 20
        Number of periods
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        TEMA values
    """
    close, min_periods = validate_input(close, period, min_periods)
    
    # Calculate EMAs
    ema1 = ema(close, period, min_periods)
    ema2 = ema(ema1, period, min_periods)
    ema3 = ema(ema2, period, min_periods)
    
    # TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
    result = 3 * ema1 - 3 * ema2 + ema3
    
    return result