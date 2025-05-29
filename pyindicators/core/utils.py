"""Core utility functions optimized with Numba."""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional


@njit(cache=True)
def validate_input(
    data: np.ndarray, period: int, min_periods: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Validate input data and period.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    period : int
        Period for calculation
    min_periods : int, optional
        Minimum periods required for calculation
        
    Returns
    -------
    tuple
        Validated data and minimum periods
    """
    if len(data) == 0:
        raise ValueError("Input data is empty")
    
    if period <= 0:
        raise ValueError("Period must be positive")
    
    if min_periods is None:
        min_periods = period
    
    if len(data) < min_periods:
        raise ValueError(f"Not enough data points. Need at least {min_periods}")
    
    return data.astype(np.float64), min_periods


@njit(cache=True)
def rolling_window(data: np.ndarray, window: int) -> np.ndarray:
    """
    Create a rolling window view of the data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    window : int
        Window size
        
    Returns
    -------
    np.ndarray
        2D array where each row is a window
    """
    n = len(data)
    if n < window:
        raise ValueError("Data length is less than window size")
    
    result = np.empty((n - window + 1, window), dtype=data.dtype)
    for i in range(n - window + 1):
        result[i] = data[i:i + window]
    
    return result


@njit(cache=True)
def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Calculate True Range.
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
        
    Returns
    -------
    np.ndarray
        True range values
    """
    n = len(high)
    tr = np.empty(n, dtype=np.float64)
    
    # First value is just high - low
    tr[0] = high[0] - low[0]
    
    # Subsequent values consider previous close
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    return tr


@njit(cache=True)
def wilder_smoothing(data: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's smoothing (used in RSI, ATR, etc.).
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    period : int
        Smoothing period
        
    Returns
    -------
    np.ndarray
        Smoothed values
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # Initialize with simple average
    result[period-1] = np.mean(data[:period])
    
    # Apply Wilder's smoothing
    alpha = 1.0 / period
    for i in range(period, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    return result


@njit(cache=True, parallel=True)
def ewma_vectorized(data: np.ndarray, alpha: float, min_periods: int = 1) -> np.ndarray:
    """
    Vectorized Exponential Weighted Moving Average.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    alpha : float
        Smoothing factor (0 < alpha <= 1)
    min_periods : int
        Minimum periods for first valid value
        
    Returns
    -------
    np.ndarray
        EWMA values
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    if n == 0:
        return result
    
    # Initialize
    result[0] = data[0]
    
    # Calculate EWMA
    for i in range(1, n):
        if np.isnan(data[i]):
            result[i] = result[i-1]
        else:
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    # Set initial values to NaN based on min_periods
    if min_periods > 1:
        result[:min_periods-1] = np.nan
    
    return result


@njit(cache=True)
def calculate_gains_losses(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate gains and losses from price changes.
    
    Parameters
    ----------
    data : np.ndarray
        Price data
        
    Returns
    -------
    tuple
        Gains and losses arrays
    """
    n = len(data)
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        change = data[i] - data[i-1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = -change
    
    return gains, losses