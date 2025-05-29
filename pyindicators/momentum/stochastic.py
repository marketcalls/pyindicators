"""Stochastic Oscillator implementation."""

import numpy as np
from numba import njit
from typing import Optional, Tuple

from pyindicators.core.utils import validate_input


@njit(cache=True)
def stochastic_fast(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    min_periods: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Fast Stochastic Oscillator (%K and %D).
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    period : int, default 14
        Period for %K calculation
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    tuple of np.ndarray
        %K and %D values
    """
    # Validate inputs
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("Input arrays must have the same length")
    
    close, min_periods = validate_input(close, period, min_periods)
    n = len(close)
    
    k_percent = np.empty(n, dtype=np.float64)
    k_percent[:] = np.nan
    
    # Calculate %K
    for i in range(period-1, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            k_percent[i] = 50.0  # Neutral when range is zero
        else:
            k_percent[i] = 100.0 * (close[i] - lowest) / (highest - lowest)
    
    # Calculate %D (3-period SMA of %K)
    d_percent = np.empty(n, dtype=np.float64)
    d_percent[:] = np.nan
    
    for i in range(period+1, n):  # Need at least 3 %K values
        d_percent[i] = np.mean(k_percent[i-2:i+1])
    
    return k_percent, d_percent


@njit(cache=True)
def stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    k_smooth: int = 3,
    d_period: int = 3,
    min_periods: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Full Stochastic Oscillator (Slow Stochastic).
    
    The Stochastic Oscillator is a momentum indicator that shows the location
    of the close relative to the high-low range over a set number of periods.
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    k_period : int, default 14
        Period for raw %K calculation
    k_smooth : int, default 3
        Smoothing period for %K
    d_period : int, default 3
        Period for %D (signal line)
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    tuple of np.ndarray
        Smoothed %K and %D values
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import stochastic
    >>> n = 100
    >>> high = np.random.randn(n) + 102
    >>> low = np.random.randn(n) + 98
    >>> close = np.random.randn(n) + 100
    >>> k, d = stochastic(high, low, close)
    """
    # First calculate fast stochastic
    fast_k, _ = stochastic_fast(high, low, close, k_period, min_periods)
    
    n = len(close)
    slow_k = np.empty(n, dtype=np.float64)
    slow_k[:] = np.nan
    
    # Smooth %K
    start_idx = k_period + k_smooth - 2
    for i in range(start_idx, n):
        slow_k[i] = np.mean(fast_k[i-k_smooth+1:i+1])
    
    # Calculate %D
    slow_d = np.empty(n, dtype=np.float64)
    slow_d[:] = np.nan
    
    start_idx = k_period + k_smooth + d_period - 3
    for i in range(start_idx, n):
        slow_d[i] = np.mean(slow_k[i-d_period+1:i+1])
    
    return slow_k, slow_d