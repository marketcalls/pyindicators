"""Volume Weighted Average Price (VWAP) implementation."""

import numpy as np
from numba import njit
from typing import Optional


@njit(cache=True)
def vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    anchor_points: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    VWAP is the average price weighted by volume, typically reset daily.
    It's used as a trading benchmark by institutional investors.
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    volume : np.ndarray
        Volume data
    anchor_points : np.ndarray, optional
        Boolean array indicating reset points (e.g., start of trading day)
        If None, calculates cumulative VWAP
        
    Returns
    -------
    np.ndarray
        VWAP values
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import vwap
    >>> n = 100
    >>> high = np.random.randn(n) + 102
    >>> low = np.random.randn(n) + 98
    >>> close = np.random.randn(n) + 100
    >>> volume = np.random.randint(1000, 10000, n)
    >>> vwap_values = vwap(high, low, close, volume)
    """
    # Validate inputs
    n = len(close)
    if len(high) != n or len(low) != n or len(volume) != n:
        raise ValueError("All input arrays must have the same length")
    
    # Calculate typical price
    typical_price = (high + low + close) / 3.0
    
    # Calculate VWAP
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    if anchor_points is None:
        # Cumulative VWAP
        cumulative_pv = 0.0
        cumulative_volume = 0.0
        
        for i in range(n):
            cumulative_pv += typical_price[i] * volume[i]
            cumulative_volume += volume[i]
            
            if cumulative_volume > 0:
                result[i] = cumulative_pv / cumulative_volume
            else:
                result[i] = typical_price[i]
    else:
        # Anchored VWAP (reset at anchor points)
        cumulative_pv = 0.0
        cumulative_volume = 0.0
        
        for i in range(n):
            if anchor_points[i]:
                # Reset at anchor point
                cumulative_pv = typical_price[i] * volume[i]
                cumulative_volume = volume[i]
            else:
                cumulative_pv += typical_price[i] * volume[i]
                cumulative_volume += volume[i]
            
            if cumulative_volume > 0:
                result[i] = cumulative_pv / cumulative_volume
            else:
                result[i] = typical_price[i]
    
    return result