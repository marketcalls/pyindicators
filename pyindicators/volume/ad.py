"""Accumulation/Distribution Line implementation."""

import numpy as np
from numba import njit


@njit(cache=True)
def money_flow_multiplier(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Calculate Money Flow Multiplier.
    
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
        Money flow multiplier values
    """
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        hl_range = high[i] - low[i]
        if hl_range != 0:
            result[i] = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
        else:
            result[i] = 0.0
    
    return result


@njit(cache=True)
def ad(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray
) -> np.ndarray:
    """
    Calculate Accumulation/Distribution indicator.
    
    A/D is a volume-based indicator designed to measure the cumulative flow
    of money into and out of a security.
    
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
        
    Returns
    -------
    np.ndarray
        A/D values
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import ad
    >>> n = 100
    >>> high = np.random.randn(n) + 102
    >>> low = np.random.randn(n) + 98
    >>> close = np.random.randn(n) + 100
    >>> volume = np.random.randint(1000, 10000, n)
    >>> ad_values = ad(high, low, close, volume)
    """
    # Validate inputs
    n = len(close)
    if len(high) != n or len(low) != n or len(volume) != n:
        raise ValueError("All input arrays must have the same length")
    
    # Calculate money flow multiplier
    mfm = money_flow_multiplier(high, low, close)
    
    # Calculate money flow volume
    mfv = mfm * volume
    
    # Calculate cumulative A/D
    result = np.empty(n, dtype=np.float64)
    result[0] = mfv[0]
    
    for i in range(1, n):
        result[i] = result[i-1] + mfv[i]
    
    return result


# Alias for compatibility
adl = ad