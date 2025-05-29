"""On Balance Volume (OBV) implementation."""

import numpy as np
from numba import njit
from typing import Optional


@njit(cache=True)
def obv(
    close: np.ndarray,
    volume: np.ndarray,
    initial_value: float = 0.0
) -> np.ndarray:
    """
    Calculate On Balance Volume (OBV).
    
    OBV is a momentum indicator that uses volume flow to predict changes in price.
    It adds volume on up days and subtracts volume on down days.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
    volume : np.ndarray
        Volume data
    initial_value : float, default 0.0
        Initial OBV value
        
    Returns
    -------
    np.ndarray
        OBV values
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import obv
    >>> close = np.random.randn(100) + 100
    >>> volume = np.random.randint(1000, 10000, 100)
    >>> obv_values = obv(close, volume)
    """
    if len(close) != len(volume):
        raise ValueError("Close and volume arrays must have the same length")
    
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    
    # Initialize first value
    result[0] = initial_value
    
    # Calculate OBV
    for i in range(1, n):
        if close[i] > close[i-1]:
            # Price up, add volume
            result[i] = result[i-1] + volume[i]
        elif close[i] < close[i-1]:
            # Price down, subtract volume
            result[i] = result[i-1] - volume[i]
        else:
            # Price unchanged, OBV unchanged
            result[i] = result[i-1]
    
    return result