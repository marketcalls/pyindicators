"""Williams %R indicator implementation."""

import numpy as np
from numba import njit
from typing import Optional

from pyindicators.core.utils import validate_input


@njit(cache=True)
def williams_r(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Williams %R.
    
    Williams %R is a momentum indicator that measures overbought and oversold levels.
    It is similar to the Stochastic Oscillator but plotted on a negative scale
    from 0 to -100.
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    period : int, default 14
        Lookback period
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        Williams %R values (-100 to 0)
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import williams_r
    >>> n = 100
    >>> high = np.random.randn(n) + 102
    >>> low = np.random.randn(n) + 98
    >>> close = np.random.randn(n) + 100
    >>> wr = williams_r(high, low, close, period=14)
    """
    # Validate inputs
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("Input arrays must have the same length")
    
    close, min_periods = validate_input(close, period, min_periods)
    n = len(close)
    
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # Calculate Williams %R
    for i in range(period-1, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            result[i] = -50.0  # Neutral when range is zero
        else:
            result[i] = -100.0 * (highest - close[i]) / (highest - lowest)
    
    return result