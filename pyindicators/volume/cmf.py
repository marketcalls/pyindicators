"""Chaikin Money Flow (CMF) implementation."""

import numpy as np
from numba import njit
from typing import Optional

from pyindicators.core.utils import validate_input
from pyindicators.volume.ad import money_flow_multiplier


@njit(cache=True)
def cmf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int = 20,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Chaikin Money Flow (CMF).
    
    CMF measures the amount of money flow volume over a specific period.
    Values range from -1 to +1, with positive values indicating buying pressure
    and negative values indicating selling pressure.
    
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
    period : int, default 20
        Period for CMF calculation
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        CMF values (-1 to +1)
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import cmf
    >>> n = 100
    >>> high = np.random.randn(n) + 102
    >>> low = np.random.randn(n) + 98
    >>> close = np.random.randn(n) + 100
    >>> volume = np.random.randint(1000, 10000, n)
    >>> cmf_values = cmf(high, low, close, volume)
    """
    # Validate inputs
    n = len(close)
    if len(high) != n or len(low) != n or len(volume) != n:
        raise ValueError("All input arrays must have the same length")
    
    close, min_periods = validate_input(close, period, min_periods)
    
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # Calculate money flow multiplier
    mfm = money_flow_multiplier(high, low, close)
    
    # Calculate money flow volume
    mfv = mfm * volume
    
    # Calculate CMF
    for i in range(period-1, n):
        mfv_sum = np.sum(mfv[i-period+1:i+1])
        volume_sum = np.sum(volume[i-period+1:i+1])
        
        if volume_sum != 0:
            result[i] = mfv_sum / volume_sum
        else:
            result[i] = 0.0
    
    return result