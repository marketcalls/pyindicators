"""Money Flow Index (MFI) implementation."""

import numpy as np
from numba import njit
from typing import Optional

from pyindicators.core.utils import validate_input


@njit(cache=True)
def mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int = 14,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Money Flow Index (MFI).
    
    MFI is a momentum indicator that uses price and volume to identify
    overbought or oversold conditions. It's similar to RSI but incorporates volume.
    
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
    period : int, default 14
        Period for MFI calculation
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        MFI values (0 to 100)
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import mfi
    >>> n = 100
    >>> high = np.random.randn(n) + 102
    >>> low = np.random.randn(n) + 98
    >>> close = np.random.randn(n) + 100
    >>> volume = np.random.randint(1000, 10000, n)
    >>> mfi_values = mfi(high, low, close, volume)
    """
    # Validate inputs
    n = len(close)
    if len(high) != n or len(low) != n or len(volume) != n:
        raise ValueError("All input arrays must have the same length")
    
    close, min_periods = validate_input(close, period, min_periods)
    
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # Calculate typical price
    typical_price = (high + low + close) / 3.0
    
    # Calculate raw money flow
    raw_money_flow = typical_price * volume
    
    # Calculate positive and negative money flow
    positive_flow = np.zeros(n, dtype=np.float64)
    negative_flow = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        if typical_price[i] > typical_price[i-1]:
            positive_flow[i] = raw_money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_flow[i] = raw_money_flow[i]
    
    # Calculate MFI
    for i in range(period, n):
        positive_sum = np.sum(positive_flow[i-period+1:i+1])
        negative_sum = np.sum(negative_flow[i-period+1:i+1])
        
        if negative_sum == 0:
            result[i] = 100.0
        else:
            money_ratio = positive_sum / negative_sum
            result[i] = 100.0 - (100.0 / (1.0 + money_ratio))
    
    return result