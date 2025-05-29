"""Rate of Change (ROC) and Momentum indicators."""

import numpy as np
from numba import njit
from typing import Optional

from pyindicators.core.utils import validate_input


@njit(cache=True)
def roc(
    close: np.ndarray,
    period: int = 10,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Rate of Change (ROC).
    
    ROC measures the percentage change in price from n periods ago.
    It oscillates above and below zero, with positive values indicating
    upward momentum and negative values indicating downward momentum.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
    period : int, default 10
        Number of periods for ROC calculation
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        ROC values as percentages
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import roc
    >>> close = np.random.randn(100) + 100
    >>> roc_values = roc(close, period=10)
    """
    close, min_periods = validate_input(close, period, min_periods)
    n = len(close)
    
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # Calculate ROC
    for i in range(period, n):
        if close[i-period] != 0:
            result[i] = ((close[i] - close[i-period]) / close[i-period]) * 100.0
        else:
            result[i] = 0.0
    
    return result


@njit(cache=True)
def momentum(
    close: np.ndarray,
    period: int = 10,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Momentum indicator.
    
    Momentum is the raw difference in price from n periods ago.
    It's similar to ROC but shows absolute change rather than percentage.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
    period : int, default 10
        Number of periods for momentum calculation
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        Momentum values
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import momentum
    >>> close = np.random.randn(100) + 100
    >>> mom_values = momentum(close, period=10)
    """
    close, min_periods = validate_input(close, period, min_periods)
    n = len(close)
    
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # Calculate Momentum
    for i in range(period, n):
        result[i] = close[i] - close[i-period]
    
    return result