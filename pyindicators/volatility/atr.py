"""Average True Range (ATR) implementation."""

import numpy as np
from numba import njit
from typing import Optional

from pyindicators.core.utils import validate_input, true_range, wilder_smoothing


@njit(cache=True)
def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Average True Range (ATR).
    
    ATR is a volatility indicator that measures the average range of price
    movement, taking into account gaps and limit moves.
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    period : int, default 14
        Period for ATR calculation
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        ATR values
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import atr
    >>> n = 100
    >>> high = np.random.randn(n) + 102
    >>> low = np.random.randn(n) + 98
    >>> close = np.random.randn(n) + 100
    >>> atr_values = atr(high, low, close)
    """
    # Validate inputs
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("Input arrays must have the same length")
    
    close, min_periods = validate_input(close, period, min_periods)
    
    # Calculate True Range
    tr = true_range(high, low, close)
    
    # Apply Wilder's smoothing to get ATR
    atr_values = wilder_smoothing(tr, period)
    
    return atr_values