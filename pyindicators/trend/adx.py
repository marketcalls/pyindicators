"""Average Directional Index (ADX) implementation."""

import numpy as np
from numba import njit
from typing import Optional, Tuple

from pyindicators.core.utils import validate_input, true_range, wilder_smoothing


@njit(cache=True)
def directional_movement(
    high: np.ndarray,
    low: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate positive and negative directional movement.
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
        
    Returns
    -------
    tuple of np.ndarray
        Plus DM and Minus DM
    """
    n = len(high)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    return plus_dm, minus_dm


@njit(cache=True)
def plus_di(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Plus Directional Indicator (+DI).
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    period : int, default 14
        Period for calculation
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        +DI values
    """
    # Validate inputs
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("Input arrays must have the same length")
    
    close, min_periods = validate_input(close, period, min_periods)
    
    # Calculate directional movement
    plus_dm, _ = directional_movement(high, low)
    
    # Calculate true range
    tr = true_range(high, low, close)
    
    # Apply Wilder's smoothing
    smooth_plus_dm = wilder_smoothing(plus_dm, period)
    smooth_tr = wilder_smoothing(tr, period)
    
    # Calculate +DI
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    for i in range(period-1, n):
        if smooth_tr[i] != 0:
            result[i] = (smooth_plus_dm[i] / smooth_tr[i]) * 100
        else:
            result[i] = 0
    
    return result


@njit(cache=True)
def minus_di(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Minus Directional Indicator (-DI).
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    period : int, default 14
        Period for calculation
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        -DI values
    """
    # Validate inputs
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("Input arrays must have the same length")
    
    close, min_periods = validate_input(close, period, min_periods)
    
    # Calculate directional movement
    _, minus_dm = directional_movement(high, low)
    
    # Calculate true range
    tr = true_range(high, low, close)
    
    # Apply Wilder's smoothing
    smooth_minus_dm = wilder_smoothing(minus_dm, period)
    smooth_tr = wilder_smoothing(tr, period)
    
    # Calculate -DI
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    for i in range(period-1, n):
        if smooth_tr[i] != 0:
            result[i] = (smooth_minus_dm[i] / smooth_tr[i]) * 100
        else:
            result[i] = 0
    
    return result


@njit(cache=True)
def adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Average Directional Index (ADX).
    
    ADX is used to determine the strength of a trend, regardless of direction.
    Values above 25 indicate a strong trend, while values below 20 suggest
    a weak trend or ranging market.
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    period : int, default 14
        Period for calculation
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        ADX values
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import adx
    >>> n = 100
    >>> high = np.random.randn(n) + 102
    >>> low = np.random.randn(n) + 98
    >>> close = np.random.randn(n) + 100
    >>> adx_values = adx(high, low, close)
    """
    # Calculate +DI and -DI
    plus_di_values = plus_di(high, low, close, period, min_periods)
    minus_di_values = minus_di(high, low, close, period, min_periods)
    
    n = len(close)
    dx = np.empty(n, dtype=np.float64)
    dx[:] = np.nan
    
    # Calculate DX
    for i in range(period-1, n):
        di_sum = plus_di_values[i] + minus_di_values[i]
        if di_sum != 0:
            dx[i] = abs(plus_di_values[i] - minus_di_values[i]) / di_sum * 100
        else:
            dx[i] = 0
    
    # Apply Wilder's smoothing to get ADX
    adx_values = wilder_smoothing(dx, period)
    
    return adx_values