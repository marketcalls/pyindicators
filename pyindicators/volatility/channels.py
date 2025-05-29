"""Channel indicators (Keltner, Donchian)."""

import numpy as np
from numba import njit
from typing import Optional, Tuple

from pyindicators.core.utils import validate_input
from pyindicators.trend.moving_averages import ema
from pyindicators.volatility.atr import atr


@njit(cache=True)
def keltner_channels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
    multiplier: float = 2.0,
    atr_period: int = 10,
    min_periods: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Keltner Channels.
    
    Keltner Channels are volatility-based envelopes set above and below
    an exponential moving average.
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    period : int, default 20
        Period for EMA calculation
    multiplier : float, default 2.0
        ATR multiplier for channel width
    atr_period : int, default 10
        Period for ATR calculation
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    tuple of np.ndarray
        Upper channel, middle line (EMA), lower channel
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import keltner_channels
    >>> n = 100
    >>> high = np.random.randn(n) + 102
    >>> low = np.random.randn(n) + 98
    >>> close = np.random.randn(n) + 100
    >>> upper, middle, lower = keltner_channels(high, low, close)
    """
    # Validate inputs
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("Input arrays must have the same length")
    
    close, min_periods = validate_input(close, period, min_periods)
    
    # Calculate middle line (EMA)
    middle_line = ema(close, period, min_periods)
    
    # Calculate ATR
    atr_values = atr(high, low, close, atr_period, min_periods)
    
    # Calculate channels
    upper_channel = middle_line + (multiplier * atr_values)
    lower_channel = middle_line - (multiplier * atr_values)
    
    return upper_channel, middle_line, lower_channel


@njit(cache=True)
def donchian_channels(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20,
    min_periods: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Donchian Channels.
    
    Donchian Channels plot the highest high and lowest low over a given period.
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    period : int, default 20
        Lookback period
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    tuple of np.ndarray
        Upper channel (highest high), middle line, lower channel (lowest low)
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import donchian_channels
    >>> n = 100
    >>> high = np.random.randn(n) + 102
    >>> low = np.random.randn(n) + 98
    >>> upper, middle, lower = donchian_channels(high, low)
    """
    # Validate inputs
    if len(high) != len(low):
        raise ValueError("High and low arrays must have the same length")
    
    high, min_periods = validate_input(high, period, min_periods)
    n = len(high)
    
    upper_channel = np.empty(n, dtype=np.float64)
    lower_channel = np.empty(n, dtype=np.float64)
    middle_line = np.empty(n, dtype=np.float64)
    
    upper_channel[:] = np.nan
    lower_channel[:] = np.nan
    middle_line[:] = np.nan
    
    # Calculate channels
    for i in range(period-1, n):
        upper_channel[i] = np.max(high[i-period+1:i+1])
        lower_channel[i] = np.min(low[i-period+1:i+1])
        middle_line[i] = (upper_channel[i] + lower_channel[i]) / 2.0
    
    return upper_channel, middle_line, lower_channel