"""Relative Strength Index (RSI) implementation."""

import numpy as np
from numba import njit
from typing import Optional

from pyindicators.core.utils import validate_input, calculate_gains_losses, wilder_smoothing


@njit(cache=True)
def rsi(
    close: np.ndarray,
    period: int = 14,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI is a momentum oscillator that measures the speed and magnitude of price changes.
    Values range from 0 to 100, with readings above 70 indicating overbought conditions
    and readings below 30 indicating oversold conditions.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
    period : int, default 14
        Number of periods for RSI calculation
    min_periods : int, optional
        Minimum periods required for calculation
        
    Returns
    -------
    np.ndarray
        RSI values
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import rsi
    >>> close = np.random.randn(100) + 100
    >>> rsi_values = rsi(close, period=14)
    """
    close, min_periods = validate_input(close, period, min_periods)
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # Calculate gains and losses
    gains, losses = calculate_gains_losses(close)
    
    # Apply Wilder's smoothing
    avg_gains = wilder_smoothing(gains, period)
    avg_losses = wilder_smoothing(losses, period)
    
    # Calculate RSI
    for i in range(period-1, n):
        if avg_losses[i] == 0:
            result[i] = 100.0
        else:
            rs = avg_gains[i] / avg_losses[i]
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result