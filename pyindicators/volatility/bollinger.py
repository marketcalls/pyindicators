"""Bollinger Bands implementation."""

import numpy as np
from numba import njit
from typing import Optional, Tuple

from pyindicators.trend.moving_averages import sma
from pyindicators.volatility.std import standard_deviation


@njit(cache=True)
def bollinger_bands(
    close: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
    min_periods: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of a middle band (SMA) and two outer bands
    that are standard deviations away from the middle band.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
    period : int, default 20
        Period for moving average and standard deviation
    std_dev : float, default 2.0
        Number of standard deviations for bands
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    tuple of np.ndarray
        Upper band, middle band (SMA), and lower band
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import bollinger_bands
    >>> close = np.random.randn(100) + 100
    >>> upper, middle, lower = bollinger_bands(close)
    """
    # Calculate middle band (SMA)
    middle_band = sma(close, period, min_periods)
    
    # Calculate standard deviation
    std = standard_deviation(close, period, min_periods)
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    
    return upper_band, middle_band, lower_band


@njit(cache=True)
def bollinger_bandwidth(
    close: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Bollinger Bandwidth.
    
    Bandwidth measures the percentage difference between the upper and lower bands.
    It's useful for identifying periods of low volatility (squeeze).
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
    period : int, default 20
        Period for calculation
    std_dev : float, default 2.0
        Number of standard deviations
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        Bandwidth values as percentages
    """
    upper, middle, lower = bollinger_bands(close, period, std_dev, min_periods)
    
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    for i in range(period-1, n):
        if middle[i] != 0:
            result[i] = ((upper[i] - lower[i]) / middle[i]) * 100
        else:
            result[i] = 0
    
    return result


@njit(cache=True)
def bollinger_percent_b(
    close: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Bollinger %B.
    
    %B shows where price is in relation to the bands.
    %B = (Price - Lower Band) / (Upper Band - Lower Band)
    
    Values:
    - Above 1: Price is above upper band
    - 1: Price is at upper band
    - 0.5: Price is at middle band
    - 0: Price is at lower band
    - Below 0: Price is below lower band
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
    period : int, default 20
        Period for calculation
    std_dev : float, default 2.0
        Number of standard deviations
    min_periods : int, optional
        Minimum periods required
        
    Returns
    -------
    np.ndarray
        %B values
    """
    upper, _, lower = bollinger_bands(close, period, std_dev, min_periods)
    
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    for i in range(period-1, n):
        band_width = upper[i] - lower[i]
        if band_width != 0:
            result[i] = (close[i] - lower[i]) / band_width
        else:
            result[i] = 0.5  # Middle when bands converge
    
    return result