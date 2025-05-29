"""Standard deviation calculation."""

import numpy as np
from numba import njit, prange
from typing import Optional

from pyindicators.core.utils import validate_input


@njit(cache=True, parallel=True)
def standard_deviation(
    close: np.ndarray,
    period: int = 20,
    min_periods: Optional[int] = None,
    ddof: int = 0
) -> np.ndarray:
    """
    Calculate rolling standard deviation.
    
    Parameters
    ----------
    close : np.ndarray
        Data array (typically close prices)
    period : int, default 20
        Period for standard deviation calculation
    min_periods : int, optional
        Minimum periods required
    ddof : int, default 0
        Delta degrees of freedom (0 for population, 1 for sample)
        
    Returns
    -------
    np.ndarray
        Standard deviation values
        
    Examples
    --------
    >>> import numpy as np
    >>> from pyindicators import standard_deviation
    >>> close = np.random.randn(100) + 100
    >>> std_values = standard_deviation(close, period=20)
    """
    close, min_periods = validate_input(close, period, min_periods)
    n = len(close)
    
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # Use parallel processing for efficiency
    for i in prange(period-1, n):
        window = close[i-period+1:i+1]
        mean = np.mean(window)
        
        # Calculate variance
        variance = 0.0
        for j in range(period):
            variance += (window[j] - mean) ** 2
        
        # Apply degrees of freedom correction
        if period - ddof > 0:
            variance = variance / (period - ddof)
            result[i] = np.sqrt(variance)
        else:
            result[i] = 0.0
    
    return result