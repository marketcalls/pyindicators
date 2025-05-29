"""Pandas DataFrame and Series support for PyIndicators.

This module provides convenient wrappers that allow using PyIndicators
with pandas DataFrames and Series, similar to TA-Lib's interface.
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional

# Import all indicators
from pyindicators.momentum import (
    rsi as _rsi,
    stochastic as _stochastic,
    williams_r as _williams_r,
    roc as _roc,
    momentum as _momentum,
)
from pyindicators.trend import (
    sma as _sma,
    ema as _ema,
    wma as _wma,
    macd as _macd,
    adx as _adx,
    dema as _dema,
    tema as _tema,
)
from pyindicators.volatility import (
    bollinger_bands as _bollinger_bands,
    atr as _atr,
    keltner_channels as _keltner_channels,
    donchian_channels as _donchian_channels,
    standard_deviation as _standard_deviation,
)
from pyindicators.volume import (
    obv as _obv,
    vwap as _vwap,
    ad as _ad,
    mfi as _mfi,
    cmf as _cmf,
)


def _ensure_numpy(data: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Convert pandas Series to numpy array if needed."""
    if isinstance(data, pd.Series):
        return data.to_numpy()
    return data


def _wrap_result(result: np.ndarray, index: pd.Index, name: str = None) -> pd.Series:
    """Wrap numpy result in pandas Series with proper index and name."""
    return pd.Series(result, index=index, name=name)


# Momentum Indicators
def rsi(close: Union[pd.Series, np.ndarray], period: int = 14) -> Union[pd.Series, np.ndarray]:
    """Calculate Relative Strength Index (RSI).
    
    Parameters
    ----------
    close : pd.Series or np.ndarray
        Close prices
    period : int, default 14
        Period for RSI calculation
        
    Returns
    -------
    pd.Series or np.ndarray
        RSI values (same type as input)
    """
    is_series = isinstance(close, pd.Series)
    close_array = _ensure_numpy(close)
    result = _rsi(close_array, period)
    
    if is_series:
        return _wrap_result(result, close.index, f'RSI_{period}')
    return result


def stochastic(high: Union[pd.Series, np.ndarray], 
                low: Union[pd.Series, np.ndarray], 
                close: Union[pd.Series, np.ndarray],
                k_period: int = 14,
                d_period: int = 3) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """Calculate Stochastic Oscillator.
    
    Parameters
    ----------
    high : pd.Series or np.ndarray
        High prices
    low : pd.Series or np.ndarray
        Low prices
    close : pd.Series or np.ndarray
        Close prices
    k_period : int, default 14
        Period for %K calculation
    d_period : int, default 3
        Period for %D calculation (SMA of %K)
        
    Returns
    -------
    tuple of pd.Series or np.ndarray
        %K and %D values
    """
    is_series = isinstance(high, pd.Series)
    high_array = _ensure_numpy(high)
    low_array = _ensure_numpy(low)
    close_array = _ensure_numpy(close)
    
    k, d = _stochastic(high_array, low_array, close_array, k_period, d_period)
    
    if is_series:
        index = high.index
        return (_wrap_result(k, index, f'Stoch_K_{k_period}'),
                _wrap_result(d, index, f'Stoch_D_{d_period}'))
    return k, d


def williams_r(high: Union[pd.Series, np.ndarray],
                low: Union[pd.Series, np.ndarray],
                close: Union[pd.Series, np.ndarray],
                period: int = 14) -> Union[pd.Series, np.ndarray]:
    """Calculate Williams %R."""
    is_series = isinstance(high, pd.Series)
    high_array = _ensure_numpy(high)
    low_array = _ensure_numpy(low)
    close_array = _ensure_numpy(close)
    
    result = _williams_r(high_array, low_array, close_array, period)
    
    if is_series:
        return _wrap_result(result, high.index, f'Williams_R_{period}')
    return result


def roc(close: Union[pd.Series, np.ndarray], period: int = 10) -> Union[pd.Series, np.ndarray]:
    """Calculate Rate of Change (ROC)."""
    is_series = isinstance(close, pd.Series)
    close_array = _ensure_numpy(close)
    result = _roc(close_array, period)
    
    if is_series:
        return _wrap_result(result, close.index, f'ROC_{period}')
    return result


def momentum(close: Union[pd.Series, np.ndarray], period: int = 10) -> Union[pd.Series, np.ndarray]:
    """Calculate Momentum."""
    is_series = isinstance(close, pd.Series)
    close_array = _ensure_numpy(close)
    result = _momentum(close_array, period)
    
    if is_series:
        return _wrap_result(result, close.index, f'MOM_{period}')
    return result


# Trend Indicators
def sma(close: Union[pd.Series, np.ndarray], period: int = 20) -> Union[pd.Series, np.ndarray]:
    """Calculate Simple Moving Average (SMA)."""
    is_series = isinstance(close, pd.Series)
    close_array = _ensure_numpy(close)
    result = _sma(close_array, period)
    
    if is_series:
        return _wrap_result(result, close.index, f'SMA_{period}')
    return result


def ema(close: Union[pd.Series, np.ndarray], period: int = 20) -> Union[pd.Series, np.ndarray]:
    """Calculate Exponential Moving Average (EMA)."""
    is_series = isinstance(close, pd.Series)
    close_array = _ensure_numpy(close)
    result = _ema(close_array, period)
    
    if is_series:
        return _wrap_result(result, close.index, f'EMA_{period}')
    return result


def wma(close: Union[pd.Series, np.ndarray], period: int = 20) -> Union[pd.Series, np.ndarray]:
    """Calculate Weighted Moving Average (WMA)."""
    is_series = isinstance(close, pd.Series)
    close_array = _ensure_numpy(close)
    result = _wma(close_array, period)
    
    if is_series:
        return _wrap_result(result, close.index, f'WMA_{period}')
    return result


def macd(close: Union[pd.Series, np.ndarray],
         fast_period: int = 12,
         slow_period: int = 26,
         signal_period: int = 9) -> Tuple[Union[pd.Series, np.ndarray], 
                                         Union[pd.Series, np.ndarray], 
                                         Union[pd.Series, np.ndarray]]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    is_series = isinstance(close, pd.Series)
    close_array = _ensure_numpy(close)
    
    macd_line, signal_line, histogram = _macd(close_array, fast_period, slow_period, signal_period)
    
    if is_series:
        index = close.index
        return (_wrap_result(macd_line, index, 'MACD'),
                _wrap_result(signal_line, index, 'MACD_Signal'),
                _wrap_result(histogram, index, 'MACD_Histogram'))
    return macd_line, signal_line, histogram


def adx(high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        period: int = 14) -> Union[pd.Series, np.ndarray]:
    """Calculate Average Directional Index (ADX)."""
    is_series = isinstance(high, pd.Series)
    high_array = _ensure_numpy(high)
    low_array = _ensure_numpy(low)
    close_array = _ensure_numpy(close)
    
    result = _adx(high_array, low_array, close_array, period)
    
    if is_series:
        return _wrap_result(result, high.index, f'ADX_{period}')
    return result


def dema(close: Union[pd.Series, np.ndarray], period: int = 20) -> Union[pd.Series, np.ndarray]:
    """Calculate Double Exponential Moving Average (DEMA)."""
    is_series = isinstance(close, pd.Series)
    close_array = _ensure_numpy(close)
    result = _dema(close_array, period)
    
    if is_series:
        return _wrap_result(result, close.index, f'DEMA_{period}')
    return result


def tema(close: Union[pd.Series, np.ndarray], period: int = 20) -> Union[pd.Series, np.ndarray]:
    """Calculate Triple Exponential Moving Average (TEMA)."""
    is_series = isinstance(close, pd.Series)
    close_array = _ensure_numpy(close)
    result = _tema(close_array, period)
    
    if is_series:
        return _wrap_result(result, close.index, f'TEMA_{period}')
    return result


# Volatility Indicators
def bollinger_bands(close: Union[pd.Series, np.ndarray],
                    period: int = 20,
                    std_dev: float = 2.0) -> Tuple[Union[pd.Series, np.ndarray],
                                                   Union[pd.Series, np.ndarray],
                                                   Union[pd.Series, np.ndarray]]:
    """Calculate Bollinger Bands."""
    is_series = isinstance(close, pd.Series)
    close_array = _ensure_numpy(close)
    
    upper, middle, lower = _bollinger_bands(close_array, period, std_dev)
    
    if is_series:
        index = close.index
        return (_wrap_result(upper, index, 'BB_Upper'),
                _wrap_result(middle, index, 'BB_Middle'),
                _wrap_result(lower, index, 'BB_Lower'))
    return upper, middle, lower


def atr(high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        period: int = 14) -> Union[pd.Series, np.ndarray]:
    """Calculate Average True Range (ATR)."""
    is_series = isinstance(high, pd.Series)
    high_array = _ensure_numpy(high)
    low_array = _ensure_numpy(low)
    close_array = _ensure_numpy(close)
    
    result = _atr(high_array, low_array, close_array, period)
    
    if is_series:
        return _wrap_result(result, high.index, f'ATR_{period}')
    return result


def keltner_channels(high: Union[pd.Series, np.ndarray],
                     low: Union[pd.Series, np.ndarray],
                     close: Union[pd.Series, np.ndarray],
                     period: int = 20,
                     multiplier: float = 2.0) -> Tuple[Union[pd.Series, np.ndarray],
                                                      Union[pd.Series, np.ndarray],
                                                      Union[pd.Series, np.ndarray]]:
    """Calculate Keltner Channels."""
    is_series = isinstance(high, pd.Series)
    high_array = _ensure_numpy(high)
    low_array = _ensure_numpy(low)
    close_array = _ensure_numpy(close)
    
    upper, middle, lower = _keltner_channels(high_array, low_array, close_array, period, multiplier)
    
    if is_series:
        index = high.index
        return (_wrap_result(upper, index, 'KC_Upper'),
                _wrap_result(middle, index, 'KC_Middle'),
                _wrap_result(lower, index, 'KC_Lower'))
    return upper, middle, lower


def donchian_channels(high: Union[pd.Series, np.ndarray],
                      low: Union[pd.Series, np.ndarray],
                      period: int = 20) -> Tuple[Union[pd.Series, np.ndarray],
                                                Union[pd.Series, np.ndarray],
                                                Union[pd.Series, np.ndarray]]:
    """Calculate Donchian Channels."""
    is_series = isinstance(high, pd.Series)
    high_array = _ensure_numpy(high)
    low_array = _ensure_numpy(low)
    
    upper, middle, lower = _donchian_channels(high_array, low_array, period)
    
    if is_series:
        index = high.index
        return (_wrap_result(upper, index, 'DC_Upper'),
                _wrap_result(middle, index, 'DC_Middle'),
                _wrap_result(lower, index, 'DC_Lower'))
    return upper, middle, lower


def standard_deviation(close: Union[pd.Series, np.ndarray],
                       period: int = 20) -> Union[pd.Series, np.ndarray]:
    """Calculate Standard Deviation."""
    is_series = isinstance(close, pd.Series)
    close_array = _ensure_numpy(close)
    result = _standard_deviation(close_array, period)
    
    if is_series:
        return _wrap_result(result, close.index, f'STD_{period}')
    return result


# Volume Indicators
def obv(close: Union[pd.Series, np.ndarray],
        volume: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """Calculate On Balance Volume (OBV)."""
    is_series = isinstance(close, pd.Series)
    close_array = _ensure_numpy(close)
    volume_array = _ensure_numpy(volume)
    
    result = _obv(close_array, volume_array)
    
    if is_series:
        return _wrap_result(result, close.index, 'OBV')
    return result


def vwap(high: Union[pd.Series, np.ndarray],
         low: Union[pd.Series, np.ndarray],
         close: Union[pd.Series, np.ndarray],
         volume: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """Calculate Volume Weighted Average Price (VWAP)."""
    is_series = isinstance(high, pd.Series)
    high_array = _ensure_numpy(high)
    low_array = _ensure_numpy(low)
    close_array = _ensure_numpy(close)
    volume_array = _ensure_numpy(volume)
    
    result = _vwap(high_array, low_array, close_array, volume_array)
    
    if is_series:
        return _wrap_result(result, high.index, 'VWAP')
    return result


def ad(high: Union[pd.Series, np.ndarray],
       low: Union[pd.Series, np.ndarray],
       close: Union[pd.Series, np.ndarray],
       volume: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """Calculate Accumulation/Distribution Line."""
    is_series = isinstance(high, pd.Series)
    high_array = _ensure_numpy(high)
    low_array = _ensure_numpy(low)
    close_array = _ensure_numpy(close)
    volume_array = _ensure_numpy(volume)
    
    result = _ad(high_array, low_array, close_array, volume_array)
    
    if is_series:
        return _wrap_result(result, high.index, 'AD')
    return result


def mfi(high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        volume: Union[pd.Series, np.ndarray],
        period: int = 14) -> Union[pd.Series, np.ndarray]:
    """Calculate Money Flow Index (MFI)."""
    is_series = isinstance(high, pd.Series)
    high_array = _ensure_numpy(high)
    low_array = _ensure_numpy(low)
    close_array = _ensure_numpy(close)
    volume_array = _ensure_numpy(volume)
    
    result = _mfi(high_array, low_array, close_array, volume_array, period)
    
    if is_series:
        return _wrap_result(result, high.index, f'MFI_{period}')
    return result


def cmf(high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        volume: Union[pd.Series, np.ndarray],
        period: int = 20) -> Union[pd.Series, np.ndarray]:
    """Calculate Chaikin Money Flow (CMF)."""
    is_series = isinstance(high, pd.Series)
    high_array = _ensure_numpy(high)
    low_array = _ensure_numpy(low)
    close_array = _ensure_numpy(close)
    volume_array = _ensure_numpy(volume)
    
    result = _cmf(high_array, low_array, close_array, volume_array, period)
    
    if is_series:
        return _wrap_result(result, high.index, f'CMF_{period}')
    return result


# DataFrame convenience functions
def add_all_indicators(df: pd.DataFrame,
                      high_col: str = 'High',
                      low_col: str = 'Low',
                      close_col: str = 'Close',
                      volume_col: str = 'Volume',
                      prefix: str = '') -> pd.DataFrame:
    """Add all available indicators to a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data
    high_col : str, default 'High'
        Column name for high prices
    low_col : str, default 'Low'
        Column name for low prices
    close_col : str, default 'Close'
        Column name for close prices
    volume_col : str, default 'Volume'
        Column name for volume
    prefix : str, default ''
        Prefix for indicator column names
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added indicator columns
    """
    df = df.copy()
    
    # Momentum
    df[f'{prefix}RSI'] = rsi(df[close_col])
    df[f'{prefix}ROC'] = roc(df[close_col])
    df[f'{prefix}MOM'] = momentum(df[close_col])
    k, d = stochastic(df[high_col], df[low_col], df[close_col])
    df[f'{prefix}STOCH_K'] = k
    df[f'{prefix}STOCH_D'] = d
    df[f'{prefix}WILLIAMS_R'] = williams_r(df[high_col], df[low_col], df[close_col])
    
    # Trend
    df[f'{prefix}SMA_20'] = sma(df[close_col], 20)
    df[f'{prefix}EMA_20'] = ema(df[close_col], 20)
    df[f'{prefix}SMA_50'] = sma(df[close_col], 50)
    df[f'{prefix}EMA_50'] = ema(df[close_col], 50)
    macd_line, signal, hist = macd(df[close_col])
    df[f'{prefix}MACD'] = macd_line
    df[f'{prefix}MACD_SIGNAL'] = signal
    df[f'{prefix}MACD_HIST'] = hist
    df[f'{prefix}ADX'] = adx(df[high_col], df[low_col], df[close_col])
    
    # Volatility
    bb_upper, bb_middle, bb_lower = bollinger_bands(df[close_col])
    df[f'{prefix}BB_UPPER'] = bb_upper
    df[f'{prefix}BB_MIDDLE'] = bb_middle
    df[f'{prefix}BB_LOWER'] = bb_lower
    df[f'{prefix}ATR'] = atr(df[high_col], df[low_col], df[close_col])
    
    # Volume
    if volume_col in df.columns:
        df[f'{prefix}OBV'] = obv(df[close_col], df[volume_col])
        df[f'{prefix}VWAP'] = vwap(df[high_col], df[low_col], df[close_col], df[volume_col])
        df[f'{prefix}AD'] = ad(df[high_col], df[low_col], df[close_col], df[volume_col])
        df[f'{prefix}MFI'] = mfi(df[high_col], df[low_col], df[close_col], df[volume_col])
        df[f'{prefix}CMF'] = cmf(df[high_col], df[low_col], df[close_col], df[volume_col])
    
    return df