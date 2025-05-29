"""
PyIndicators - High-performance technical indicators library with Numba optimization.

A fast and efficient library for calculating technical indicators used in financial analysis.
Supports both NumPy arrays and pandas DataFrames/Series for maximum flexibility.
"""

__version__ = "0.1.0"

from pyindicators.momentum import (
    rsi,
    stochastic,
    williams_r,
    roc,
    momentum,
)

from pyindicators.trend import (
    sma,
    ema,
    wma,
    macd,
    adx,
    dema,
    tema,
)

from pyindicators.volatility import (
    bollinger_bands,
    atr,
    keltner_channels,
    donchian_channels,
    standard_deviation,
)

from pyindicators.volume import (
    obv,
    vwap,
    ad,
    mfi,
    cmf,
)

__all__ = [
    # Version
    "__version__",
    # Momentum indicators
    "rsi",
    "stochastic",
    "williams_r",
    "roc",
    "momentum",
    # Trend indicators
    "sma",
    "ema",
    "wma",
    "macd",
    "adx",
    "dema",
    "tema",
    # Volatility indicators
    "bollinger_bands",
    "atr",
    "keltner_channels",
    "donchian_channels",
    "standard_deviation",
    # Volume indicators
    "obv",
    "vwap",
    "ad",
    "mfi",
    "cmf",
]