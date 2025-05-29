"""
PyIndicators - The most user-friendly technical analysis library for Python!

🚀 **100x easier to use than TA-Lib** with:
- 🧠 Auto-detection of data formats and column names
- 📊 One-line analysis with built-in backtesting
- ⚡ Real-time streaming indicators for live trading
- 📱 Interactive Jupyter widgets and visualizations
- 🔗 Fluent pipelines for complex strategies
- 🏎️ Numba-optimized performance (C-like speed)
- 🐼 Full pandas DataFrame/Series support

Quick Start:
    >>> from pyindicators.easy import analyze
    >>> results = analyze('your_data.csv')  # That's it!

Examples:
    # NumPy (fastest)
    >>> from pyindicators import rsi, macd
    >>> rsi_values = rsi(close_prices, period=14)
    
    # Pandas (most convenient)
    >>> from pyindicators import pandas_wrapper as ta
    >>> df['RSI'] = ta.rsi(df['Close'])
    
    # Smart DataFrame (auto-detection)
    >>> from pyindicators.easy import SmartDataFrame
    >>> sdf = SmartDataFrame('data.csv')
    >>> sdf.add_indicators('all').plot()
    
    # Pipelines (advanced)
    >>> from pyindicators.pipeline import IndicatorPipeline
    >>> signals = (IndicatorPipeline(data)
    ...     .rsi().macd().bollinger_bands()
    ...     .add_signal('buy', lambda df: df['RSI_14'] < 30)
    ...     .get())
    
    # Interactive (Jupyter)
    >>> from pyindicators.widgets import interactive_analysis
    >>> explorer = interactive_analysis('data.csv')
    
    # Streaming (live trading)
    >>> from pyindicators.streaming import LiveTrader
    >>> trader = LiveTrader()
    >>> result = trader.on_tick(live_data)
"""

__version__ = "0.2.0"

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