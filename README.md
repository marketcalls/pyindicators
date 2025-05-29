# PyIndicators

[![PyPI version](https://badge.fury.io/py/pyindicators.svg)](https://badge.fury.io/py/pyindicators)
[![Python](https://img.shields.io/pypi/pyversions/pyindicators.svg)](https://pypi.org/project/pyindicators/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance Python library for calculating technical indicators, optimized with Numba for speed comparable to C implementations. Now with full pandas DataFrame and Series support for maximum ease of use!

## Features

- **Blazing Fast**: JIT-compiled with Numba for C-like performance
- **Comprehensive**: Includes momentum, trend, volatility, and volume indicators
- **Easy to Use**: Simple API with NumPy arrays or pandas DataFrames/Series
- **Pandas Support**: Seamless integration with pandas for data analysis workflows
- **Well Tested**: Extensive test coverage
- **Lightweight**: Minimal dependencies (NumPy and Numba, pandas optional)

## Installation

```bash
pip install pyindicators
```

## Quick Start

### Using NumPy Arrays (Fastest)

```python
import numpy as np
from pyindicators import rsi, sma, bollinger_bands, macd

# Generate sample data
close_prices = np.random.randn(100) + 100

# Calculate indicators
rsi_values = rsi(close_prices, period=14)
sma_values = sma(close_prices, period=20)
upper, middle, lower = bollinger_bands(close_prices, period=20, std_dev=2)
macd_line, signal_line, histogram = macd(close_prices)
```

### Using Pandas DataFrames (Most Convenient)

```python
import pandas as pd
from pyindicators import pandas_wrapper as ta

# Load your data
df = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)

# Calculate indicators - returns pandas Series with proper names
df['RSI'] = ta.rsi(df['Close'], period=14)
df['SMA_20'] = ta.sma(df['Close'], period=20)
df['EMA_20'] = ta.ema(df['Close'], period=20)

# Multiple outputs returned as tuple of Series
df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.bollinger_bands(df['Close'])
df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ta.macd(df['Close'])

# Or add all indicators at once!
df_with_indicators = ta.add_all_indicators(df)
```

## Available Indicators

### Momentum Indicators
- **RSI** (Relative Strength Index)
- **Stochastic Oscillator** (Fast & Slow)
- **Williams %R**
- **ROC** (Rate of Change)
- **Momentum**

### Trend Indicators
- **SMA** (Simple Moving Average)
- **EMA** (Exponential Moving Average)
- **WMA** (Weighted Moving Average)
- **DEMA** (Double Exponential Moving Average)
- **TEMA** (Triple Exponential Moving Average)
- **MACD** (Moving Average Convergence Divergence)
- **ADX** (Average Directional Index)

### Volatility Indicators
- **Bollinger Bands**
- **ATR** (Average True Range)
- **Keltner Channels**
- **Donchian Channels**
- **Standard Deviation**

### Volume Indicators
- **OBV** (On Balance Volume)
- **VWAP** (Volume Weighted Average Price)
- **A/D** (Accumulation/Distribution)
- **MFI** (Money Flow Index)
- **CMF** (Chaikin Money Flow)

## Performance

PyIndicators leverages Numba's JIT compilation to achieve performance comparable to C implementations:

```python
import numpy as np
import time
from pyindicators import rsi

# Large dataset
data = np.random.randn(1_000_000) + 100

# Measure performance
start = time.time()
result = rsi(data, period=14)
elapsed = time.time() - start

print(f"Calculated RSI for 1M data points in {elapsed:.3f} seconds")
```

## Examples

### Working with Pandas DataFrames

```python
import pandas as pd
import yfinance as yf
from pyindicators import pandas_wrapper as ta

# Download stock data
df = yf.download('AAPL', start='2023-01-01', end='2024-01-01')

# Calculate various indicators
df['RSI'] = ta.rsi(df['Close'])
df['MACD'], df['Signal'], df['Histogram'] = ta.macd(df['Close'])
df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.bollinger_bands(df['Close'])
df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])

# Create trading signals
buy_signal = (df['RSI'] < 30) & (df['Close'] < df['BB_Lower'])
sell_signal = (df['RSI'] > 70) & (df['Close'] > df['BB_Upper'])

# Add all indicators with a prefix
df_full = ta.add_all_indicators(df, prefix='TA_')
```

### Calculate Multiple Indicators

```python
import numpy as np
from pyindicators import rsi, macd, bollinger_bands, atr

# Sample OHLC data
n = 1000
high = np.random.randn(n) + 102
low = np.random.randn(n) + 98
close = np.random.randn(n) + 100
volume = np.random.randint(100000, 1000000, n)

# Momentum indicator
rsi_values = rsi(close, period=14)

# Trend indicator
macd_line, signal, histogram = macd(close)

# Volatility indicators
upper, middle, lower = bollinger_bands(close, period=20)
atr_values = atr(high, low, close, period=14)
```

### Custom Indicator Combinations

```python
from pyindicators import ema, atr, adx

def trend_strength_indicator(high, low, close, period=14):
    """Custom indicator combining ADX and ATR."""
    adx_values = adx(high, low, close, period)
    atr_values = atr(high, low, close, period)
    ema_values = ema(close, period)
    
    # Normalize ATR by EMA
    normalized_atr = atr_values / ema_values * 100
    
    # Combine ADX and normalized ATR
    trend_strength = adx_values * (1 + normalized_atr / 100)
    
    return trend_strength
```

## API Reference

All indicators follow a consistent API pattern:

```python
indicator_name(data, period=default_period, **kwargs)
```

- **data**: NumPy array of price data (close, high, low, or volume as required)
- **period**: Lookback period for the indicator
- **kwargs**: Additional parameters specific to each indicator

### Example: RSI

```python
rsi(close, period=14, min_periods=None)
```

Parameters:
- `close`: Array of closing prices
- `period`: RSI period (default: 14)
- `min_periods`: Minimum periods required for calculation

Returns:
- NumPy array of RSI values (0-100)

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run with coverage
pytest --cov=pyindicators
```

### Benchmarking

```bash
python examples/benchmark.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Numba team for the amazing JIT compiler
- Technical analysis community for indicator formulas and algorithms

## Support

If you find this project helpful, please give it a ⭐️ on GitHub!