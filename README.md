# PyIndicators

[![PyPI version](https://badge.fury.io/py/pyindicators.svg)](https://badge.fury.io/py/pyindicators)
[![Python](https://img.shields.io/pypi/pyversions/pyindicators.svg)](https://pypi.org/project/pyindicators/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance Python library for calculating technical indicators, optimized with Numba for speed comparable to C implementations.

## Features

- **Blazing Fast**: JIT-compiled with Numba for C-like performance
- **Comprehensive**: Includes momentum, trend, volatility, and volume indicators
- **Easy to Use**: Simple API with NumPy array inputs/outputs
- **Well Tested**: Extensive test coverage
- **Lightweight**: Minimal dependencies (NumPy and Numba)

## Installation

```bash
pip install pyindicators
```

## Quick Start

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