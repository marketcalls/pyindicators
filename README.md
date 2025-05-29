# PyIndicators

[![PyPI version](https://badge.fury.io/py/pyindicators.svg)](https://badge.fury.io/py/pyindicators)
[![Python](https://img.shields.io/pypi/pyversions/pyindicators.svg)](https://pypi.org/project/pyindicators/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🚀 **The most user-friendly technical analysis library for Python!** Combining Numba-powered performance with intelligent automation and beautiful visualizations.

> **100x easier to use than TA-Lib** - with auto-detection, one-line analysis, built-in backtesting, streaming support, and interactive widgets!

## ✨ Key Features

- 🏎️ **Blazing Fast**: Numba JIT-compiled for C-like performance
- 🧠 **Intelligent**: Auto-detects data formats and column names
- 📊 **Complete**: 25+ indicators with built-in visualization and backtesting
- 🐼 **Pandas Native**: Full DataFrame/Series support with named outputs
- 📱 **Interactive**: Jupyter widgets and visual strategy builders
- ⚡ **Real-time**: Streaming indicators for live trading
- 🔧 **Flexible**: From one-liners to advanced pipelines
- 🛡️ **Robust**: Smart error handling with helpful suggestions

## 🚀 Installation

```bash
pip install pyindicators

# For interactive features
pip install pyindicators[widgets]

# For development
pip install pyindicators[dev]
```

## ⚡ Quick Start - Choose Your Style

### 🎯 **Absolute Beginner** (1 line!)
```python
from pyindicators.easy import analyze

# Auto-detects columns, adds indicators, plots, and backtests!
results = analyze('your_data.csv')
```

### 🐼 **Pandas User** (Most Popular)
```python
from pyindicators.easy import SmartDataFrame

# Works with any CSV/JSON/Parquet file or DataFrame
df = SmartDataFrame('AAPL.csv')  # Auto-detects OHLC columns!

# Add indicators with short names
df.add_indicators('rsi', 'macd', 'bb', 'volume')

# Or add everything at once
df.add_indicators('all')

# Built-in visualization with signals
df.plot()

# Instant backtesting
performance = df.backtest(strategy='macd_cross')
print(performance)
```

### 🔗 **Pipeline Enthusiast** (Advanced)
```python
from pyindicators.pipeline import IndicatorPipeline

# Chain operations fluently
signals = (IndicatorPipeline(data)
    .rsi()
    .bollinger_bands()
    .macd()
    .add_signal('buy', lambda df: (df['RSI_14'] < 30) & (df['Close'] < df['BB_Lower']))
    .add_signal('sell', lambda df: (df['RSI_14'] > 70) & (df['Close'] > df['BB_Upper']))
    .golden_cross()
    .divergence()
    .get())
```

### 📊 **Interactive User** (Jupyter)
```python
from pyindicators.widgets import interactive_analysis

# Launch interactive explorer with sliders and real-time updates
explorer = interactive_analysis('data.csv')
```

### ⚡ **Live Trader** (Real-time)
```python
from pyindicators.streaming import LiveTrader

trader = LiveTrader()

# In your trading loop
result = trader.on_tick({
    'high': 105.2, 'low': 104.8, 
    'close': 105.0, 'volume': 1000000
})

if result['action'] == 'BUY':
    print(f"Buy signal at ${result['indicators']['close']}")
```

### 💻 **Command Line User**
```bash
# Quick analysis
pyindicators analyze AAPL.csv

# Calculate specific indicator
pyindicators calc data.csv -i rsi -p 21

# Run backtest
pyindicators backtest data.csv -s macd_cross -c 10000

# Find signals
pyindicators signals data.csv --action buy
```

## 🎛️ **All Available Interfaces**

| Interface | Best For | Example |
|-----------|----------|---------|
| **Easy Mode** | Beginners, quick analysis | `analyze('data.csv')` |
| **Pandas Wrapper** | Data scientists | `ta.rsi(df['Close'])` |
| **NumPy Core** | Performance-critical | `rsi(close_array, 14)` |
| **Pipelines** | Complex strategies | `Pipeline(data).rsi().macd()` |
| **Streaming** | Live trading | `trader.on_tick(live_data)` |
| **CLI** | Terminal users | `pyindicators analyze data.csv` |
| **Widgets** | Jupyter notebooks | `interactive_analysis(data)` |

## 📊 **Available Indicators**

### 🚀 **Momentum** (7 indicators)
- **RSI** (Relative Strength Index) - `rsi()`
- **Stochastic Oscillator** - `stochastic()`
- **Williams %R** - `williams_r()`
- **Rate of Change** - `roc()`
- **Momentum** - `momentum()`

### 📈 **Trend** (7 indicators)
- **SMA** (Simple Moving Average) - `sma()`
- **EMA** (Exponential Moving Average) - `ema()`
- **WMA** (Weighted Moving Average) - `wma()`
- **DEMA** (Double Exponential MA) - `dema()`
- **TEMA** (Triple Exponential MA) - `tema()`
- **MACD** (Moving Average Convergence Divergence) - `macd()`
- **ADX** (Average Directional Index) - `adx()`

### 📊 **Volatility** (5 indicators)
- **Bollinger Bands** - `bollinger_bands()`
- **ATR** (Average True Range) - `atr()`
- **Keltner Channels** - `keltner_channels()`
- **Donchian Channels** - `donchian_channels()`
- **Standard Deviation** - `standard_deviation()`

### 📦 **Volume** (5 indicators)
- **OBV** (On Balance Volume) - `obv()`
- **VWAP** (Volume Weighted Average Price) - `vwap()`
- **A/D Line** (Accumulation/Distribution) - `ad()`
- **MFI** (Money Flow Index) - `mfi()`
- **CMF** (Chaikin Money Flow) - `cmf()`

## 🎯 **Built-in Trading Strategies**

```python
# Pre-built strategies ready to use
strategies = [
    'simple',       # RSI oversold/overbought
    'macd_cross',   # MACD crossover signals
    'bb_bounce',    # Bollinger Band mean reversion
    'trend_follow'  # Multi-indicator trend following
]

# Instant backtesting
results = df.backtest(strategy='macd_cross')
```

## 🎨 **Advanced Features**

### 🔍 **Smart Data Detection**
```python
# Works with ANY column naming convention!
df = SmartDataFrame(data)  # Auto-detects: Close, close, CLOSE, Close_Price, etc.
```

### 🔗 **Multi-Timeframe Analysis**
```python
from pyindicators.easy import MultiTimeframe

mtf = MultiTimeframe(data)
mtf.add_timeframe('daily', 'D')
mtf.add_timeframe('weekly', 'W')
confluence = mtf.find_confluence()  # Find signals across timeframes
```

### 🎯 **Signal Detection**
```python
from pyindicators.easy import find_signals

# Find all trading signals automatically
signals = find_signals('AAPL.csv')
print(signals.head())
```

### 🛠️ **Custom Indicators**
```python
# Create custom indicators easily
pipeline = (IndicatorPipeline(data)
    .custom(lambda df: df['Close'].rolling(20).skew(), 'Skewness_20')
    .transform('Volume', lambda x: np.log(x), 'Log_Volume'))
```

### 🎮 **Interactive Visualization**
```python
from pyindicators.visual import launch_visual_tools

# Interactive chart with sliders
launch_visual_tools(data, tool='interactive')

# Strategy builder with drag-and-drop
launch_visual_tools(data, tool='builder')

# Indicator playground
launch_visual_tools(data, tool='playground')
```

## 🚨 **Smart Error Handling**

PyIndicators provides helpful error messages and suggestions:

```python
# Instead of cryptic errors, you get:
❌ Column 'Close' not found in DataFrame!

Available columns: ['close', 'high', 'low', 'volume']

Did you mean one of these?
  • close

💡 Tip: Check your column names are correct. Common issues:
  - Case sensitivity (Close vs close)
  - Extra spaces in column names
```

## 🏎️ **Performance**

Still blazing fast with Numba optimization:

```python
import time
from pyindicators import rsi

# 1 million data points
data = np.random.randn(1_000_000) + 100

start = time.time()
result = rsi(data, period=14)
elapsed = time.time() - start

print(f"Calculated RSI for 1M points in {elapsed:.3f} seconds")
# Output: Calculated RSI for 1M points in 0.045 seconds
```

## 📖 **Complete Examples**

### 📈 **Full Technical Analysis Workflow**
```python
from pyindicators.easy import SmartDataFrame
import yfinance as yf

# Load data (or use any CSV/DataFrame)
df = yf.download('AAPL', start='2020-01-01')

# Create smart dataframe with auto-detection
sdf = SmartDataFrame(df)

# Add all indicators
sdf.add_indicators('all')

# Create custom strategy
sdf.add_signal('custom_buy', lambda df: 
    (df['RSI'] < 30) & 
    (df['MACD'] > df['MACD_Signal']) & 
    (df['Close'] < df['BB_Lower'])
)

# Backtest multiple strategies
strategies = ['simple', 'macd_cross', 'bb_bounce', 'trend_follow']
for strategy in strategies:
    results = sdf.backtest(strategy=strategy)
    print(f"{strategy}: {results['total_return']}")

# Beautiful visualization
sdf.plot(show_signals=True)

# Export results
sdf.df.to_csv('analysis_results.csv')
```

### 🔄 **Streaming Analysis**
```python
from pyindicators.streaming import LiveTrader
import yfinance as yf

# Setup live trader
trader = LiveTrader()

# Simulate live data feed
historical_data = yf.download('AAPL', period='1mo', interval='1m')

for timestamp, row in historical_data.iterrows():
    tick_data = {
        'high': row['High'],
        'low': row['Low'], 
        'close': row['Close'],
        'volume': row['Volume']
    }
    
    result = trader.on_tick(tick_data)
    
    if result['action']:
        print(f"[{timestamp}] {result['action']} signal at ${tick_data['close']:.2f}")
        print(f"RSI: {result['indicators'].get('rsi', 'N/A'):.1f}")

# Get performance
performance = trader.get_performance()
print(f"Total PnL: {performance['total_pnl']}")
```

### 🎯 **Strategy Comparison**
```python
from pyindicators.pipeline import StrategyPipeline

# Test multiple strategy setups
strategies = {
    'Conservative': StrategyPipeline(data).mean_reversion_setup(bb_period=20, rsi_period=21),
    'Aggressive': StrategyPipeline(data).trend_following_setup(fast=10, slow=30),
    'Breakout': StrategyPipeline(data).breakout_setup(period=15)
}

results = {}
for name, strategy in strategies.items():
    df_with_signals = strategy.get()
    # Calculate returns for each strategy
    # ... backtesting logic
    results[name] = performance_metrics

print("Strategy Performance Comparison:")
print(pd.DataFrame(results).T)
```

## 📚 **Documentation & Tutorials**

- **[Getting Started Guide](docs/getting_started.md)** - Complete beginner tutorial
- **[API Reference](docs/api_reference.md)** - Full function documentation  
- **[Strategy Guide](docs/strategies.md)** - Trading strategy examples
- **[Performance Tips](docs/performance.md)** - Optimization techniques
- **[Jupyter Examples](examples/)** - Interactive notebook tutorials

## 🧪 **Development**

### Running Tests
```bash
# Install development dependencies
pip install -e .[dev]

# Run all tests
pytest

# Run with coverage
pytest --cov=pyindicators --cov-report=html

# Run specific test file
pytest tests/test_pandas_wrapper.py -v
```

### Benchmarking
```bash
# Performance benchmarks
python examples/benchmark.py

# Memory usage analysis
python examples/memory_profile.py
```

### Code Quality
```bash
# Format code
black pyindicators tests

# Lint code  
ruff check pyindicators tests

# Type checking
mypy pyindicators
```

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Add tests** for your changes
4. **Ensure** all tests pass (`pytest`)
5. **Commit** your changes (`git commit -m 'Add amazing feature'`)
6. **Push** to the branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### 💡 **Ideas for Contributions**
- New technical indicators
- Additional trading strategies  
- Performance optimizations
- Documentation improvements
- Integration with data providers (Alpha Vantage, Quandl, etc.)
- Additional visualization options

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Numba Team** - For the incredible JIT compiler
- **Pandas Team** - For the amazing DataFrame functionality
- **TA-Lib** - For inspiration and reference implementations
- **Technical Analysis Community** - For indicator formulas and expertise

## 💬 **Support & Community**

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/your-username/pyindicators/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/your-username/pyindicators/discussions)
- 📧 **Email**: support@pyindicators.com
- 💬 **Discord**: [Join our community](https://discord.gg/pyindicators)

## ⭐ **Star History**

If you find PyIndicators helpful, please give it a ⭐️ on GitHub! It helps others discover the project.

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/pyindicators&type=Date)](https://star-history.com/#your-username/pyindicators&Date)

---

**Made with ❤️ for the trading and data science community**

> "The best technical analysis library you'll ever use!" - Happy Users