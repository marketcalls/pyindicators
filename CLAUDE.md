# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyIndicators is the most user-friendly technical analysis library for Python, designed to be 100x easier to use than TA-Lib while maintaining C-like performance through Numba optimization. 

### Key Innovations
- **Auto-detection**: Automatically detects OHLC column names regardless of format
- **One-line analysis**: Complete technical analysis with `analyze('data.csv')`
- **Multiple interfaces**: From beginner-friendly to expert-level pipelines
- **Built-in backtesting**: Instant strategy performance evaluation
- **Streaming support**: Real-time indicators for live trading
- **Interactive widgets**: Jupyter notebook integration with sliders
- **Smart error handling**: Helpful messages instead of cryptic errors

## Architecture

The project follows a modular structure with multiple interface layers:

### Core Indicators (Numba-optimized)
- `pyindicators/momentum/` - Momentum indicators (RSI, Stochastic, ROC, etc.)
- `pyindicators/trend/` - Trend indicators (SMA, EMA, MACD, ADX, etc.)
- `pyindicators/volatility/` - Volatility indicators (Bollinger Bands, ATR, etc.)
- `pyindicators/volume/` - Volume indicators (OBV, VWAP, MFI, etc.)
- `pyindicators/core/` - Core utility functions and shared components

### User-Friendly Interfaces (New!)
- `pyindicators/pandas_wrapper.py` - Pandas DataFrame/Series support with named outputs
- `pyindicators/easy.py` - One-line analysis and SmartDataFrame with auto-detection
- `pyindicators/pipeline.py` - Fluent API for chaining indicators and creating strategies
- `pyindicators/streaming.py` - Real-time streaming indicators for live trading
- `pyindicators/visual.py` - Interactive charts and strategy builders
- `pyindicators/widgets.py` - Jupyter notebook widgets with sliders
- `pyindicators/cli.py` - Command-line interface for terminal users
- `pyindicators/errors.py` - Smart error handling with helpful suggestions

All core calculations use Numba JIT compilation for maximum performance while the new interfaces provide unmatched ease of use.

## Development Commands

### Setup Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dependencies
pip install -e .[dev]
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyindicators

# Run specific test file
pytest tests/test_momentum.py

# Run with verbose output
pytest -v
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

### Building and Publishing
```bash
# Build distribution
python -m build

# Test upload to PyPI
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Upload to PyPI
twine upload dist/*
```

## Key Design Principles

1. **Performance First**: All indicators use Numba JIT compilation for C-like speed
2. **NumPy Arrays**: All inputs and outputs are NumPy arrays for efficiency
3. **Consistent API**: All indicators follow the pattern: `indicator(data, period, **kwargs)`
4. **NaN Handling**: Leading NaN values for periods where calculation isn't possible
5. **No Dependencies**: Only NumPy and Numba required (minimal dependencies)

## Adding New Indicators

1. Choose the appropriate category (momentum, trend, volatility, volume)
2. Create a new file in the category directory
3. Implement the indicator function with `@njit` decorator
4. Add comprehensive docstring with parameters and examples
5. Export the function in the category's `__init__.py`
6. Add to the main `pyindicators/__init__.py`
7. Write tests in the corresponding test file
8. Update README.md with the new indicator

Example structure:
```python
@njit(cache=True)
def new_indicator(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate New Indicator.
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
    period : int, default 14
        Period for calculation
        
    Returns
    -------
    np.ndarray
        Indicator values
    """
    # Implementation
```

## Performance Considerations

- Use `@njit(cache=True)` for all calculation functions
- Avoid Python objects inside Numba functions
- Pre-allocate arrays when possible
- Use `prange` for parallel loops when beneficial
- Minimize function calls inside hot loops