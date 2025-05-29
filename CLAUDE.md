# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyIndicators is a high-performance Python library for calculating technical indicators, optimized with Numba for C-like speed. The library provides implementations of common technical analysis indicators used in financial markets.

## Architecture

The project follows a modular structure with indicators organized by category:
- `pyindicators/momentum/` - Momentum indicators (RSI, Stochastic, ROC, etc.)
- `pyindicators/trend/` - Trend indicators (SMA, EMA, MACD, ADX, etc.)
- `pyindicators/volatility/` - Volatility indicators (Bollinger Bands, ATR, etc.)
- `pyindicators/volume/` - Volume indicators (OBV, VWAP, MFI, etc.)
- `pyindicators/core/` - Core utility functions and shared components

All indicators are implemented using Numba's JIT compilation for maximum performance.

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