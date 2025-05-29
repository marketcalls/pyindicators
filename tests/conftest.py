"""Pytest configuration and fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    
    # Generate realistic price movements
    returns = np.random.normal(0.0002, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate high/low based on close
    daily_range = np.random.uniform(0.005, 0.02, n)
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    
    # Ensure open is between high and low
    open_price = low + np.random.uniform(0.3, 0.7, n) * (high - low)
    
    # Generate volume
    volume = np.random.randint(100000, 1000000, n)
    
    return {
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }


@pytest.fixture
def trending_data():
    """Generate trending price data."""
    n = 100
    trend = np.linspace(100, 150, n)
    noise = np.random.normal(0, 2, n)
    return trend + noise


@pytest.fixture
def sideways_data():
    """Generate sideways/ranging price data."""
    n = 100
    base = 100
    return base + np.random.normal(0, 2, n)