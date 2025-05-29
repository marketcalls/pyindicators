"""Tests for volatility indicators."""

import numpy as np
import pytest

from pyindicators.volatility import (
    bollinger_bands, bollinger_bandwidth, bollinger_percent_b,
    atr, keltner_channels, donchian_channels, standard_deviation
)


class TestBollingerBands:
    def test_bollinger_bands_basic(self):
        close = np.array([20.0, 21.0, 22.0, 21.5, 20.5, 21.0, 22.5, 23.0, 22.0, 21.0])
        upper, middle, lower = bollinger_bands(close, period=5, std_dev=2.0)
        
        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)
        
        # Check that upper > middle > lower
        valid_idx = ~np.isnan(upper)
        assert np.all(upper[valid_idx] >= middle[valid_idx])
        assert np.all(middle[valid_idx] >= lower[valid_idx])
    
    def test_bollinger_bandwidth(self):
        close = np.random.randn(50) + 100
        bandwidth = bollinger_bandwidth(close, period=20)
        
        assert len(bandwidth) == len(close)
        assert np.all(bandwidth[~np.isnan(bandwidth)] >= 0)
    
    def test_bollinger_percent_b(self):
        close = np.random.randn(50) + 100
        percent_b = bollinger_percent_b(close, period=20)
        
        assert len(percent_b) == len(close)


class TestATR:
    def test_atr_basic(self):
        high = np.array([48.70, 48.72, 48.90, 48.87, 48.82, 49.05, 49.20, 49.35, 49.92, 50.19])
        low = np.array([47.79, 48.14, 48.39, 48.37, 48.24, 48.64, 48.94, 48.86, 49.50, 49.87])
        close = np.array([48.16, 48.61, 48.75, 48.63, 48.74, 49.03, 49.07, 49.32, 49.91, 50.13])
        
        result = atr(high, low, close, period=5)
        
        assert len(result) == len(close)
        assert np.all(result[~np.isnan(result)] > 0)
    
    def test_atr_with_gaps(self):
        # Test with price gaps
        high = np.array([100.0, 105.0, 103.0, 102.0, 104.0])
        low = np.array([98.0, 102.0, 101.0, 100.0, 102.0])
        close = np.array([99.0, 104.0, 102.0, 101.0, 103.0])
        
        result = atr(high, low, close, period=3)
        assert len(result) == len(close)


class TestChannels:
    def test_keltner_channels(self):
        high = np.random.randn(50) + 102
        low = np.random.randn(50) + 98
        close = np.random.randn(50) + 100
        
        upper, middle, lower = keltner_channels(high, low, close, period=20)
        
        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)
        
        # Check that upper > middle > lower
        valid_idx = ~np.isnan(upper)
        assert np.all(upper[valid_idx] >= middle[valid_idx])
        assert np.all(middle[valid_idx] >= lower[valid_idx])
    
    def test_donchian_channels(self):
        high = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 16.0, 14.0, 17.0])
        low = np.array([8.0, 9.0, 9.5, 10.0, 11.0, 10.5, 12.0, 13.0, 12.5, 14.0])
        
        upper, middle, lower = donchian_channels(high, low, period=5)
        
        assert len(upper) == len(high)
        assert len(middle) == len(high)
        assert len(lower) == len(low)
        
        # Check specific values
        assert upper[4] == 14.0  # max of high[0:5]
        assert lower[4] == 8.0   # min of low[0:5]


class TestStandardDeviation:
    def test_std_basic(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = standard_deviation(close, period=3)
        
        assert len(result) == len(close)
        assert np.all(result[~np.isnan(result)] >= 0)
    
    def test_std_with_ddof(self):
        close = np.random.randn(50) + 100
        result_pop = standard_deviation(close, period=20, ddof=0)  # Population
        result_sample = standard_deviation(close, period=20, ddof=1)  # Sample
        
        # Sample std should be slightly larger than population std
        valid_idx = ~np.isnan(result_pop)
        assert np.all(result_sample[valid_idx] >= result_pop[valid_idx])