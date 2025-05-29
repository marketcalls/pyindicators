"""Tests for momentum indicators."""

import numpy as np
import pytest

from pyindicators.momentum import (
    rsi, stochastic, stochastic_fast, williams_r, roc, momentum
)


class TestRSI:
    def test_rsi_basic(self):
        close = np.array([44.0, 44.25, 44.5, 43.75, 44.75, 45.5, 45.25, 46.0, 46.5, 46.25])
        result = rsi(close, period=5)
        
        assert len(result) == len(close)
        assert np.all(np.isnan(result[:4]))
        assert np.all((result[4:] >= 0) & (result[4:] <= 100))
    
    def test_rsi_all_up(self):
        close = np.arange(100, dtype=float)
        result = rsi(close, period=14)
        
        assert np.all(result[13:] > 70)  # Should be overbought
    
    def test_rsi_all_down(self):
        close = np.arange(100, 0, -1, dtype=float)
        result = rsi(close, period=14)
        
        assert np.all(result[13:] < 30)  # Should be oversold


class TestStochastic:
    def test_stochastic_basic(self):
        high = np.array([127.01, 127.62, 126.59, 127.35, 128.17, 128.43, 127.37, 126.42, 126.90, 126.85])
        low = np.array([125.36, 126.16, 124.93, 126.09, 126.82, 126.48, 126.03, 124.83, 126.39, 125.72])
        close = np.array([125.84, 126.98, 126.52, 126.75, 127.16, 127.29, 127.18, 125.86, 126.85, 126.23])
        
        k, d = stochastic(high, low, close, k_period=5, k_smooth=3, d_period=3)
        
        assert len(k) == len(close)
        assert len(d) == len(close)
        assert np.all((k[~np.isnan(k)] >= 0) & (k[~np.isnan(k)] <= 100))
        assert np.all((d[~np.isnan(d)] >= 0) & (d[~np.isnan(d)] <= 100))
    
    def test_stochastic_fast(self):
        high = np.random.randn(50) + 102
        low = np.random.randn(50) + 98
        close = np.random.randn(50) + 100
        
        k, d = stochastic_fast(high, low, close, period=14)
        
        assert len(k) == len(close)
        assert len(d) == len(close)


class TestWilliamsR:
    def test_williams_r_basic(self):
        high = np.array([127.01, 127.62, 126.59, 127.35, 128.17, 128.43, 127.37, 126.42, 126.90, 126.85])
        low = np.array([125.36, 126.16, 124.93, 126.09, 126.82, 126.48, 126.03, 124.83, 126.39, 125.72])
        close = np.array([125.84, 126.98, 126.52, 126.75, 127.16, 127.29, 127.18, 125.86, 126.85, 126.23])
        
        result = williams_r(high, low, close, period=5)
        
        assert len(result) == len(close)
        assert np.all((result[~np.isnan(result)] >= -100) & (result[~np.isnan(result)] <= 0))


class TestROCMomentum:
    def test_roc_basic(self):
        close = np.array([100.0, 102.0, 101.0, 103.0, 104.0, 102.0, 105.0, 106.0, 104.0, 107.0])
        result = roc(close, period=5)
        
        assert len(result) == len(close)
        assert np.all(np.isnan(result[:5]))
    
    def test_momentum_basic(self):
        close = np.array([100.0, 102.0, 101.0, 103.0, 104.0, 102.0, 105.0, 106.0, 104.0, 107.0])
        result = momentum(close, period=5)
        
        assert len(result) == len(close)
        assert np.all(np.isnan(result[:5]))
        assert result[5] == close[5] - close[0]