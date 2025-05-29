"""Tests for trend indicators."""

import numpy as np
import pytest

from pyindicators.trend import (
    sma, ema, wma, dema, tema, macd, macd_signal, macd_histogram,
    adx, plus_di, minus_di
)


class TestMovingAverages:
    def test_sma_basic(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = sma(close, period=3)
        
        assert len(result) == len(close)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 2.0  # (1+2+3)/3
        assert result[3] == 3.0  # (2+3+4)/3
    
    def test_ema_basic(self):
        close = np.arange(50, dtype=float)
        result = ema(close, period=10)
        
        assert len(result) == len(close)
        assert result[0] == close[0]
        assert np.all(~np.isnan(result))
    
    def test_wma_basic(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wma(close, period=3)
        
        assert len(result) == len(close)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # WMA = (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6 = 2.333...
        assert np.isclose(result[2], 2.333333, rtol=1e-5)
    
    def test_dema_tema(self):
        close = np.random.randn(100) + 100
        dema_result = dema(close, period=20)
        tema_result = tema(close, period=20)
        
        assert len(dema_result) == len(close)
        assert len(tema_result) == len(close)


class TestMACD:
    def test_macd_basic(self):
        close = np.random.randn(100) + 100
        macd_line, signal, histogram = macd(close)
        
        assert len(macd_line) == len(close)
        assert len(signal) == len(close)
        assert len(histogram) == len(close)
        assert np.allclose(histogram[~np.isnan(histogram)], 
                          (macd_line - signal)[~np.isnan(histogram)])
    
    def test_macd_custom_periods(self):
        close = np.random.randn(100) + 100
        macd_line, signal, histogram = macd(close, fast_period=10, slow_period=20, signal_period=5)
        
        assert len(macd_line) == len(close)
    
    def test_macd_invalid_periods(self):
        close = np.random.randn(100) + 100
        with pytest.raises(ValueError):
            macd(close, fast_period=26, slow_period=12)


class TestADX:
    def test_directional_indicators(self):
        high = np.random.randn(50) + 102
        low = np.random.randn(50) + 98
        close = np.random.randn(50) + 100
        
        plus_di_result = plus_di(high, low, close, period=14)
        minus_di_result = minus_di(high, low, close, period=14)
        
        assert len(plus_di_result) == len(close)
        assert len(minus_di_result) == len(close)
        assert np.all((plus_di_result[~np.isnan(plus_di_result)] >= 0) & 
                     (plus_di_result[~np.isnan(plus_di_result)] <= 100))
    
    def test_adx_basic(self):
        high = np.random.randn(50) + 102
        low = np.random.randn(50) + 98
        close = np.random.randn(50) + 100
        
        adx_result = adx(high, low, close, period=14)
        
        assert len(adx_result) == len(close)
        assert np.all((adx_result[~np.isnan(adx_result)] >= 0) & 
                     (adx_result[~np.isnan(adx_result)] <= 100))