"""Tests for volume indicators."""

import numpy as np
import pytest

from pyindicators.volume import obv, vwap, ad, mfi, cmf


class TestOBV:
    def test_obv_basic(self):
        close = np.array([10.0, 10.5, 10.3, 10.8, 10.2, 10.7, 11.0, 10.9, 11.2, 11.0])
        volume = np.array([1000, 1200, 1100, 1300, 900, 1400, 1600, 1500, 1700, 1200])
        
        result = obv(close, volume)
        
        assert len(result) == len(close)
        assert result[0] == 0  # Initial value
        assert result[1] == 1200  # Price up, add volume
        assert result[2] == 100   # Price down, subtract volume
    
    def test_obv_with_initial_value(self):
        close = np.array([100.0, 101.0, 99.0])
        volume = np.array([1000, 2000, 1500])
        
        result = obv(close, volume, initial_value=5000)
        
        assert result[0] == 5000
        assert result[1] == 7000  # 5000 + 2000
        assert result[2] == 5500  # 7000 - 1500


class TestVWAP:
    def test_vwap_cumulative(self):
        high = np.array([101.0, 102.0, 103.0, 102.5, 101.5])
        low = np.array([99.0, 100.0, 101.0, 100.5, 99.5])
        close = np.array([100.0, 101.0, 102.0, 101.5, 100.5])
        volume = np.array([1000, 1500, 2000, 1200, 1800])
        
        result = vwap(high, low, close, volume)
        
        assert len(result) == len(close)
        assert np.all(~np.isnan(result))
    
    def test_vwap_anchored(self):
        high = np.random.randn(20) + 102
        low = np.random.randn(20) + 98
        close = np.random.randn(20) + 100
        volume = np.random.randint(1000, 5000, 20)
        
        # Create anchor points (reset every 5 bars)
        anchor_points = np.zeros(20, dtype=bool)
        anchor_points[::5] = True
        
        result = vwap(high, low, close, volume, anchor_points)
        
        assert len(result) == len(close)


class TestAD:
    def test_ad_basic(self):
        high = np.array([127.01, 127.62, 126.59, 127.35, 128.17])
        low = np.array([125.36, 126.16, 124.93, 126.09, 126.82])
        close = np.array([125.84, 126.98, 126.52, 126.75, 127.16])
        volume = np.array([10000, 12000, 11000, 13000, 14000])
        
        result = ad(high, low, close, volume)
        
        assert len(result) == len(close)
        assert result[0] != 0  # Should have some value based on MFM
    
    def test_ad_no_range(self):
        # Test when high == low
        high = np.array([100.0, 100.0, 100.0])
        low = np.array([100.0, 100.0, 100.0])
        close = np.array([100.0, 100.0, 100.0])
        volume = np.array([1000, 1000, 1000])
        
        result = ad(high, low, close, volume)
        assert np.all(result == 0)


class TestMFI:
    def test_mfi_basic(self):
        high = np.random.randn(50) + 102
        low = np.random.randn(50) + 98
        close = np.random.randn(50) + 100
        volume = np.random.randint(1000, 10000, 50)
        
        result = mfi(high, low, close, volume, period=14)
        
        assert len(result) == len(close)
        assert np.all((result[~np.isnan(result)] >= 0) & 
                     (result[~np.isnan(result)] <= 100))
    
    def test_mfi_extreme_values(self):
        # All prices increasing
        high = np.arange(50, dtype=float) + 100
        low = np.arange(50, dtype=float) + 98
        close = np.arange(50, dtype=float) + 99
        volume = np.ones(50) * 1000
        
        result = mfi(high, low, close, volume, period=14)
        
        # Should be near 100 (all positive money flow)
        assert np.all(result[14:] > 90)


class TestCMF:
    def test_cmf_basic(self):
        high = np.random.randn(50) + 102
        low = np.random.randn(50) + 98
        close = np.random.randn(50) + 100
        volume = np.random.randint(1000, 10000, 50)
        
        result = cmf(high, low, close, volume, period=20)
        
        assert len(result) == len(close)
        assert np.all((result[~np.isnan(result)] >= -1) & 
                     (result[~np.isnan(result)] <= 1))
    
    def test_cmf_zero_volume(self):
        high = np.array([101.0, 102.0, 103.0])
        low = np.array([99.0, 100.0, 101.0])
        close = np.array([100.0, 101.0, 102.0])
        volume = np.array([0, 0, 0])
        
        result = cmf(high, low, close, volume, period=2)
        assert np.all(result[~np.isnan(result)] == 0)