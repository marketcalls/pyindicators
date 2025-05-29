"""Tests for pandas wrapper functionality."""

import pytest
import numpy as np
import pandas as pd
from pyindicators import pandas_wrapper as indicators


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data as pandas DataFrame."""
    n = 100
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    
    daily_range = np.random.uniform(0.005, 0.02, n)
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_price = low + np.random.uniform(0.3, 0.7, n) * (high - low)
    
    volume = 1000000 + np.abs(returns) * 1000000 * 50
    
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume.astype(int)
    }, index=dates)
    
    return df


class TestPandasWrapper:
    """Test pandas wrapper functionality."""
    
    def test_series_input_output(self, sample_data):
        """Test that Series input returns Series output."""
        close_series = sample_data['Close']
        
        # Test single output indicators
        result = indicators.rsi(close_series)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close_series)
        assert result.index.equals(close_series.index)
        assert result.name == 'RSI_14'
        
        # Test with different period
        result = indicators.sma(close_series, period=50)
        assert result.name == 'SMA_50'
    
    def test_array_input_output(self, sample_data):
        """Test that array input returns array output."""
        close_array = sample_data['Close'].to_numpy()
        
        result = indicators.rsi(close_array)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(close_array)
    
    def test_multiple_output_indicators(self, sample_data):
        """Test indicators that return multiple outputs."""
        df = sample_data
        
        # Test Bollinger Bands
        upper, middle, lower = indicators.bollinger_bands(df['Close'])
        assert all(isinstance(x, pd.Series) for x in [upper, middle, lower])
        assert upper.name == 'BB_Upper'
        assert middle.name == 'BB_Middle'
        assert lower.name == 'BB_Lower'
        
        # Test MACD
        macd, signal, hist = indicators.macd(df['Close'])
        assert all(isinstance(x, pd.Series) for x in [macd, signal, hist])
        assert macd.name == 'MACD'
        assert signal.name == 'MACD_Signal'
        assert hist.name == 'MACD_Histogram'
        
        # Test Stochastic
        k, d = indicators.stochastic(df['High'], df['Low'], df['Close'])
        assert all(isinstance(x, pd.Series) for x in [k, d])
        assert k.name == 'Stoch_K_14'
        assert d.name == 'Stoch_D_3'
    
    def test_momentum_indicators(self, sample_data):
        """Test all momentum indicators."""
        df = sample_data
        
        # RSI
        rsi_result = indicators.rsi(df['Close'], period=21)
        assert rsi_result.name == 'RSI_21'
        assert np.all((rsi_result[~np.isnan(rsi_result)] >= 0) & 
                     (rsi_result[~np.isnan(rsi_result)] <= 100))
        
        # ROC
        roc_result = indicators.roc(df['Close'], period=12)
        assert roc_result.name == 'ROC_12'
        
        # Momentum
        mom_result = indicators.momentum(df['Close'], period=10)
        assert mom_result.name == 'MOM_10'
        
        # Williams %R
        williams_result = indicators.williams_r(df['High'], df['Low'], df['Close'])
        assert williams_result.name == 'Williams_R_14'
        assert np.all((williams_result[~np.isnan(williams_result)] >= -100) & 
                     (williams_result[~np.isnan(williams_result)] <= 0))
    
    def test_trend_indicators(self, sample_data):
        """Test all trend indicators."""
        df = sample_data
        
        # Moving averages
        sma_result = indicators.sma(df['Close'], period=30)
        assert sma_result.name == 'SMA_30'
        
        ema_result = indicators.ema(df['Close'], period=30)
        assert ema_result.name == 'EMA_30'
        
        wma_result = indicators.wma(df['Close'], period=30)
        assert wma_result.name == 'WMA_30'
        
        dema_result = indicators.dema(df['Close'], period=30)
        assert dema_result.name == 'DEMA_30'
        
        tema_result = indicators.tema(df['Close'], period=30)
        assert tema_result.name == 'TEMA_30'
        
        # ADX
        adx_result = indicators.adx(df['High'], df['Low'], df['Close'])
        assert adx_result.name == 'ADX_14'
        assert np.all((adx_result[~np.isnan(adx_result)] >= 0) & 
                     (adx_result[~np.isnan(adx_result)] <= 100))
    
    def test_volatility_indicators(self, sample_data):
        """Test all volatility indicators."""
        df = sample_data
        
        # ATR
        atr_result = indicators.atr(df['High'], df['Low'], df['Close'], period=21)
        assert atr_result.name == 'ATR_21'
        assert np.all(atr_result[~np.isnan(atr_result)] >= 0)
        
        # Keltner Channels
        kc_upper, kc_middle, kc_lower = indicators.keltner_channels(
            df['High'], df['Low'], df['Close']
        )
        assert kc_upper.name == 'KC_Upper'
        assert np.all(kc_upper[~np.isnan(kc_upper)] >= kc_middle[~np.isnan(kc_middle)])
        assert np.all(kc_lower[~np.isnan(kc_lower)] <= kc_middle[~np.isnan(kc_middle)])
        
        # Donchian Channels
        dc_upper, dc_middle, dc_lower = indicators.donchian_channels(
            df['High'], df['Low']
        )
        assert dc_upper.name == 'DC_Upper'
        
        # Standard Deviation
        std_result = indicators.standard_deviation(df['Close'], period=30)
        assert std_result.name == 'STD_30'
        assert np.all(std_result[~np.isnan(std_result)] >= 0)
    
    def test_volume_indicators(self, sample_data):
        """Test all volume indicators."""
        df = sample_data
        
        # OBV
        obv_result = indicators.obv(df['Close'], df['Volume'])
        assert obv_result.name == 'OBV'
        
        # VWAP
        vwap_result = indicators.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        assert vwap_result.name == 'VWAP'
        
        # AD
        ad_result = indicators.ad(df['High'], df['Low'], df['Close'], df['Volume'])
        assert ad_result.name == 'AD'
        
        # MFI
        mfi_result = indicators.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        assert mfi_result.name == 'MFI_14'
        assert np.all((mfi_result[~np.isnan(mfi_result)] >= 0) & 
                     (mfi_result[~np.isnan(mfi_result)] <= 100))
        
        # CMF
        cmf_result = indicators.cmf(df['High'], df['Low'], df['Close'], df['Volume'])
        assert cmf_result.name == 'CMF_20'
        assert np.all((cmf_result[~np.isnan(cmf_result)] >= -1) & 
                     (cmf_result[~np.isnan(cmf_result)] <= 1))
    
    def test_add_all_indicators(self, sample_data):
        """Test the add_all_indicators convenience function."""
        df = sample_data.copy()
        
        # Add all indicators
        result = indicators.add_all_indicators(df)
        
        # Check that original columns are preserved
        for col in df.columns:
            assert col in result.columns
        
        # Check that indicators were added
        expected_indicators = [
            'RSI', 'ROC', 'MOM', 'STOCH_K', 'STOCH_D', 'WILLIAMS_R',
            'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ADX',
            'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'ATR',
            'OBV', 'VWAP', 'AD', 'MFI', 'CMF'
        ]
        
        for indicator in expected_indicators:
            assert indicator in result.columns
        
        # Test with prefix
        result_prefixed = indicators.add_all_indicators(df, prefix='ta_')
        for indicator in expected_indicators:
            assert f'ta_{indicator}' in result_prefixed.columns
    
    def test_custom_column_names(self, sample_data):
        """Test with custom column names."""
        # Create DataFrame with custom column names
        df = sample_data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        result = indicators.add_all_indicators(
            df,
            high_col='high',
            low_col='low',
            close_col='close',
            volume_col='volume'
        )
        
        # Check that indicators were added
        assert 'RSI' in result.columns
        assert 'MACD' in result.columns
        assert 'BB_UPPER' in result.columns
    
    def test_consistency_with_numpy(self, sample_data):
        """Test that results are consistent between pandas and numpy inputs."""
        close_series = sample_data['Close']
        close_array = close_series.to_numpy()
        
        # Test RSI
        rsi_series = indicators.rsi(close_series)
        rsi_array = indicators.rsi(close_array)
        np.testing.assert_array_equal(rsi_series.values, rsi_array)
        
        # Test SMA
        sma_series = indicators.sma(close_series, period=20)
        sma_array = indicators.sma(close_array, period=20)
        np.testing.assert_array_equal(sma_series.values, sma_array)
        
        # Test MACD
        macd_s, signal_s, hist_s = indicators.macd(close_series)
        macd_a, signal_a, hist_a = indicators.macd(close_array)
        np.testing.assert_array_equal(macd_s.values, macd_a)
        np.testing.assert_array_equal(signal_s.values, signal_a)
        np.testing.assert_array_equal(hist_s.values, hist_a)