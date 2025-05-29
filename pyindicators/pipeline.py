"""Indicator pipelines for chaining and combining indicators elegantly.

This module provides a fluent API for building complex indicator combinations
with automatic caching and optimization.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Callable, Optional, Tuple
from functools import wraps
from pyindicators import pandas_wrapper as ta


class IndicatorPipeline:
    """Fluent interface for chaining technical indicators."""
    
    def __init__(self, data: Union[pd.DataFrame, pd.Series]):
        """Initialize pipeline with data.
        
        Parameters
        ----------
        data : DataFrame or Series
            The input data to process
        """
        if isinstance(data, pd.Series):
            self.df = pd.DataFrame({'value': data})
            self._series_mode = True
            self._series_name = data.name or 'value'
        else:
            self.df = data.copy()
            self._series_mode = False
        
        self._cache = {}
        self._steps = []
        
    def __getitem__(self, key):
        """Allow direct access to calculated indicators."""
        return self.df[key]
    
    # Momentum indicators
    def rsi(self, period: int = 14, column: str = None) -> 'IndicatorPipeline':
        """Add RSI to the pipeline."""
        col = column or self._get_default_column()
        self.df[f'RSI_{period}'] = ta.rsi(self.df[col], period)
        self._steps.append(f'RSI({period})')
        return self
    
    def stochastic(self, k_period: int = 14, d_period: int = 3) -> 'IndicatorPipeline':
        """Add Stochastic Oscillator."""
        if self._has_ohlc():
            k, d = ta.stochastic(self.df['High'], self.df['Low'], self.df['Close'], 
                               k_period, d_period)
            self.df[f'Stoch_K_{k_period}'] = k
            self.df[f'Stoch_D_{d_period}'] = d
            self._steps.append(f'Stochastic({k_period},{d_period})')
        return self
    
    def momentum(self, period: int = 10, column: str = None) -> 'IndicatorPipeline':
        """Add Momentum indicator."""
        col = column or self._get_default_column()
        self.df[f'MOM_{period}'] = ta.momentum(self.df[col], period)
        self._steps.append(f'Momentum({period})')
        return self
    
    # Trend indicators
    def sma(self, period: int = 20, column: str = None) -> 'IndicatorPipeline':
        """Add Simple Moving Average."""
        col = column or self._get_default_column()
        self.df[f'SMA_{period}'] = ta.sma(self.df[col], period)
        self._steps.append(f'SMA({period})')
        return self
    
    def ema(self, period: int = 20, column: str = None) -> 'IndicatorPipeline':
        """Add Exponential Moving Average."""
        col = column or self._get_default_column()
        self.df[f'EMA_{period}'] = ta.ema(self.df[col], period)
        self._steps.append(f'EMA({period})')
        return self
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9, 
             column: str = None) -> 'IndicatorPipeline':
        """Add MACD."""
        col = column or self._get_default_column()
        macd_line, signal_line, histogram = ta.macd(self.df[col], fast, slow, signal)
        self.df['MACD'] = macd_line
        self.df['MACD_Signal'] = signal_line
        self.df['MACD_Histogram'] = histogram
        self._steps.append(f'MACD({fast},{slow},{signal})')
        return self
    
    # Volatility indicators
    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0, 
                       column: str = None) -> 'IndicatorPipeline':
        """Add Bollinger Bands."""
        col = column or self._get_default_column()
        upper, middle, lower = ta.bollinger_bands(self.df[col], period, std_dev)
        self.df['BB_Upper'] = upper
        self.df['BB_Middle'] = middle
        self.df['BB_Lower'] = lower
        self._steps.append(f'BB({period},{std_dev})')
        return self
    
    def atr(self, period: int = 14) -> 'IndicatorPipeline':
        """Add Average True Range."""
        if self._has_ohlc():
            self.df[f'ATR_{period}'] = ta.atr(self.df['High'], self.df['Low'], 
                                             self.df['Close'], period)
            self._steps.append(f'ATR({period})')
        return self
    
    # Volume indicators
    def volume_indicators(self) -> 'IndicatorPipeline':
        """Add all volume-based indicators if volume data is available."""
        if 'Volume' in self.df.columns:
            self.df['OBV'] = ta.obv(self.df['Close'], self.df['Volume'])
            if self._has_ohlc():
                self.df['VWAP'] = ta.vwap(self.df['High'], self.df['Low'], 
                                         self.df['Close'], self.df['Volume'])
                self.df['MFI'] = ta.mfi(self.df['High'], self.df['Low'], 
                                       self.df['Close'], self.df['Volume'])
            self._steps.append('Volume Indicators')
        return self
    
    # Composite indicators
    def trend_strength(self) -> 'IndicatorPipeline':
        """Add trend strength composite indicator."""
        # Ensure we have required indicators
        if 'RSI_14' not in self.df.columns:
            self.rsi(14)
        if 'ADX_14' not in self.df.columns and self._has_ohlc():
            self.df['ADX_14'] = ta.adx(self.df['High'], self.df['Low'], 
                                       self.df['Close'], 14)
        
        # Calculate trend strength
        if 'ADX_14' in self.df.columns:
            self.df['Trend_Strength'] = (self.df['RSI_14'] - 50).abs() * \
                                       self.df['ADX_14'] / 50
        else:
            self.df['Trend_Strength'] = (self.df['RSI_14'] - 50).abs()
        
        self._steps.append('Trend Strength')
        return self
    
    def volatility_adjusted_momentum(self, lookback: int = 20) -> 'IndicatorPipeline':
        """Add volatility-adjusted momentum."""
        col = self._get_default_column()
        
        # Calculate returns
        returns = self.df[col].pct_change(lookback)
        
        # Calculate volatility
        volatility = self.df[col].rolling(lookback).std() / self.df[col].rolling(lookback).mean()
        
        # Volatility-adjusted momentum
        self.df['Vol_Adj_Momentum'] = returns / volatility
        self._steps.append(f'Vol-Adj Momentum({lookback})')
        return self
    
    # Custom calculations
    def custom(self, func: Callable, name: str, *args, **kwargs) -> 'IndicatorPipeline':
        """Apply custom calculation to the pipeline.
        
        Parameters
        ----------
        func : callable
            Function that takes DataFrame and returns Series or DataFrame
        name : str
            Name for the resulting column(s)
        *args, **kwargs
            Additional arguments for the function
        """
        result = func(self.df, *args, **kwargs)
        if isinstance(result, pd.Series):
            self.df[name] = result
        elif isinstance(result, pd.DataFrame):
            self.df = pd.concat([self.df, result], axis=1)
        self._steps.append(f'Custom: {name}')
        return self
    
    def transform(self, column: str, func: Callable, new_name: str) -> 'IndicatorPipeline':
        """Transform a column using a function."""
        self.df[new_name] = func(self.df[column])
        self._steps.append(f'Transform: {column} -> {new_name}')
        return self
    
    # Filtering and conditions
    def where(self, condition: Union[pd.Series, Callable]) -> 'IndicatorPipeline':
        """Filter data based on condition."""
        if callable(condition):
            mask = condition(self.df)
        else:
            mask = condition
        
        self.df = self.df[mask]
        self._steps.append('Filter')
        return self
    
    def add_signal(self, name: str, condition: Union[pd.Series, Callable]) -> 'IndicatorPipeline':
        """Add a signal column based on condition."""
        if callable(condition):
            self.df[name] = condition(self.df).astype(int)
        else:
            self.df[name] = condition.astype(int)
        self._steps.append(f'Signal: {name}')
        return self
    
    # Utility methods
    def dropna(self) -> 'IndicatorPipeline':
        """Drop rows with NaN values."""
        self.df = self.df.dropna()
        self._steps.append('Drop NaN')
        return self
    
    def tail(self, n: int = 100) -> 'IndicatorPipeline':
        """Keep only the last n rows."""
        self.df = self.df.tail(n)
        self._steps.append(f'Tail({n})')
        return self
    
    def resample(self, rule: str, agg: Dict[str, str] = None) -> 'IndicatorPipeline':
        """Resample the data to a different timeframe."""
        if agg is None:
            agg = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
        
        # Keep only columns that exist
        agg = {k: v for k, v in agg.items() if k in self.df.columns}
        
        self.df = self.df.resample(rule).agg(agg).dropna()
        self._steps.append(f'Resample({rule})')
        return self
    
    # Signal generation
    def golden_cross(self, fast: int = 50, slow: int = 200) -> 'IndicatorPipeline':
        """Add golden/death cross signals."""
        col = self._get_default_column()
        
        # Ensure MAs exist
        if f'SMA_{fast}' not in self.df.columns:
            self.sma(fast)
        if f'SMA_{slow}' not in self.df.columns:
            self.sma(slow)
        
        # Detect crosses
        fast_ma = self.df[f'SMA_{fast}']
        slow_ma = self.df[f'SMA_{slow}']
        
        golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        self.df['Golden_Cross'] = golden_cross.astype(int)
        self.df['Death_Cross'] = death_cross.astype(int)
        
        self._steps.append(f'Golden Cross({fast},{slow})')
        return self
    
    def divergence(self, price_col: str = None, indicator_col: str = 'RSI_14', 
                   window: int = 14) -> 'IndicatorPipeline':
        """Detect divergences between price and indicator."""
        price = self.df[price_col or self._get_default_column()]
        
        # Ensure indicator exists
        if indicator_col not in self.df.columns:
            if 'RSI' in indicator_col:
                self.rsi(14)
            else:
                raise ValueError(f"Indicator {indicator_col} not found")
        
        indicator = self.df[indicator_col]
        
        # Find local highs and lows
        price_highs = price.rolling(window).max() == price
        price_lows = price.rolling(window).min() == price
        
        ind_highs = indicator.rolling(window).max() == indicator
        ind_lows = indicator.rolling(window).min() == indicator
        
        # Detect divergences
        self.df['Bullish_Divergence'] = (price_lows & (price < price.shift(window)) & 
                                         ind_lows & (indicator > indicator.shift(window))).astype(int)
        self.df['Bearish_Divergence'] = (price_highs & (price > price.shift(window)) & 
                                         ind_highs & (indicator < indicator.shift(window))).astype(int)
        
        self._steps.append(f'Divergence({indicator_col})')
        return self
    
    # Output methods
    def get(self) -> Union[pd.DataFrame, pd.Series]:
        """Get the resulting DataFrame or Series."""
        if self._series_mode and len(self.df.columns) == 1:
            return self.df.iloc[:, 0]
        return self.df
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return self.df.to_dict('series')
    
    def describe(self) -> pd.DataFrame:
        """Get statistical description of all numeric columns."""
        return self.df.describe()
    
    def correlations(self) -> pd.DataFrame:
        """Get correlation matrix of all indicators."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        return self.df[numeric_cols].corr()
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution."""
        return {
            'steps': self._steps,
            'columns': list(self.df.columns),
            'shape': self.df.shape,
            'index_range': f"{self.df.index[0]} to {self.df.index[-1]}" if len(self.df) > 0 else "empty"
        }
    
    # Helper methods
    def _get_default_column(self) -> str:
        """Get default price column."""
        if self._series_mode:
            return 'value'
        elif 'Close' in self.df.columns:
            return 'Close'
        elif 'close' in self.df.columns:
            return 'close'
        else:
            # Return first numeric column
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return numeric_cols[0]
            raise ValueError("No suitable price column found")
    
    def _has_ohlc(self) -> bool:
        """Check if OHLC data is available."""
        required = ['High', 'Low', 'Close']
        return all(col in self.df.columns for col in required)


# Factory functions for common pipelines
def momentum_pipeline(data: pd.DataFrame) -> IndicatorPipeline:
    """Create a pipeline with common momentum indicators."""
    return (IndicatorPipeline(data)
            .rsi()
            .momentum()
            .stochastic()
            .macd())


def trend_pipeline(data: pd.DataFrame) -> IndicatorPipeline:
    """Create a pipeline with common trend indicators."""
    return (IndicatorPipeline(data)
            .sma(20)
            .sma(50)
            .ema(20)
            .macd()
            .golden_cross())


def volatility_pipeline(data: pd.DataFrame) -> IndicatorPipeline:
    """Create a pipeline with volatility indicators."""
    return (IndicatorPipeline(data)
            .bollinger_bands()
            .atr()
            .volatility_adjusted_momentum())


def complete_pipeline(data: pd.DataFrame) -> IndicatorPipeline:
    """Create a pipeline with all major indicators."""
    pipeline = (IndicatorPipeline(data)
                .rsi()
                .macd()
                .bollinger_bands()
                .sma(20)
                .sma(50)
                .ema(20))
    
    if 'Volume' in data.columns:
        pipeline = pipeline.volume_indicators()
    
    if all(col in data.columns for col in ['High', 'Low', 'Close']):
        pipeline = pipeline.atr().stochastic()
    
    return pipeline


# Advanced pipeline patterns
class StrategyPipeline(IndicatorPipeline):
    """Extended pipeline with built-in trading strategies."""
    
    def mean_reversion_setup(self, bb_period: int = 20, rsi_period: int = 14) -> 'StrategyPipeline':
        """Setup indicators for mean reversion strategy."""
        return (self
                .bollinger_bands(bb_period)
                .rsi(rsi_period)
                .add_signal('MR_Buy', lambda df: (df[f'RSI_{rsi_period}'] < 30) & 
                           (df[self._get_default_column()] <= df['BB_Lower']))
                .add_signal('MR_Sell', lambda df: (df[f'RSI_{rsi_period}'] > 70) & 
                           (df[self._get_default_column()] >= df['BB_Upper'])))
    
    def trend_following_setup(self, fast: int = 20, slow: int = 50) -> 'StrategyPipeline':
        """Setup indicators for trend following strategy."""
        return (self
                .sma(fast)
                .sma(slow)
                .macd()
                .atr()
                .add_signal('TF_Buy', lambda df: (df[f'SMA_{fast}'] > df[f'SMA_{slow}']) & 
                           (df['MACD'] > df['MACD_Signal']))
                .add_signal('TF_Sell', lambda df: (df[f'SMA_{fast}'] < df[f'SMA_{slow}']) & 
                           (df['MACD'] < df['MACD_Signal'])))
    
    def breakout_setup(self, period: int = 20) -> 'StrategyPipeline':
        """Setup indicators for breakout strategy."""
        col = self._get_default_column()
        
        # Add rolling high/low
        self.df[f'High_{period}'] = self.df[col].rolling(period).max()
        self.df[f'Low_{period}'] = self.df[col].rolling(period).min()
        
        return (self
                .atr()
                .volume_indicators()
                .add_signal('Breakout_Buy', lambda df: df[col] > df[f'High_{period}'].shift(1))
                .add_signal('Breakout_Sell', lambda df: df[col] < df[f'Low_{period}'].shift(1)))


# Usage examples
def example_pipeline_usage():
    """Example of using indicator pipelines."""
    # Sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 101,
        'Low': np.random.randn(100).cumsum() + 99,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)
    
    # Simple pipeline
    result = (IndicatorPipeline(data)
              .rsi()
              .bollinger_bands()
              .macd()
              .get())
    
    # Complex pipeline with conditions
    signals = (IndicatorPipeline(data)
               .rsi()
               .bollinger_bands()
               .add_signal('Oversold', lambda df: df['RSI_14'] < 30)
               .add_signal('Overbought', lambda df: df['RSI_14'] > 70)
               .divergence()
               .dropna()
               .get())
    
    # Strategy pipeline
    strategy = (StrategyPipeline(data)
                .mean_reversion_setup()
                .trend_following_setup()
                .get())
    
    return result, signals, strategy