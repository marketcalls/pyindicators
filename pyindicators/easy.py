"""Ultra-easy interface for PyIndicators - making technical analysis effortless.

This module provides the simplest possible interface for using indicators,
with intelligent defaults and auto-detection of data formats.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional, Tuple, Any
import re
from pyindicators import pandas_wrapper as ta


class SmartDataFrame:
    """Intelligent wrapper for DataFrames with auto-detection and easy indicator access."""
    
    def __init__(self, data: Union[pd.DataFrame, str, dict]):
        """Initialize with data from various sources.
        
        Parameters
        ----------
        data : DataFrame, str (filepath), or dict
            The price data to analyze
        """
        self.df = self._load_data(data)
        self.columns = self._detect_columns()
        self._validate_data()
        
    def _load_data(self, data) -> pd.DataFrame:
        """Load data from various sources."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, str):
            # Auto-detect file format
            if data.endswith('.csv'):
                return pd.read_csv(data, index_col=0, parse_dates=True)
            elif data.endswith('.parquet'):
                return pd.read_parquet(data)
            elif data.endswith('.json'):
                return pd.read_json(data)
            else:
                raise ValueError(f"Unsupported file format: {data}")
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        else:
            raise ValueError("Data must be DataFrame, filepath, or dict")
    
    def _detect_columns(self) -> Dict[str, str]:
        """Auto-detect OHLCV column names."""
        columns = {}
        df_columns = [col.lower() for col in self.df.columns]
        
        # Patterns for each column type
        patterns = {
            'open': r'(open|o$|opening)',
            'high': r'(high|h$|hi)',
            'low': r'(low|l$|lo)',
            'close': r'(close|c$|closing|price$|last)',
            'volume': r'(volume|vol|v$|quantity)',
        }
        
        # Try to match patterns
        for col_type, pattern in patterns.items():
            for i, col in enumerate(df_columns):
                if re.search(pattern, col):
                    columns[col_type] = self.df.columns[i]
                    break
        
        # If close not found but we have a single price column
        if 'close' not in columns:
            price_cols = [col for col in self.df.columns if 'price' in col.lower()]
            if len(price_cols) == 1:
                columns['close'] = price_cols[0]
            elif len(self.df.columns) == 1:
                columns['close'] = self.df.columns[0]
        
        return columns
    
    def _validate_data(self):
        """Validate that we have minimum required data."""
        if 'close' not in self.columns:
            raise ValueError("Could not detect close price column. Please specify manually.")
    
    def add_indicators(self, *indicators: str, **kwargs) -> 'SmartDataFrame':
        """Add multiple indicators with smart defaults.
        
        Examples
        --------
        >>> df.add_indicators('rsi', 'macd', 'bb')  # Short names work!
        >>> df.add_indicators('all')  # Add all indicators
        >>> df.add_indicators('momentum', 'trend')  # Add categories
        """
        # Indicator aliases for easier use
        aliases = {
            'bb': 'bollinger_bands',
            'stoch': 'stochastic',
            'williams': 'williams_r',
            'mom': 'momentum',
            'std': 'standard_deviation',
            'adl': 'ad',
        }
        
        # Categories
        categories = {
            'momentum': ['rsi', 'stochastic', 'williams_r', 'roc', 'momentum'],
            'trend': ['sma', 'ema', 'macd', 'adx'],
            'volatility': ['bollinger_bands', 'atr', 'standard_deviation'],
            'volume': ['obv', 'vwap', 'ad', 'mfi', 'cmf'],
        }
        
        if 'all' in indicators:
            self.df = ta.add_all_indicators(
                self.df,
                high_col=self.columns.get('high', 'High'),
                low_col=self.columns.get('low', 'Low'),
                close_col=self.columns['close'],
                volume_col=self.columns.get('volume', 'Volume')
            )
            return self
        
        for indicator in indicators:
            indicator = indicator.lower()
            
            # Check if it's a category
            if indicator in categories:
                self.add_indicators(*categories[indicator], **kwargs)
                continue
            
            # Resolve aliases
            indicator = aliases.get(indicator, indicator)
            
            # Add the indicator
            self._add_single_indicator(indicator, **kwargs)
        
        return self
    
    def _add_single_indicator(self, indicator: str, **kwargs):
        """Add a single indicator to the dataframe."""
        close = self.df[self.columns['close']]
        
        # Get OHLC columns if available
        high = self.df.get(self.columns.get('high'))
        low = self.df.get(self.columns.get('low'))
        volume = self.df.get(self.columns.get('volume'))
        
        # Map indicator names to functions and required columns
        if indicator == 'rsi':
            self.df['RSI'] = ta.rsi(close, **kwargs)
        elif indicator == 'sma':
            period = kwargs.get('period', 20)
            self.df[f'SMA_{period}'] = ta.sma(close, period=period)
        elif indicator == 'ema':
            period = kwargs.get('period', 20)
            self.df[f'EMA_{period}'] = ta.ema(close, period=period)
        elif indicator == 'macd':
            macd, signal, hist = ta.macd(close, **kwargs)
            self.df['MACD'] = macd
            self.df['MACD_Signal'] = signal
            self.df['MACD_Hist'] = hist
        elif indicator == 'bollinger_bands':
            upper, middle, lower = ta.bollinger_bands(close, **kwargs)
            self.df['BB_Upper'] = upper
            self.df['BB_Middle'] = middle
            self.df['BB_Lower'] = lower
        elif indicator == 'stochastic' and high is not None and low is not None:
            k, d = ta.stochastic(high, low, close, **kwargs)
            self.df['Stoch_K'] = k
            self.df['Stoch_D'] = d
        elif indicator == 'atr' and high is not None and low is not None:
            self.df['ATR'] = ta.atr(high, low, close, **kwargs)
        elif indicator == 'obv' and volume is not None:
            self.df['OBV'] = ta.obv(close, volume)
        elif indicator == 'vwap' and all(x is not None for x in [high, low, volume]):
            self.df['VWAP'] = ta.vwap(high, low, close, volume)
        # Add more indicators as needed
    
    def signals(self, strategy: str = 'simple') -> pd.Series:
        """Generate trading signals based on indicators.
        
        Strategies
        ----------
        simple : RSI oversold/overbought
        macd_cross : MACD crossover
        bb_bounce : Bollinger Band bounce
        trend_follow : Trend following with multiple indicators
        """
        signals = pd.Series(0, index=self.df.index)
        
        if strategy == 'simple':
            if 'RSI' not in self.df.columns:
                self.add_indicators('rsi')
            signals[self.df['RSI'] < 30] = 1  # Buy
            signals[self.df['RSI'] > 70] = -1  # Sell
            
        elif strategy == 'macd_cross':
            if 'MACD' not in self.df.columns:
                self.add_indicators('macd')
            macd_cross_up = (self.df['MACD'] > self.df['MACD_Signal']) & \
                           (self.df['MACD'].shift(1) <= self.df['MACD_Signal'].shift(1))
            macd_cross_down = (self.df['MACD'] < self.df['MACD_Signal']) & \
                             (self.df['MACD'].shift(1) >= self.df['MACD_Signal'].shift(1))
            signals[macd_cross_up] = 1
            signals[macd_cross_down] = -1
            
        elif strategy == 'bb_bounce':
            if 'BB_Lower' not in self.df.columns:
                self.add_indicators('bb')
            close = self.df[self.columns['close']]
            signals[close <= self.df['BB_Lower']] = 1
            signals[close >= self.df['BB_Upper']] = -1
            
        elif strategy == 'trend_follow':
            # Ensure we have required indicators
            if 'RSI' not in self.df.columns:
                self.add_indicators('rsi', 'macd', 'ema')
            close = self.df[self.columns['close']]
            ema = self.df.get('EMA_20', ta.ema(close, 20))
            
            buy = (
                (self.df['RSI'] > 50) &
                (self.df['MACD'] > self.df['MACD_Signal']) &
                (close > ema)
            )
            sell = (
                (self.df['RSI'] < 50) &
                (self.df['MACD'] < self.df['MACD_Signal']) &
                (close < ema)
            )
            signals[buy] = 1
            signals[sell] = -1
        
        return signals
    
    def backtest(self, signals: Optional[pd.Series] = None, 
                 strategy: str = 'simple',
                 initial_capital: float = 10000,
                 commission: float = 0.001) -> Dict[str, Any]:
        """Simple backtesting of signals.
        
        Returns
        -------
        dict with performance metrics
        """
        if signals is None:
            signals = self.signals(strategy)
        
        close = self.df[self.columns['close']]
        
        # Calculate returns
        returns = close.pct_change()
        
        # Position sizing (simple: all-in/all-out)
        positions = signals.fillna(0)
        
        # Calculate strategy returns
        strategy_returns = positions.shift(1) * returns
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        cumulative_buy_hold = (1 + returns).cumprod()
        
        # Calculate metrics
        total_return = cumulative_returns.iloc[-1] - 1
        buy_hold_return = cumulative_buy_hold.iloc[-1] - 1
        
        # Sharpe ratio (annualized)
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        
        # Max drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = strategy_returns[strategy_returns > 0].count()
        total_trades = strategy_returns[strategy_returns != 0].count()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': f"{total_return:.2%}",
            'buy_hold_return': f"{buy_hold_return:.2%}",
            'sharpe_ratio': f"{sharpe:.2f}",
            'max_drawdown': f"{max_drawdown:.2%}",
            'win_rate': f"{win_rate:.2%}",
            'total_trades': total_trades,
            'final_value': f"${initial_capital * (1 + total_return):,.2f}"
        }
    
    def plot(self, indicators: List[str] = None, show_signals: bool = True):
        """Plot price and indicators with signals."""
        import matplotlib.pyplot as plt
        
        if indicators is None:
            indicators = ['close', 'volume', 'rsi', 'macd']
        
        # Ensure indicators are calculated
        if 'rsi' in indicators and 'RSI' not in self.df.columns:
            self.add_indicators('rsi')
        if 'macd' in indicators and 'MACD' not in self.df.columns:
            self.add_indicators('macd')
        
        # Create subplots
        n_plots = len([i for i in indicators if i != 'volume'])
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Price plot
        if 'close' in indicators:
            ax = axes[plot_idx]
            close = self.df[self.columns['close']]
            ax.plot(self.df.index, close, label='Close', color='black', linewidth=1.5)
            
            # Add indicators on price chart
            if 'BB_Upper' in self.df.columns:
                ax.plot(self.df.index, self.df['BB_Upper'], 'r--', alpha=0.5)
                ax.plot(self.df.index, self.df['BB_Lower'], 'r--', alpha=0.5)
                ax.fill_between(self.df.index, self.df['BB_Upper'], self.df['BB_Lower'], 
                               alpha=0.1, color='gray')
            
            # Add signals
            if show_signals:
                signals = self.signals()
                buy_signals = signals == 1
                sell_signals = signals == -1
                ax.scatter(self.df.index[buy_signals], close[buy_signals], 
                          color='green', marker='^', s=100, label='Buy')
                ax.scatter(self.df.index[sell_signals], close[sell_signals], 
                          color='red', marker='v', s=100, label='Sell')
            
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # RSI plot
        if 'rsi' in indicators and 'RSI' in self.df.columns:
            ax = axes[plot_idx]
            ax.plot(self.df.index, self.df['RSI'], label='RSI', color='purple')
            ax.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax.fill_between(self.df.index, 30, 70, alpha=0.1, color='gray')
            ax.set_ylabel('RSI')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # MACD plot
        if 'macd' in indicators and 'MACD' in self.df.columns:
            ax = axes[plot_idx]
            ax.plot(self.df.index, self.df['MACD'], label='MACD', color='blue')
            ax.plot(self.df.index, self.df['MACD_Signal'], label='Signal', color='red')
            ax.bar(self.df.index, self.df['MACD_Hist'], label='Histogram', 
                   color='gray', alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_ylabel('MACD')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the data and calculated indicators."""
        close = self.df[self.columns['close']]
        
        summary = {
            'date_range': f"{self.df.index[0]} to {self.df.index[-1]}",
            'total_days': len(self.df),
            'columns_detected': self.columns,
            'indicators_calculated': [col for col in self.df.columns 
                                    if col not in self.columns.values()],
            'current_price': f"${close.iloc[-1]:.2f}",
            'price_change': f"{(close.iloc[-1]/close.iloc[0] - 1):.2%}",
        }
        
        # Add current indicator values
        if 'RSI' in self.df.columns:
            summary['current_rsi'] = f"{self.df['RSI'].iloc[-1]:.1f}"
        
        return summary


def analyze(data: Union[pd.DataFrame, str, dict], 
            indicators: Union[str, List[str]] = 'common',
            strategy: str = 'simple',
            plot: bool = True) -> SmartDataFrame:
    """One-line analysis function.
    
    Examples
    --------
    >>> # Analyze a CSV file with common indicators
    >>> results = analyze('AAPL.csv')
    
    >>> # Analyze with specific indicators and strategy
    >>> results = analyze(df, indicators=['rsi', 'macd'], strategy='macd_cross')
    
    >>> # Get backtest results
    >>> perf = results.backtest()
    """
    # Create smart dataframe
    sdf = SmartDataFrame(data)
    
    # Add indicators
    if indicators == 'common':
        indicators = ['rsi', 'macd', 'bb', 'volume']
    elif indicators == 'all':
        indicators = ['all']
    elif isinstance(indicators, str):
        indicators = [indicators]
    
    sdf.add_indicators(*indicators)
    
    # Generate signals
    signals = sdf.signals(strategy)
    
    # Plot if requested
    if plot:
        sdf.plot()
    
    # Print summary
    print("\n=== Analysis Summary ===")
    for key, value in sdf.summary().items():
        print(f"{key}: {value}")
    
    # Run backtest
    backtest_results = sdf.backtest(strategy=strategy)
    print("\n=== Backtest Results ===")
    for key, value in backtest_results.items():
        print(f"{key}: {value}")
    
    return sdf


# Convenience functions for ultra-simple usage
def quick_rsi(data: Union[pd.Series, np.ndarray, str], period: int = 14) -> pd.Series:
    """Calculate RSI with minimal code.
    
    Examples
    --------
    >>> rsi_values = quick_rsi('AAPL.csv')
    >>> rsi_values = quick_rsi(df['Close'])
    """
    if isinstance(data, str):
        df = pd.read_csv(data, index_col=0, parse_dates=True)
        data = df.iloc[:, -1]  # Assume last column is close
    
    return ta.rsi(data, period)


def find_signals(data: Union[pd.DataFrame, str], 
                 min_volume: float = None) -> pd.DataFrame:
    """Find all trading signals in the data.
    
    Returns DataFrame with dates and signal types.
    """
    sdf = SmartDataFrame(data)
    sdf.add_indicators('all')
    
    signals = []
    
    # RSI signals
    if 'RSI' in sdf.df.columns:
        rsi_oversold = sdf.df['RSI'] < 30
        rsi_overbought = sdf.df['RSI'] > 70
        
        for date in sdf.df.index[rsi_oversold]:
            signals.append({
                'date': date,
                'signal': 'RSI Oversold',
                'value': sdf.df.loc[date, 'RSI'],
                'action': 'BUY'
            })
        
        for date in sdf.df.index[rsi_overbought]:
            signals.append({
                'date': date,
                'signal': 'RSI Overbought',
                'value': sdf.df.loc[date, 'RSI'],
                'action': 'SELL'
            })
    
    # MACD crossovers
    if 'MACD' in sdf.df.columns:
        macd_cross_up = (sdf.df['MACD'] > sdf.df['MACD_Signal']) & \
                       (sdf.df['MACD'].shift(1) <= sdf.df['MACD_Signal'].shift(1))
        macd_cross_down = (sdf.df['MACD'] < sdf.df['MACD_Signal']) & \
                         (sdf.df['MACD'].shift(1) >= sdf.df['MACD_Signal'].shift(1))
        
        for date in sdf.df.index[macd_cross_up]:
            signals.append({
                'date': date,
                'signal': 'MACD Bullish Cross',
                'value': sdf.df.loc[date, 'MACD'],
                'action': 'BUY'
            })
        
        for date in sdf.df.index[macd_cross_down]:
            signals.append({
                'date': date,
                'signal': 'MACD Bearish Cross',
                'value': sdf.df.loc[date, 'MACD'],
                'action': 'SELL'
            })
    
    # Bollinger Band touches
    if 'BB_Upper' in sdf.df.columns:
        close = sdf.df[sdf.columns['close']]
        bb_lower_touch = close <= sdf.df['BB_Lower']
        bb_upper_touch = close >= sdf.df['BB_Upper']
        
        for date in sdf.df.index[bb_lower_touch]:
            signals.append({
                'date': date,
                'signal': 'BB Lower Touch',
                'value': close[date],
                'action': 'BUY'
            })
        
        for date in sdf.df.index[bb_upper_touch]:
            signals.append({
                'date': date,
                'signal': 'BB Upper Touch',
                'value': close[date],
                'action': 'SELL'
            })
    
    return pd.DataFrame(signals).sort_values('date', ascending=False)


# Multi-timeframe analysis
class MultiTimeframe:
    """Analyze multiple timeframes simultaneously."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.timeframes = {}
    
    def add_timeframe(self, name: str, resample_rule: str):
        """Add a timeframe for analysis.
        
        Examples
        --------
        >>> mtf = MultiTimeframe(df)
        >>> mtf.add_timeframe('daily', 'D')
        >>> mtf.add_timeframe('weekly', 'W')
        >>> mtf.add_timeframe('monthly', 'M')
        """
        resampled = self.data.resample(resample_rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        self.timeframes[name] = SmartDataFrame(resampled)
        return self
    
    def analyze_all(self, indicators: List[str] = None):
        """Run analysis on all timeframes."""
        if indicators is None:
            indicators = ['rsi', 'macd', 'bb']
        
        results = {}
        for name, tf in self.timeframes.items():
            tf.add_indicators(*indicators)
            results[name] = {
                'data': tf.df,
                'signals': tf.signals(),
                'summary': tf.summary()
            }
        
        return results
    
    def find_confluence(self) -> pd.DataFrame:
        """Find confluence signals across timeframes."""
        all_signals = []
        
        for name, tf in self.timeframes.items():
            signals = tf.signals()
            signal_dates = signals[signals != 0].index
            
            for date in signal_dates:
                all_signals.append({
                    'date': date,
                    'timeframe': name,
                    'signal': int(signals[date])
                })
        
        confluence_df = pd.DataFrame(all_signals)
        
        # Group by date and count agreeing signals
        confluence = confluence_df.groupby('date')['signal'].agg(['sum', 'count'])
        confluence['strength'] = confluence['sum'] / confluence['count']
        
        return confluence[confluence['count'] > 1].sort_values('strength', ascending=False)