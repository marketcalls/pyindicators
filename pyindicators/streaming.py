"""Real-time streaming indicator calculations for live trading.

This module provides efficient streaming calculations that update
indicators incrementally as new data arrives.
"""

import numpy as np
from numba import njit, typed
from collections import deque
from typing import Dict, Any, Optional, Callable, Union
import pandas as pd


class StreamingIndicator:
    """Base class for streaming indicator calculations."""
    
    def __init__(self, period: int):
        self.period = period
        self.values = deque(maxlen=period)
        self.indicator_values = deque(maxlen=1000)  # Store last 1000 values
        self.is_ready = False
        
    def update(self, value: float) -> Optional[float]:
        """Update indicator with new value and return current indicator value."""
        raise NotImplementedError
        
    def reset(self):
        """Reset the indicator state."""
        self.values.clear()
        self.indicator_values.clear()
        self.is_ready = False


class StreamingRSI(StreamingIndicator):
    """Streaming RSI calculation."""
    
    def __init__(self, period: int = 14):
        super().__init__(period)
        self.gains = deque(maxlen=period)
        self.losses = deque(maxlen=period)
        self.prev_close = None
        self.avg_gain = 0
        self.avg_loss = 0
        
    def update(self, close: float) -> Optional[float]:
        if self.prev_close is not None:
            change = close - self.prev_close
            gain = max(change, 0)
            loss = abs(min(change, 0))
            
            self.gains.append(gain)
            self.losses.append(loss)
            
            if len(self.gains) == self.period:
                if not self.is_ready:
                    # First calculation
                    self.avg_gain = sum(self.gains) / self.period
                    self.avg_loss = sum(self.losses) / self.period
                    self.is_ready = True
                else:
                    # Subsequent calculations using EMA
                    self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
                    self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period
                
                if self.avg_loss == 0:
                    rsi = 100
                else:
                    rs = self.avg_gain / self.avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                self.indicator_values.append(rsi)
                self.prev_close = close
                return rsi
        
        self.prev_close = close
        return None


class StreamingSMA(StreamingIndicator):
    """Streaming Simple Moving Average calculation."""
    
    def update(self, value: float) -> Optional[float]:
        self.values.append(value)
        
        if len(self.values) == self.period:
            self.is_ready = True
            sma = sum(self.values) / self.period
            self.indicator_values.append(sma)
            return sma
        
        return None


class StreamingEMA(StreamingIndicator):
    """Streaming Exponential Moving Average calculation."""
    
    def __init__(self, period: int = 20):
        super().__init__(period)
        self.multiplier = 2 / (period + 1)
        self.ema = None
        
    def update(self, value: float) -> Optional[float]:
        self.values.append(value)
        
        if len(self.values) == self.period and self.ema is None:
            # Initialize with SMA
            self.ema = sum(self.values) / self.period
            self.is_ready = True
            
        if self.is_ready:
            self.ema = (value - self.ema) * self.multiplier + self.ema
            self.indicator_values.append(self.ema)
            return self.ema
            
        return None


class StreamingBollingerBands(StreamingIndicator):
    """Streaming Bollinger Bands calculation."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(period)
        self.std_dev = std_dev
        
    def update(self, value: float) -> Optional[Dict[str, float]]:
        self.values.append(value)
        
        if len(self.values) == self.period:
            self.is_ready = True
            middle = sum(self.values) / self.period
            
            # Calculate standard deviation
            variance = sum((x - middle) ** 2 for x in self.values) / self.period
            std = variance ** 0.5
            
            upper = middle + (self.std_dev * std)
            lower = middle - (self.std_dev * std)
            
            result = {'upper': upper, 'middle': middle, 'lower': lower}
            self.indicator_values.append(result)
            return result
            
        return None


class StreamingMACD:
    """Streaming MACD calculation."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_ema = StreamingEMA(fast_period)
        self.slow_ema = StreamingEMA(slow_period)
        self.signal_ema = StreamingEMA(signal_period)
        self.is_ready = False
        
    def update(self, value: float) -> Optional[Dict[str, float]]:
        fast = self.fast_ema.update(value)
        slow = self.slow_ema.update(value)
        
        if fast is not None and slow is not None:
            macd_line = fast - slow
            signal_line = self.signal_ema.update(macd_line)
            
            if signal_line is not None:
                self.is_ready = True
                histogram = macd_line - signal_line
                return {
                    'macd': macd_line,
                    'signal': signal_line,
                    'histogram': histogram
                }
        
        return None
    
    def reset(self):
        self.fast_ema.reset()
        self.slow_ema.reset()
        self.signal_ema.reset()
        self.is_ready = False


class StreamingATR:
    """Streaming Average True Range calculation."""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.true_ranges = deque(maxlen=period)
        self.atr = None
        self.prev_close = None
        self.is_ready = False
        
    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self.prev_close is not None:
            # Calculate true range
            tr1 = high - low
            tr2 = abs(high - self.prev_close)
            tr3 = abs(low - self.prev_close)
            true_range = max(tr1, tr2, tr3)
            
            self.true_ranges.append(true_range)
            
            if len(self.true_ranges) == self.period:
                if self.atr is None:
                    # First ATR calculation
                    self.atr = sum(self.true_ranges) / self.period
                else:
                    # Subsequent calculations using EMA
                    self.atr = (self.atr * (self.period - 1) + true_range) / self.period
                
                self.is_ready = True
                self.prev_close = close
                return self.atr
        
        self.prev_close = close
        return None


class StreamingIndicatorSet:
    """Manage multiple streaming indicators together."""
    
    def __init__(self):
        self.indicators = {}
        self.history = []
        self.max_history = 1000
        
    def add_indicator(self, name: str, indicator: StreamingIndicator):
        """Add an indicator to the set."""
        self.indicators[name] = indicator
        return self
        
    def update(self, data: Dict[str, float]) -> Dict[str, Any]:
        """Update all indicators with new data.
        
        Parameters
        ----------
        data : dict
            Dictionary with keys: 'high', 'low', 'close', 'volume', etc.
            
        Returns
        -------
        dict
            Current values of all indicators
        """
        results = {'timestamp': pd.Timestamp.now()}
        
        # Update each indicator
        for name, indicator in self.indicators.items():
            if isinstance(indicator, StreamingRSI):
                value = indicator.update(data['close'])
                if value is not None:
                    results[name] = value
                    
            elif isinstance(indicator, (StreamingSMA, StreamingEMA)):
                value = indicator.update(data['close'])
                if value is not None:
                    results[name] = value
                    
            elif isinstance(indicator, StreamingBollingerBands):
                value = indicator.update(data['close'])
                if value is not None:
                    results[f'{name}_upper'] = value['upper']
                    results[f'{name}_middle'] = value['middle']
                    results[f'{name}_lower'] = value['lower']
                    
            elif isinstance(indicator, StreamingMACD):
                value = indicator.update(data['close'])
                if value is not None:
                    results[f'{name}_line'] = value['macd']
                    results[f'{name}_signal'] = value['signal']
                    results[f'{name}_histogram'] = value['histogram']
                    
            elif isinstance(indicator, StreamingATR):
                value = indicator.update(data['high'], data['low'], data['close'])
                if value is not None:
                    results[name] = value
        
        # Store in history
        if len(results) > 1:  # More than just timestamp
            self.history.append(results)
            if len(self.history) > self.max_history:
                self.history.pop(0)
        
        return results
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get historical data as DataFrame."""
        if not self.history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.history)
        df.set_index('timestamp', inplace=True)
        return df
    
    def check_signals(self) -> Dict[str, str]:
        """Check for trading signals based on current indicator values."""
        if not self.history or len(self.history) < 2:
            return {}
        
        current = self.history[-1]
        previous = self.history[-2]
        signals = {}
        
        # RSI signals
        if 'rsi' in current:
            if current['rsi'] < 30:
                signals['rsi'] = 'oversold'
            elif current['rsi'] > 70:
                signals['rsi'] = 'overbought'
                
        # MACD crossover
        if 'macd_line' in current and 'macd_signal' in current:
            if 'macd_line' in previous and 'macd_signal' in previous:
                if (previous['macd_line'] <= previous['macd_signal'] and 
                    current['macd_line'] > current['macd_signal']):
                    signals['macd'] = 'bullish_cross'
                elif (previous['macd_line'] >= previous['macd_signal'] and 
                      current['macd_line'] < current['macd_signal']):
                    signals['macd'] = 'bearish_cross'
        
        # Bollinger Band signals
        if 'close' in current and 'bb_upper' in current and 'bb_lower' in current:
            if current['close'] >= current['bb_upper']:
                signals['bb'] = 'upper_touch'
            elif current['close'] <= current['bb_lower']:
                signals['bb'] = 'lower_touch'
        
        return signals


class LiveTrader:
    """Simple live trading interface using streaming indicators."""
    
    def __init__(self, indicators: Optional[Dict[str, StreamingIndicator]] = None):
        self.indicator_set = StreamingIndicatorSet()
        
        # Add default indicators if none provided
        if indicators is None:
            indicators = {
                'rsi': StreamingRSI(14),
                'sma_20': StreamingSMA(20),
                'ema_20': StreamingEMA(20),
                'bb': StreamingBollingerBands(20, 2),
                'macd': StreamingMACD(),
                'atr': StreamingATR(14)
            }
        
        for name, indicator in indicators.items():
            self.indicator_set.add_indicator(name, indicator)
        
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.trades = []
        
    def on_tick(self, data: Dict[str, float]) -> Dict[str, Any]:
        """Process new tick data.
        
        Parameters
        ----------
        data : dict
            Must contain: high, low, close, volume
            
        Returns
        -------
        dict
            Contains: indicators, signals, position, action
        """
        # Update indicators
        indicators = self.indicator_set.update(data)
        
        # Check for signals
        signals = self.indicator_set.check_signals()
        
        # Determine action based on signals
        action = self._determine_action(signals, data['close'])
        
        return {
            'indicators': indicators,
            'signals': signals,
            'position': self.position,
            'action': action
        }
    
    def _determine_action(self, signals: Dict[str, str], current_price: float) -> Optional[str]:
        """Determine trading action based on signals."""
        action = None
        
        # Simple logic - can be customized
        if self.position == 0:  # No position
            if 'rsi' in signals and signals['rsi'] == 'oversold':
                if 'bb' in signals and signals['bb'] == 'lower_touch':
                    action = 'BUY'
                    self.position = 1
                    self.trades.append({
                        'type': 'BUY',
                        'price': current_price,
                        'timestamp': pd.Timestamp.now()
                    })
                    
            elif 'macd' in signals and signals['macd'] == 'bullish_cross':
                action = 'BUY'
                self.position = 1
                self.trades.append({
                    'type': 'BUY',
                    'price': current_price,
                    'timestamp': pd.Timestamp.now()
                })
                
        elif self.position == 1:  # Long position
            if 'rsi' in signals and signals['rsi'] == 'overbought':
                action = 'SELL'
                self.position = 0
                self.trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'timestamp': pd.Timestamp.now()
                })
                
            elif 'macd' in signals and signals['macd'] == 'bearish_cross':
                action = 'SELL'
                self.position = 0
                self.trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'timestamp': pd.Timestamp.now()
                })
        
        return action
    
    def get_performance(self) -> Dict[str, Any]:
        """Get trading performance metrics."""
        if len(self.trades) < 2:
            return {'total_trades': len(self.trades), 'pnl': 0}
        
        pnl = 0
        wins = 0
        losses = 0
        
        for i in range(0, len(self.trades) - 1, 2):
            if i + 1 < len(self.trades):
                buy_price = self.trades[i]['price']
                sell_price = self.trades[i + 1]['price']
                trade_pnl = sell_price - buy_price
                pnl += trade_pnl
                
                if trade_pnl > 0:
                    wins += 1
                else:
                    losses += 1
        
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': len(self.trades),
            'completed_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': f"{win_rate:.1%}",
            'total_pnl': f"${pnl:.2f}"
        }


# Example usage function
def simulate_live_trading(historical_data: pd.DataFrame, speed: int = 1) -> LiveTrader:
    """Simulate live trading on historical data.
    
    Parameters
    ----------
    historical_data : DataFrame
        Must have columns: High, Low, Close, Volume
    speed : int
        Speed of simulation (1 = real-time, higher = faster)
        
    Returns
    -------
    LiveTrader
        The trader object with results
    """
    import time
    
    trader = LiveTrader()
    
    print("Starting live trading simulation...")
    print("-" * 50)
    
    for i, (timestamp, row) in enumerate(historical_data.iterrows()):
        tick_data = {
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume']
        }
        
        result = trader.on_tick(tick_data)
        
        # Print updates for significant events
        if result['action']:
            print(f"\n[{timestamp}] {result['action']} at ${tick_data['close']:.2f}")
            if result['signals']:
                print(f"Signals: {result['signals']}")
        
        # Show periodic updates
        if i % 50 == 0 and i > 0:
            perf = trader.get_performance()
            print(f"\n--- Update at bar {i} ---")
            print(f"Position: {result['position']}")
            print(f"Performance: {perf}")
        
        # Simulate delay
        time.sleep(0.1 / speed)
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(trader.get_performance())
    
    return trader