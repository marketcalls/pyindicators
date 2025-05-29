"""Basic usage examples for PyIndicators."""

import numpy as np
import matplotlib.pyplot as plt
from pyindicators import (
    rsi, macd, bollinger_bands, sma, ema,
    stochastic, atr, obv, vwap
)


def generate_sample_data(n=1000):
    """Generate realistic OHLCV data."""
    # Random walk for price
    returns = np.random.normal(0.0002, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    daily_range = np.random.uniform(0.005, 0.02, n)
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_price = low + np.random.uniform(0.3, 0.7, n) * (high - low)
    
    # Volume with some correlation to price movement
    base_volume = 1000000
    volume = base_volume + np.abs(returns) * base_volume * 50
    volume = volume.astype(int)
    
    return open_price, high, low, close, volume


def example_momentum_indicators():
    """Example usage of momentum indicators."""
    print("=== Momentum Indicators Example ===")
    
    _, _, _, close, _ = generate_sample_data()
    
    # Calculate RSI
    rsi_values = rsi(close, period=14)
    print(f"RSI (latest 5 values): {rsi_values[-5:]}")
    
    # Identify overbought/oversold conditions
    overbought = rsi_values > 70
    oversold = rsi_values < 30
    print(f"Overbought periods: {np.sum(overbought[~np.isnan(rsi_values)])} days")
    print(f"Oversold periods: {np.sum(oversold[~np.isnan(rsi_values)])} days")


def example_trend_indicators():
    """Example usage of trend indicators."""
    print("\n=== Trend Indicators Example ===")
    
    _, _, _, close, _ = generate_sample_data()
    
    # Moving averages
    sma_20 = sma(close, period=20)
    ema_20 = ema(close, period=20)
    
    # MACD
    macd_line, signal_line, histogram = macd(close)
    
    # Find MACD crossovers
    bullish_cross = np.zeros(len(close), dtype=bool)
    bearish_cross = np.zeros(len(close), dtype=bool)
    
    for i in range(1, len(close)):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
                bullish_cross[i] = True
            elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
                bearish_cross[i] = True
    
    print(f"Bullish MACD crossovers: {np.sum(bullish_cross)}")
    print(f"Bearish MACD crossovers: {np.sum(bearish_cross)}")


def example_volatility_indicators():
    """Example usage of volatility indicators."""
    print("\n=== Volatility Indicators Example ===")
    
    _, high, low, close, _ = generate_sample_data()
    
    # Bollinger Bands
    upper, middle, lower = bollinger_bands(close, period=20, std_dev=2)
    
    # Calculate %B
    bandwidth = (upper - lower) / middle * 100
    percent_b = (close - lower) / (upper - lower)
    
    print(f"Average Bollinger Bandwidth: {np.nanmean(bandwidth):.2f}%")
    print(f"Current %B: {percent_b[-1]:.2f}")
    
    # ATR
    atr_values = atr(high, low, close, period=14)
    print(f"Average True Range (latest): {atr_values[-1]:.2f}")


def example_volume_indicators():
    """Example usage of volume indicators."""
    print("\n=== Volume Indicators Example ===")
    
    _, high, low, close, volume = generate_sample_data()
    
    # OBV
    obv_values = obv(close, volume)
    obv_change = (obv_values[-1] - obv_values[-20]) / obv_values[-20] * 100
    print(f"OBV 20-day change: {obv_change:.2f}%")
    
    # VWAP
    vwap_values = vwap(high, low, close, volume)
    price_vs_vwap = (close[-1] / vwap_values[-1] - 1) * 100
    print(f"Price vs VWAP: {price_vs_vwap:+.2f}%")


def example_combined_analysis():
    """Example of combining multiple indicators."""
    print("\n=== Combined Analysis Example ===")
    
    _, high, low, close, volume = generate_sample_data(500)
    
    # Calculate multiple indicators
    rsi_vals = rsi(close, period=14)
    macd_line, signal, _ = macd(close)
    upper_bb, middle_bb, lower_bb = bollinger_bands(close, period=20)
    atr_vals = atr(high, low, close, period=14)
    
    # Simple trading signals
    signals = np.zeros(len(close))
    
    for i in range(50, len(close)):
        if np.isnan(rsi_vals[i]) or np.isnan(macd_line[i]):
            continue
            
        # Bullish signal: RSI oversold + MACD bullish + price near lower BB
        if (rsi_vals[i] < 30 and 
            macd_line[i] > signal[i] and 
            close[i] < middle_bb[i]):
            signals[i] = 1
            
        # Bearish signal: RSI overbought + MACD bearish + price near upper BB
        elif (rsi_vals[i] > 70 and 
              macd_line[i] < signal[i] and 
              close[i] > middle_bb[i]):
            signals[i] = -1
    
    buy_signals = np.sum(signals == 1)
    sell_signals = np.sum(signals == -1)
    
    print(f"Buy signals generated: {buy_signals}")
    print(f"Sell signals generated: {sell_signals}")


def plot_example():
    """Create a visualization example."""
    print("\n=== Plotting Example ===")
    
    _, high, low, close, volume = generate_sample_data(200)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Price and Bollinger Bands
    ax1 = axes[0]
    upper, middle, lower = bollinger_bands(close, period=20)
    ax1.plot(close, label='Close', color='black', linewidth=1)
    ax1.plot(upper, label='Upper BB', color='red', alpha=0.7)
    ax1.plot(middle, label='SMA(20)', color='blue', alpha=0.7)
    ax1.plot(lower, label='Lower BB', color='red', alpha=0.7)
    ax1.fill_between(range(len(close)), upper, lower, alpha=0.1, color='gray')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # RSI
    ax2 = axes[1]
    rsi_vals = rsi(close, period=14)
    ax2.plot(rsi_vals, label='RSI(14)', color='purple')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # MACD
    ax3 = axes[2]
    macd_line, signal_line, histogram = macd(close)
    ax3.plot(macd_line, label='MACD', color='blue')
    ax3.plot(signal_line, label='Signal', color='red')
    ax3.bar(range(len(histogram)), histogram, label='Histogram', color='gray', alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_ylabel('MACD')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Volume
    ax4 = axes[3]
    ax4.bar(range(len(volume)), volume, color='steelblue', alpha=0.7)
    ax4.set_ylabel('Volume')
    ax4.set_xlabel('Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/indicators_plot.png', dpi=150)
    print("Plot saved to examples/indicators_plot.png")


if __name__ == "__main__":
    # Run all examples
    example_momentum_indicators()
    example_trend_indicators()
    example_volatility_indicators()
    example_volume_indicators()
    example_combined_analysis()
    
    # Create plot if matplotlib is available
    try:
        plot_example()
    except ImportError:
        print("\nMatplotlib not installed. Skipping plot example.")