"""Example usage of PyIndicators with pandas DataFrames and Series.

This example demonstrates how to use PyIndicators with pandas for a more
user-friendly experience, similar to TA-Lib's interface.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyindicators import pandas_wrapper as ta


def load_sample_data():
    """Load or generate sample stock data."""
    # In practice, you would load real data from a CSV or API
    # For this example, we'll generate realistic sample data
    
    n = 252  # One year of daily data
    dates = pd.date_range('2023-01-01', periods=n, freq='B')  # Business days
    
    # Generate realistic price movements
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, n)  # Daily returns
    price = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    daily_range = np.random.uniform(0.005, 0.025, n)
    high = price * (1 + daily_range / 2)
    low = price * (1 - daily_range / 2)
    open_price = low + np.random.uniform(0.3, 0.7, n) * (high - low)
    
    # Volume with trend and volatility correlation
    base_volume = 10_000_000
    volume = base_volume + np.abs(returns) * base_volume * 100
    volume = volume.astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': price,
        'Volume': volume
    }, index=dates)
    
    return df


def example_basic_usage():
    """Basic usage example with individual indicators."""
    print("=== Basic Usage Example ===\n")
    
    # Load data
    df = load_sample_data()
    print(f"Loaded {len(df)} days of data")
    print(df.head())
    print()
    
    # Calculate individual indicators - returns pandas Series with proper names
    df['RSI'] = ta.rsi(df['Close'], period=14)
    df['SMA_20'] = ta.sma(df['Close'], period=20)
    df['EMA_20'] = ta.ema(df['Close'], period=20)
    
    # Multiple output indicators
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.bollinger_bands(df['Close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ta.macd(df['Close'])
    
    # Volume indicators
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    
    print("Indicators calculated. Latest values:")
    print(df[['Close', 'RSI', 'SMA_20', 'MACD', 'OBV']].tail())
    
    return df


def example_all_indicators():
    """Example using the convenience function to add all indicators."""
    print("\n=== Add All Indicators Example ===\n")
    
    # Load data
    df = load_sample_data()
    
    # Add all indicators at once
    df_with_indicators = ta.add_all_indicators(df)
    
    print(f"Original columns: {list(df.columns)}")
    print(f"\nTotal columns after adding indicators: {len(df_with_indicators.columns)}")
    print(f"New indicator columns: {[col for col in df_with_indicators.columns if col not in df.columns][:10]}...")
    
    # You can also add indicators with a prefix
    df_prefixed = ta.add_all_indicators(df, prefix='TA_')
    print(f"\nWith prefix: {[col for col in df_prefixed.columns if col.startswith('TA_')][:5]}...")
    
    return df_with_indicators


def example_trading_signals():
    """Example of creating trading signals using multiple indicators."""
    print("\n=== Trading Signals Example ===\n")
    
    # Load data
    df = load_sample_data()
    
    # Calculate indicators
    df['RSI'] = ta.rsi(df['Close'])
    df['MACD'], df['MACD_Signal'], _ = ta.macd(df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.bollinger_bands(df['Close'])
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Create trading signals
    df['Signal'] = 0
    
    # Buy signal: RSI oversold, MACD bullish crossover, price near lower BB
    buy_condition = (
        (df['RSI'] < 30) & 
        (df['MACD'] > df['MACD_Signal']) & 
        (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)) &  # Crossover
        (df['Close'] < df['BB_Middle']) &
        (df['ADX'] > 25)  # Trending market
    )
    df.loc[buy_condition, 'Signal'] = 1
    
    # Sell signal: RSI overbought, MACD bearish crossover, price near upper BB
    sell_condition = (
        (df['RSI'] > 70) & 
        (df['MACD'] < df['MACD_Signal']) & 
        (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)) &  # Crossover
        (df['Close'] > df['BB_Middle']) &
        (df['MFI'] > 80)  # Money flow overbought
    )
    df.loc[sell_condition, 'Signal'] = -1
    
    # Summary
    buy_signals = (df['Signal'] == 1).sum()
    sell_signals = (df['Signal'] == -1).sum()
    
    print(f"Buy signals generated: {buy_signals}")
    print(f"Sell signals generated: {sell_signals}")
    print("\nLast 5 signals:")
    print(df[df['Signal'] != 0][['Close', 'RSI', 'MACD', 'Signal']].tail())
    
    return df


def example_custom_analysis():
    """Example of custom analysis using pandas functionality."""
    print("\n=== Custom Analysis Example ===\n")
    
    # Load data
    df = load_sample_data()
    
    # Calculate indicators
    df['RSI'] = ta.rsi(df['Close'])
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Volatility-adjusted returns
    df['Returns'] = df['Close'].pct_change()
    df['Vol_Adj_Returns'] = df['Returns'] / df['ATR'] * df['Close']
    
    # RSI divergence detection
    df['Price_High'] = df['Close'].rolling(14).max()
    df['RSI_High'] = df['RSI'].rolling(14).max()
    
    # Identify potential divergences
    df['Bearish_Divergence'] = (
        (df['Close'] == df['Price_High']) &  # New price high
        (df['RSI'] < df['RSI_High'].shift(1))  # But RSI is lower
    )
    
    # Market regime classification
    df['Volatility_Regime'] = pd.cut(
        df['ATR'] / df['Close'] * 100,  # ATR as % of price
        bins=[0, 1, 2, 100],
        labels=['Low Vol', 'Normal Vol', 'High Vol']
    )
    
    # Money flow analysis
    df['Money_Flow_Regime'] = pd.cut(
        df['CMF'],
        bins=[-1, -0.1, 0.1, 1],
        labels=['Strong Outflow', 'Neutral', 'Strong Inflow']
    )
    
    print("Volatility Regime Distribution:")
    print(df['Volatility_Regime'].value_counts())
    print("\nMoney Flow Regime Distribution:")
    print(df['Money_Flow_Regime'].value_counts())
    print(f"\nBearish Divergences detected: {df['Bearish_Divergence'].sum()}")
    
    return df


def example_visualization():
    """Example of visualizing indicators with matplotlib."""
    print("\n=== Visualization Example ===\n")
    
    # Load data
    df = load_sample_data()
    
    # Calculate indicators
    df['SMA_20'] = ta.sma(df['Close'], 20)
    df['SMA_50'] = ta.sma(df['Close'], 50)
    df['RSI'] = ta.rsi(df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.bollinger_bands(df['Close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ta.macd(df['Close'])
    
    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Price chart with moving averages and Bollinger Bands
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='Close', color='black', linewidth=1.5)
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', color='blue', alpha=0.7)
    ax1.plot(df.index, df['SMA_50'], label='SMA 50', color='red', alpha=0.7)
    ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.2, color='gray', label='BB')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Stock Price with Technical Indicators')
    
    # Volume
    ax2 = axes[1]
    colors = ['g' if c >= o else 'r' for c, o in zip(df['Close'], df['Open'])]
    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7)
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    # RSI
    ax3 = axes[2]
    ax3.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=1.5)
    ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
    ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    ax3.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # MACD
    ax4 = axes[3]
    ax4.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=1.5)
    ax4.plot(df.index, df['MACD_Signal'], label='Signal', color='red', linewidth=1.5)
    ax4.bar(df.index, df['MACD_Hist'], label='Histogram', color='gray', alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_ylabel('MACD')
    ax4.set_xlabel('Date')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/pandas_indicators_plot.png', dpi=150, bbox_inches='tight')
    print("Chart saved to examples/pandas_indicators_plot.png")
    
    return df


def main():
    """Run all examples."""
    print("PyIndicators Pandas Examples")
    print("=" * 50)
    
    # Run examples
    df1 = example_basic_usage()
    df2 = example_all_indicators()
    df3 = example_trading_signals()
    df4 = example_custom_analysis()
    
    # Create visualization
    try:
        df5 = example_visualization()
    except ImportError:
        print("\nMatplotlib not installed. Skipping visualization example.")
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    
    # Performance note
    print("\nPerformance Note:")
    print("While pandas provides a convenient interface, the underlying")
    print("calculations still use Numba-optimized functions for speed.")
    print("For maximum performance with large datasets, consider using")
    print("the NumPy interface directly.")


if __name__ == "__main__":
    main()