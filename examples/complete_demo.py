"""Complete demonstration of PyIndicators' new features.

This example showcases all the enhanced functionality that makes
PyIndicators 100x easier to use than traditional libraries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create realistic sample stock data."""
    np.random.seed(42)
    
    # Generate 2 years of daily data
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='B')
    n = len(dates)
    
    # Random walk with trend and volatility clustering
    returns = np.random.normal(0.0005, 0.015, n)
    returns[100:150] *= 2  # Volatility cluster
    returns[300:320] += 0.01  # Bull run
    returns[500:550] -= 0.008  # Bear market
    
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    daily_range = np.random.uniform(0.005, 0.025, n)
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_price = low + np.random.uniform(0.3, 0.7, n) * (high - low)
    
    # Volume with realistic patterns
    base_volume = 1_000_000
    volume_volatility = np.abs(returns) * 5
    volume = base_volume * (1 + volume_volatility + np.random.normal(0, 0.2, n))
    volume = np.maximum(volume, 100_000).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    return df


def demo_1_one_line_analysis():
    """Demonstrate the one-line analysis feature."""
    print("🎯 DEMO 1: One-Line Analysis")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    data.to_csv('sample_data.csv')
    
    # One-line analysis!
    from pyindicators.easy import analyze
    
    print("Running: analyze('sample_data.csv')")
    print("This automatically:")
    print("✓ Detects column names")
    print("✓ Adds common indicators")
    print("✓ Generates trading signals")
    print("✓ Runs backtest")
    print("✓ Creates visualization")
    
    # Note: In a real demo, this would show plots
    results = analyze(data, plot=False)  # plot=False for this demo
    
    print("\n📊 Analysis completed automatically!")
    print(f"Data shape: {results.df.shape}")
    print(f"Indicators added: {len([col for col in results.df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])}")
    
    return data


def demo_2_smart_dataframe():
    """Demonstrate SmartDataFrame with auto-detection."""
    print("\n🧠 DEMO 2: Smart DataFrame Auto-Detection")
    print("=" * 50)
    
    from pyindicators.easy import SmartDataFrame
    
    # Test with different column naming conventions
    data = create_sample_data()
    
    # Rename columns to test auto-detection
    test_data = data.rename(columns={
        'Open': 'open_price',
        'High': 'high_price', 
        'Low': 'low_price',
        'Close': 'close_price',
        'Volume': 'vol'
    })
    
    print("Testing auto-detection with columns:", list(test_data.columns))
    
    sdf = SmartDataFrame(test_data)
    print(f"✓ Auto-detected columns: {sdf.columns}")
    
    # Add indicators using short names
    print("\nAdding indicators with short names...")
    sdf.add_indicators('rsi', 'macd', 'bb', 'momentum')
    
    # Show summary
    summary = sdf.summary()
    print(f"\n📈 Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return sdf


def demo_3_pipelines():
    """Demonstrate indicator pipelines."""
    print("\n🔗 DEMO 3: Indicator Pipelines")
    print("=" * 50)
    
    from pyindicators.pipeline import IndicatorPipeline
    
    data = create_sample_data()
    
    # Create a complex pipeline
    print("Building pipeline with chained operations...")
    
    result = (IndicatorPipeline(data)
              .rsi(period=14)
              .bollinger_bands(period=20, std_dev=2)
              .macd()
              .add_signal('oversold', lambda df: df['RSI_14'] < 30)
              .add_signal('overbought', lambda df: df['RSI_14'] > 70)
              .add_signal('bb_squeeze', lambda df: (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] < 0.1)
              .golden_cross(fast=20, slow=50)
              .divergence(indicator_col='RSI_14')
              .dropna()
              .get())
    
    print(f"✓ Pipeline executed successfully!")
    print(f"  Final shape: {result.shape}")
    print(f"  Signals generated:")
    
    signal_cols = [col for col in result.columns if 'signal' in col.lower() or 'cross' in col.lower() or 'divergence' in col.lower()]
    for col in signal_cols:
        count = result[col].sum() if col in result.columns else 0
        print(f"    {col}: {count}")
    
    return result


def demo_4_streaming():
    """Demonstrate streaming indicators."""
    print("\n⚡ DEMO 4: Streaming Indicators")
    print("=" * 50)
    
    from pyindicators.streaming import LiveTrader
    
    # Create live trader
    trader = LiveTrader()
    
    # Simulate live data feed
    data = create_sample_data().tail(50)  # Last 50 days
    
    print("Simulating live trading...")
    signals_generated = 0
    
    for i, (timestamp, row) in enumerate(data.iterrows()):
        tick_data = {
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume']
        }
        
        result = trader.on_tick(tick_data)
        
        if result['action']:
            signals_generated += 1
            print(f"  📈 {result['action']} signal at ${tick_data['close']:.2f}")
        
        # Show progress every 10 ticks
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1} ticks...")
    
    # Get final performance
    performance = trader.get_performance()
    print(f"\n📊 Streaming Results:")
    print(f"  Signals generated: {signals_generated}")
    for key, value in performance.items():
        print(f"  {key}: {value}")


def demo_5_multi_strategy_comparison():
    """Compare multiple trading strategies."""
    print("\n🎯 DEMO 5: Multi-Strategy Comparison")
    print("=" * 50)
    
    from pyindicators.easy import SmartDataFrame
    
    data = create_sample_data()
    sdf = SmartDataFrame(data)
    
    # Test different strategies
    strategies = ['simple', 'macd_cross', 'bb_bounce', 'trend_follow']
    results = {}
    
    print("Backtesting strategies...")
    
    for strategy in strategies:
        # Add required indicators
        if strategy == 'simple':
            sdf.add_indicators('rsi')
        elif strategy == 'macd_cross':
            sdf.add_indicators('macd')
        elif strategy == 'bb_bounce':
            sdf.add_indicators('bb')
        elif strategy == 'trend_follow':
            sdf.add_indicators('rsi', 'macd', 'ema')
        
        # Run backtest
        performance = sdf.backtest(strategy=strategy)
        results[strategy] = performance
        
        print(f"  ✓ {strategy}: {performance['total_return']}")
    
    # Find best strategy
    returns = {k: float(v['total_return'].strip('%')) for k, v in results.items()}
    best_strategy = max(returns, key=returns.get)
    
    print(f"\n🏆 Best performing strategy: {best_strategy}")
    print(f"   Return: {results[best_strategy]['total_return']}")
    print(f"   Sharpe: {results[best_strategy]['sharpe_ratio']}")
    print(f"   Max DD: {results[best_strategy]['max_drawdown']}")


def demo_6_signal_detection():
    """Demonstrate automatic signal detection."""
    print("\n🔍 DEMO 6: Automatic Signal Detection")
    print("=" * 50)
    
    from pyindicators.easy import find_signals
    
    data = create_sample_data()
    
    print("Scanning for trading signals...")
    signals = find_signals(data)
    
    if not signals.empty:
        print(f"✓ Found {len(signals)} signals")
        
        # Group by signal type
        signal_counts = signals['signal'].value_counts()
        print("\nSignal breakdown:")
        for signal_type, count in signal_counts.head().items():
            print(f"  {signal_type}: {count}")
        
        # Show recent signals
        print(f"\nMost recent signals:")
        recent = signals.head(3)
        for _, signal in recent.iterrows():
            print(f"  {signal['date'].strftime('%Y-%m-%d')}: {signal['signal']} ({signal['action']}) at ${signal['value']:.2f}")
    else:
        print("No signals found in the data.")


def demo_7_error_handling():
    """Demonstrate smart error handling."""
    print("\n🛡️ DEMO 7: Smart Error Handling")
    print("=" * 50)
    
    from pyindicators.easy import SmartDataFrame
    from pyindicators.errors import DebugMode, validate_and_clean
    
    # Create problematic data
    bad_data = pd.DataFrame({
        'close_price': [100, 101, np.nan, 103, 104],  # Has NaN
        'volume_data': [1000, 1100, 1200, np.nan, 1300],  # Different NaN
        'extra_col': ['a', 'b', 'c', 'd', 'e']  # Non-numeric
    })
    
    print("Testing with problematic data...")
    print(f"Data shape: {bad_data.shape}")
    print(f"Columns: {list(bad_data.columns)}")
    print(f"NaN values: {bad_data.isna().sum().sum()}")
    
    try:
        # This will show helpful warnings and auto-fix
        cleaned_data = validate_and_clean(bad_data, fix_issues=True)
        print(f"✓ Data cleaned successfully!")
        print(f"  New shape: {cleaned_data.shape}")
        print(f"  NaN values: {cleaned_data.isna().sum().sum()}")
        
        # Try to use with SmartDataFrame
        sdf = SmartDataFrame(cleaned_data)
        print(f"✓ SmartDataFrame created with auto-detected columns")
        
    except Exception as e:
        print(f"Error caught: {e}")


def demo_8_performance_comparison():
    """Compare performance across different data sizes."""
    print("\n🏎️ DEMO 8: Performance Demonstration")
    print("=" * 50)
    
    import time
    from pyindicators import rsi
    from pyindicators import pandas_wrapper as ta
    
    # Test different sizes
    sizes = [1_000, 10_000, 100_000]
    
    print("Performance comparison (NumPy vs Pandas wrapper):")
    print("Size       NumPy      Pandas     Overhead")
    print("-" * 45)
    
    for size in sizes:
        # Generate test data
        data = np.random.randn(size) + 100
        series = pd.Series(data)
        
        # Time NumPy version
        start = time.time()
        result1 = rsi(data, period=14)
        numpy_time = time.time() - start
        
        # Time pandas wrapper
        start = time.time()
        result2 = ta.rsi(series, period=14)
        pandas_time = time.time() - start
        
        overhead = (pandas_time - numpy_time) / numpy_time * 100
        
        print(f"{size:,}      {numpy_time:.4f}s    {pandas_time:.4f}s    {overhead:+.1f}%")
    
    print("\n✓ Performance remains excellent even with pandas convenience!")


def main():
    """Run all demonstrations."""
    print("🚀 PyIndicators Complete Feature Demonstration")
    print("=" * 60)
    print("Showcasing features that make PyIndicators 100x easier to use!")
    print()
    
    try:
        # Run all demos
        data = demo_1_one_line_analysis()
        sdf = demo_2_smart_dataframe()
        pipeline_result = demo_3_pipelines()
        demo_4_streaming()
        demo_5_multi_strategy_comparison()
        demo_6_signal_detection()
        demo_7_error_handling()
        demo_8_performance_comparison()
        
        print("\n" + "=" * 60)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Key takeaways:")
        print("✅ One-line analysis: analyze('data.csv')")
        print("✅ Auto-detection: Works with any column names")
        print("✅ Multiple interfaces: Easy, Pandas, Pipelines, Streaming")
        print("✅ Built-in strategies: Ready-to-use backtesting")
        print("✅ Smart errors: Helpful messages and auto-fixes")
        print("✅ High performance: Numba optimization maintained")
        print()
        print("🚀 PyIndicators: 100x easier than TA-Lib!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Note: Some features require additional dependencies.")
        print("Install with: pip install pyindicators[all]")


if __name__ == "__main__":
    main()