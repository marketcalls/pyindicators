"""Performance benchmarks for PyIndicators."""

import numpy as np
import time
import statistics
from tabulate import tabulate

from pyindicators import (
    rsi, sma, ema, macd, bollinger_bands,
    stochastic, atr, obv, vwap, mfi
)


def time_function(func, *args, iterations=5, **kwargs):
    """Time a function execution."""
    times = []
    
    # Warm-up run (for JIT compilation)
    func(*args, **kwargs)
    
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean': statistics.mean(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times)
    }


def format_time(seconds):
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def benchmark_indicators():
    """Run benchmarks on all indicators."""
    print("PyIndicators Performance Benchmark")
    print("=" * 60)
    
    # Test different data sizes
    sizes = [1_000, 10_000, 100_000, 1_000_000]
    
    for size in sizes:
        print(f"\nData size: {size:,} points")
        print("-" * 60)
        
        # Generate test data
        np.random.seed(42)
        close = np.random.randn(size) + 100
        high = close + np.random.uniform(0, 2, size)
        low = close - np.random.uniform(0, 2, size)
        volume = np.random.randint(100_000, 1_000_000, size)
        
        results = []
        
        # Benchmark each indicator
        indicators = [
            ("RSI(14)", lambda: rsi(close, period=14)),
            ("SMA(20)", lambda: sma(close, period=20)),
            ("EMA(20)", lambda: ema(close, period=20)),
            ("MACD", lambda: macd(close)),
            ("Bollinger Bands", lambda: bollinger_bands(close)),
            ("Stochastic", lambda: stochastic(high, low, close)),
            ("ATR(14)", lambda: atr(high, low, close, period=14)),
            ("OBV", lambda: obv(close, volume)),
            ("VWAP", lambda: vwap(high, low, close, volume)),
            ("MFI(14)", lambda: mfi(high, low, close, volume, period=14)),
        ]
        
        for name, func in indicators:
            timing = time_function(func, iterations=5)
            results.append([
                name,
                format_time(timing['mean']),
                format_time(timing['std']),
                f"{size / timing['mean']:,.0f} pts/sec"
            ])
        
        # Display results
        headers = ["Indicator", "Mean Time", "Std Dev", "Throughput"]
        print(tabulate(results, headers=headers, tablefmt="grid"))


def benchmark_parallel_processing():
    """Benchmark parallel processing capabilities."""
    print("\n\nParallel Processing Benchmark")
    print("=" * 60)
    
    sizes = [10_000, 100_000, 1_000_000]
    
    for size in sizes:
        print(f"\nCalculating SMA with different periods on {size:,} points")
        
        close = np.random.randn(size) + 100
        periods = [5, 10, 20, 50, 100, 200]
        
        # Sequential calculation
        start = time.perf_counter()
        for period in periods:
            sma(close, period=period)
        sequential_time = time.perf_counter() - start
        
        print(f"Sequential calculation: {format_time(sequential_time)}")
        print(f"Average per indicator: {format_time(sequential_time / len(periods))}")


def benchmark_memory_efficiency():
    """Test memory efficiency."""
    print("\n\nMemory Efficiency Test")
    print("=" * 60)
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create large dataset
    size = 10_000_000
    close = np.random.randn(size) + 100
    
    data_memory = process.memory_info().rss / 1024 / 1024 - initial_memory
    print(f"Memory for {size:,} data points: {data_memory:.1f} MB")
    
    # Calculate indicator
    result = rsi(close, period=14)
    
    total_memory = process.memory_info().rss / 1024 / 1024 - initial_memory
    indicator_memory = total_memory - data_memory
    
    print(f"Additional memory for RSI calculation: {indicator_memory:.1f} MB")
    print(f"Memory efficiency: {size / total_memory:,.0f} points/MB")


def compare_with_pure_python():
    """Compare Numba-optimized vs pure Python implementation."""
    print("\n\nNumba vs Pure Python Comparison")
    print("=" * 60)
    
    def sma_pure_python(data, period):
        """Pure Python SMA implementation."""
        result = [np.nan] * len(data)
        for i in range(period - 1, len(data)):
            result[i] = sum(data[i - period + 1:i + 1]) / period
        return np.array(result)
    
    sizes = [1_000, 5_000, 10_000]
    
    print("Simple Moving Average (SMA) Comparison:")
    print(f"{'Size':<10} {'Pure Python':<15} {'Numba':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for size in sizes:
        close = np.random.randn(size) + 100
        
        # Pure Python
        start = time.perf_counter()
        sma_pure_python(close, 20)
        python_time = time.perf_counter() - start
        
        # Numba
        numba_timing = time_function(sma, close, period=20, iterations=3)
        numba_time = numba_timing['mean']
        
        speedup = python_time / numba_time
        
        print(f"{size:<10} {format_time(python_time):<15} "
              f"{format_time(numba_time):<15} {speedup:.1f}x")


if __name__ == "__main__":
    # Check if optional dependencies are available
    try:
        from tabulate import tabulate
    except ImportError:
        print("Please install tabulate for better formatting: pip install tabulate")
        exit(1)
    
    try:
        import psutil
    except ImportError:
        print("Note: Install psutil for memory benchmarks: pip install psutil")
        psutil = None
    
    # Run benchmarks
    benchmark_indicators()
    benchmark_parallel_processing()
    
    if psutil:
        benchmark_memory_efficiency()
    
    compare_with_pure_python()
    
    print("\n" + "=" * 60)
    print("Benchmark completed!")