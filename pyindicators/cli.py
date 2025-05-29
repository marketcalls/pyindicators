#!/usr/bin/env python
"""Command-line interface for PyIndicators.

Provides quick analysis and indicator calculations from the terminal.
"""

import argparse
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from tabulate import tabulate

from pyindicators.easy import analyze, SmartDataFrame, find_signals
from pyindicators.pipeline import IndicatorPipeline
from pyindicators import pandas_wrapper as ta


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from various file formats."""
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Detect file format
    suffix = path.suffix.lower()
    
    if suffix == '.csv':
        # Try to detect common date columns
        df = pd.read_csv(filepath)
        date_columns = ['date', 'Date', 'datetime', 'Datetime', 'timestamp', 'time']
        
        for col in date_columns:
            if col in df.columns:
                df = pd.read_csv(filepath, index_col=col, parse_dates=True)
                break
        else:
            # Try first column as date
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            except:
                df = pd.read_csv(filepath)
                
    elif suffix == '.json':
        df = pd.read_json(filepath)
    elif suffix == '.parquet':
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    return df


def calculate_indicator(args):
    """Calculate a specific indicator."""
    df = load_data(args.file)
    sdf = SmartDataFrame(df)
    
    # Get the close price column
    close = df[sdf.columns['close']]
    
    # Calculate indicator
    indicator = args.indicator.lower()
    
    if indicator == 'rsi':
        result = ta.rsi(close, args.period or 14)
        name = f'RSI_{args.period or 14}'
    elif indicator == 'sma':
        result = ta.sma(close, args.period or 20)
        name = f'SMA_{args.period or 20}'
    elif indicator == 'ema':
        result = ta.ema(close, args.period or 20)
        name = f'EMA_{args.period or 20}'
    elif indicator == 'macd':
        macd_line, signal, hist = ta.macd(close)
        result = pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal,
            'Histogram': hist
        })
        name = 'MACD'
    elif indicator == 'bb' or indicator == 'bollinger':
        upper, middle, lower = ta.bollinger_bands(close, args.period or 20, args.std or 2.0)
        result = pd.DataFrame({
            'BB_Upper': upper,
            'BB_Middle': middle,
            'BB_Lower': lower
        })
        name = 'Bollinger_Bands'
    else:
        print(f"Unknown indicator: {args.indicator}")
        sys.exit(1)
    
    # Output format
    if args.output == 'json':
        if isinstance(result, pd.DataFrame):
            print(result.to_json(orient='records', date_format='iso'))
        else:
            print(pd.DataFrame({name: result}).to_json(orient='records', date_format='iso'))
    elif args.output == 'csv':
        if isinstance(result, pd.DataFrame):
            print(result.to_csv())
        else:
            print(pd.DataFrame({name: result}).to_csv())
    else:  # table
        if isinstance(result, pd.DataFrame):
            # Show last N rows
            display_df = result.tail(args.rows)
            print(f"\n{name} (last {args.rows} values):")
            print(tabulate(display_df, headers='keys', tablefmt='grid'))
        else:
            # Show last N values
            display_data = [(i, v) for i, v in result.tail(args.rows).items()]
            print(f"\n{name} (last {args.rows} values):")
            print(tabulate(display_data, headers=['Date', name], tablefmt='grid'))
    
    # Show current value
    if not args.quiet:
        if isinstance(result, pd.DataFrame):
            print(f"\nCurrent values:")
            for col in result.columns:
                print(f"  {col}: {result[col].iloc[-1]:.4f}")
        else:
            print(f"\nCurrent {name}: {result.iloc[-1]:.4f}")


def analyze_data(args):
    """Run full analysis on data."""
    # Load and analyze
    sdf = analyze(args.file, 
                 indicators=args.indicators.split(',') if args.indicators else 'common',
                 strategy=args.strategy,
                 plot=False)  # No plotting in CLI
    
    # Additional info
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    # Current indicator values
    print("\nCurrent Indicator Values:")
    indicators_to_show = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'ATR_14', 'OBV']
    
    for ind in indicators_to_show:
        if ind in sdf.df.columns:
            value = sdf.df[ind].iloc[-1]
            if not pd.isna(value):
                print(f"  {ind}: {value:.4f}")
    
    # Recent signals
    if args.signals:
        print("\nRecent Trading Signals:")
        signals_df = find_signals(sdf.df)
        if not signals_df.empty:
            print(tabulate(signals_df.head(10), headers='keys', tablefmt='grid'))
        else:
            print("  No signals found")
    
    # Save results if requested
    if args.save:
        output_file = Path(args.file).stem + '_analysis.csv'
        sdf.df.to_csv(output_file)
        print(f"\nResults saved to: {output_file}")


def backtest_strategy(args):
    """Run backtest on a strategy."""
    df = load_data(args.file)
    sdf = SmartDataFrame(df)
    
    # Add required indicators
    if args.strategy == 'simple':
        sdf.add_indicators('rsi')
    elif args.strategy == 'macd_cross':
        sdf.add_indicators('macd')
    elif args.strategy == 'bb_bounce':
        sdf.add_indicators('bb')
    elif args.strategy == 'trend_follow':
        sdf.add_indicators('rsi', 'macd', 'ema')
    
    # Run backtest
    results = sdf.backtest(strategy=args.strategy, 
                          initial_capital=args.capital,
                          commission=args.commission)
    
    print("\n" + "="*60)
    print(f"BACKTEST RESULTS - {args.strategy.upper()} STRATEGY")
    print("="*60)
    
    # Display results
    for key, value in results.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Trade analysis
    if args.trades:
        print("\n" + "="*60)
        print("TRADE ANALYSIS")
        print("="*60)
        
        signals = sdf.signals(args.strategy)
        trades = signals[signals != 0]
        
        if len(trades) > 0:
            print(f"\nTotal trades: {len(trades)}")
            print(f"Buy signals: {(trades == 1).sum()}")
            print(f"Sell signals: {(trades == -1).sum()}")
            
            # Show recent trades
            print("\nRecent trades:")
            recent_trades = []
            for date, signal in trades.tail(10).items():
                recent_trades.append({
                    'Date': date,
                    'Signal': 'BUY' if signal == 1 else 'SELL',
                    'Price': sdf.df.loc[date, sdf.columns['close']]
                })
            
            print(tabulate(recent_trades, headers='keys', tablefmt='grid'))


def find_patterns(args):
    """Find chart patterns and signals."""
    df = load_data(args.file)
    signals_df = find_signals(df)
    
    print("\n" + "="*60)
    print("SIGNAL DETECTION RESULTS")
    print("="*60)
    
    if signals_df.empty:
        print("\nNo signals found in the data.")
        return
    
    # Group by signal type
    signal_counts = signals_df['signal'].value_counts()
    
    print("\nSignal Summary:")
    for signal_type, count in signal_counts.items():
        print(f"  {signal_type}: {count}")
    
    # Recent signals
    print(f"\nMost Recent Signals (up to {args.limit}):")
    print(tabulate(signals_df.head(args.limit), headers='keys', tablefmt='grid'))
    
    # Filter by action if specified
    if args.action:
        filtered = signals_df[signals_df['action'] == args.action.upper()]
        if not filtered.empty:
            print(f"\n{args.action.upper()} Signals Only:")
            print(tabulate(filtered.head(args.limit), headers='keys', tablefmt='grid'))


def compare_indicators(args):
    """Compare multiple indicators."""
    df = load_data(args.file)
    pipeline = IndicatorPipeline(df)
    
    # Add all requested indicators
    indicators = args.indicators.split(',')
    for ind in indicators:
        ind = ind.strip().lower()
        if ind == 'rsi':
            pipeline = pipeline.rsi()
        elif ind == 'sma':
            pipeline = pipeline.sma()
        elif ind == 'ema':
            pipeline = pipeline.ema()
        elif ind == 'macd':
            pipeline = pipeline.macd()
        elif ind == 'bb':
            pipeline = pipeline.bollinger_bands()
    
    # Get correlations
    result_df = pipeline.get()
    correlations = pipeline.correlations()
    
    print("\n" + "="*60)
    print("INDICATOR COMPARISON")
    print("="*60)
    
    # Show correlations
    print("\nIndicator Correlations:")
    # Filter to show only indicator columns
    indicator_cols = [col for col in correlations.columns 
                     if any(ind in col.lower() for ind in ['rsi', 'sma', 'ema', 'macd', 'bb'])]
    
    if indicator_cols:
        corr_subset = correlations.loc[indicator_cols, indicator_cols]
        print(tabulate(corr_subset, headers='keys', tablefmt='grid', floatfmt='.3f'))
    
    # Summary statistics
    print("\nIndicator Statistics:")
    stats = result_df[indicator_cols].describe()
    print(tabulate(stats, headers='keys', tablefmt='grid', floatfmt='.2f'))


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='PyIndicators CLI - Technical Analysis Made Easy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick analysis
  pyindicators analyze data.csv
  
  # Calculate specific indicator
  pyindicators calc data.csv -i rsi -p 21
  
  # Run backtest
  pyindicators backtest data.csv -s macd_cross -c 10000
  
  # Find trading signals
  pyindicators signals data.csv -l 20
  
  # Compare indicators
  pyindicators compare data.csv -i rsi,sma,ema
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Full analysis with indicators and backtest')
    analyze_parser.add_argument('file', help='Data file (CSV, JSON, Parquet)')
    analyze_parser.add_argument('-i', '--indicators', help='Indicators to use (comma-separated)')
    analyze_parser.add_argument('-s', '--strategy', default='simple', 
                              choices=['simple', 'macd_cross', 'bb_bounce', 'trend_follow'])
    analyze_parser.add_argument('--signals', action='store_true', help='Show trading signals')
    analyze_parser.add_argument('--save', action='store_true', help='Save results to file')
    
    # Calculate command
    calc_parser = subparsers.add_parser('calc', help='Calculate specific indicator')
    calc_parser.add_argument('file', help='Data file')
    calc_parser.add_argument('-i', '--indicator', required=True, 
                           help='Indicator name (rsi, sma, ema, macd, bb)')
    calc_parser.add_argument('-p', '--period', type=int, help='Period for indicator')
    calc_parser.add_argument('--std', type=float, help='Std dev for Bollinger Bands')
    calc_parser.add_argument('-o', '--output', choices=['table', 'json', 'csv'], 
                           default='table', help='Output format')
    calc_parser.add_argument('-r', '--rows', type=int, default=20, help='Number of rows to show')
    calc_parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest trading strategy')
    backtest_parser.add_argument('file', help='Data file')
    backtest_parser.add_argument('-s', '--strategy', default='simple',
                               choices=['simple', 'macd_cross', 'bb_bounce', 'trend_follow'])
    backtest_parser.add_argument('-c', '--capital', type=float, default=10000,
                               help='Initial capital')
    backtest_parser.add_argument('--commission', type=float, default=0.001,
                               help='Commission rate')
    backtest_parser.add_argument('--trades', action='store_true', 
                               help='Show trade details')
    
    # Signals command
    signals_parser = subparsers.add_parser('signals', help='Find trading signals')
    signals_parser.add_argument('file', help='Data file')
    signals_parser.add_argument('-l', '--limit', type=int, default=20,
                              help='Number of signals to show')
    signals_parser.add_argument('-a', '--action', choices=['buy', 'sell'],
                              help='Filter by action type')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple indicators')
    compare_parser.add_argument('file', help='Data file')
    compare_parser.add_argument('-i', '--indicators', required=True,
                              help='Indicators to compare (comma-separated)')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'analyze':
            analyze_data(args)
        elif args.command == 'calc':
            calculate_indicator(args)
        elif args.command == 'backtest':
            backtest_strategy(args)
        elif args.command == 'signals':
            find_patterns(args)
        elif args.command == 'compare':
            compare_indicators(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()