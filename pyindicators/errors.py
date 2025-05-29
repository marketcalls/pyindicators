"""Smart error handling and debugging for PyIndicators.

This module provides helpful error messages and debugging utilities
to make the library more user-friendly.
"""

import sys
import traceback
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import warnings


class PyIndicatorsError(Exception):
    """Base exception for PyIndicators."""
    pass


class DataError(PyIndicatorsError):
    """Error related to input data."""
    pass


class IndicatorError(PyIndicatorsError):
    """Error in indicator calculation."""
    pass


class ColumnNotFoundError(DataError):
    """Column not found in DataFrame."""
    pass


class InsufficientDataError(DataError):
    """Not enough data for calculation."""
    pass


def smart_error_handler(func: Callable) -> Callable:
    """Decorator that provides smart error messages for common issues."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get function name and arguments
            func_name = func.__name__
            
            # Analyze the error and provide helpful message
            if isinstance(e, KeyError):
                handle_key_error(e, func_name, args, kwargs)
            elif isinstance(e, ValueError):
                handle_value_error(e, func_name, args, kwargs)
            elif isinstance(e, TypeError):
                handle_type_error(e, func_name, args, kwargs)
            elif isinstance(e, IndexError):
                handle_index_error(e, func_name, args, kwargs)
            else:
                # Re-raise with additional context
                raise PyIndicatorsError(
                    f"Error in {func_name}: {str(e)}\n"
                    f"Please check your input data and parameters."
                ) from e
    
    return wrapper


def handle_key_error(e: KeyError, func_name: str, args: tuple, kwargs: dict):
    """Handle KeyError with helpful message."""
    missing_key = str(e).strip("'")
    
    # Check if it's a column error
    if any(isinstance(arg, pd.DataFrame) for arg in args):
        df = next(arg for arg in args if isinstance(arg, pd.DataFrame))
        available_columns = list(df.columns)
        
        # Suggest similar columns
        suggestions = find_similar_columns(missing_key, available_columns)
        
        error_msg = f"\n❌ Column '{missing_key}' not found in DataFrame!\n\n"
        error_msg += f"Available columns: {available_columns}\n"
        
        if suggestions:
            error_msg += f"\nDid you mean one of these?\n"
            for suggestion in suggestions:
                error_msg += f"  • {suggestion}\n"
        
        error_msg += "\n💡 Tip: Check your column names are correct. Common issues:\n"
        error_msg += "  - Case sensitivity (Close vs close)\n"
        error_msg += "  - Extra spaces in column names\n"
        error_msg += "  - Using 'Close' when your data has 'close_price'\n"
        
        raise ColumnNotFoundError(error_msg) from e
    
    # Re-raise original error if not column-related
    raise


def handle_value_error(e: ValueError, func_name: str, args: tuple, kwargs: dict):
    """Handle ValueError with helpful message."""
    error_str = str(e).lower()
    
    # Check for common value errors
    if 'period' in error_str or 'window' in error_str:
        error_msg = f"\n❌ Invalid period/window parameter!\n\n"
        error_msg += f"Error: {str(e)}\n"
        error_msg += "\n💡 Tips:\n"
        error_msg += "  - Period must be a positive integer\n"
        error_msg += "  - Period should be less than data length\n"
        error_msg += "  - For RSI, typical period is 14\n"
        error_msg += "  - For SMA, typical periods are 20, 50, 200\n"
        
        raise IndicatorError(error_msg) from e
    
    elif 'empty' in error_str or 'no data' in error_str:
        error_msg = f"\n❌ No data to process!\n\n"
        error_msg += "Please ensure:\n"
        error_msg += "  - Your DataFrame is not empty\n"
        error_msg += "  - The date range contains data\n"
        error_msg += "  - Your file was loaded correctly\n"
        
        raise InsufficientDataError(error_msg) from e
    
    # Re-raise with context
    raise IndicatorError(f"Value error in {func_name}: {str(e)}") from e


def handle_type_error(e: TypeError, func_name: str, args: tuple, kwargs: dict):
    """Handle TypeError with helpful message."""
    error_msg = f"\n❌ Type error in {func_name}!\n\n"
    error_msg += f"Error: {str(e)}\n"
    error_msg += "\n💡 Common causes:\n"
    error_msg += "  - Passing Series when DataFrame expected (or vice versa)\n"
    error_msg += "  - Passing string when numeric value expected\n"
    error_msg += "  - Missing required parameters\n"
    
    # Check for specific parameter issues
    if 'NoneType' in str(e):
        error_msg += "  - A required parameter is None/missing\n"
    
    raise IndicatorError(error_msg) from e


def handle_index_error(e: IndexError, func_name: str, args: tuple, kwargs: dict):
    """Handle IndexError with helpful message."""
    error_msg = f"\n❌ Index error - not enough data!\n\n"
    error_msg += f"Function: {func_name}\n"
    
    # Check data length
    for arg in args:
        if isinstance(arg, (pd.Series, pd.DataFrame, np.ndarray)):
            data_len = len(arg)
            error_msg += f"Data length: {data_len}\n"
            
            # Check if period is too large
            period = kwargs.get('period', None)
            if period and period >= data_len:
                error_msg += f"Period ({period}) is >= data length ({data_len})!\n"
                error_msg += "\n💡 Solutions:\n"
                error_msg += f"  - Use more data (need at least {period + 1} points)\n"
                error_msg += f"  - Reduce the period parameter\n"
                break
    
    raise InsufficientDataError(error_msg) from e


def find_similar_columns(target: str, columns: List[str], threshold: float = 0.6) -> List[str]:
    """Find columns similar to target using fuzzy matching."""
    from difflib import SequenceMatcher
    
    suggestions = []
    target_lower = target.lower()
    
    for col in columns:
        # Check exact match (case-insensitive)
        if col.lower() == target_lower:
            suggestions.append(col)
            continue
        
        # Check similarity
        similarity = SequenceMatcher(None, target_lower, col.lower()).ratio()
        if similarity >= threshold:
            suggestions.append((similarity, col))
    
    # Sort by similarity
    suggestions = sorted(suggestions, key=lambda x: x[0] if isinstance(x, tuple) else 1, reverse=True)
    
    # Return column names only
    return [s[1] if isinstance(s, tuple) else s for s in suggestions[:3]]


class DataValidator:
    """Validate input data and provide helpful feedback."""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame, 
                      required_columns: List[str] = None) -> Dict[str, Any]:
        """Validate OHLCV data and return validation report."""
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check if DataFrame
        if not isinstance(df, pd.DataFrame):
            report['valid'] = False
            report['errors'].append("Input must be a pandas DataFrame")
            return report
        
        # Check if empty
        if df.empty:
            report['valid'] = False
            report['errors'].append("DataFrame is empty")
            return report
        
        # Check required columns
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                report['valid'] = False
                report['errors'].append(f"Missing required columns: {missing}")
                
                # Suggest similar columns
                for col in missing:
                    similar = find_similar_columns(col, list(df.columns))
                    if similar:
                        report['suggestions'].append(
                            f"For '{col}', did you mean: {similar[0]}?"
                        )
        
        # Check data types
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            report['valid'] = False
            report['errors'].append("No numeric columns found")
        
        # Check for NaN values
        nan_counts = df[numeric_columns].isna().sum()
        if nan_counts.any():
            nan_cols = nan_counts[nan_counts > 0]
            for col, count in nan_cols.items():
                pct = count / len(df) * 100
                if pct > 50:
                    report['warnings'].append(
                        f"Column '{col}' has {count} NaN values ({pct:.1f}%)"
                    )
        
        # Check index
        if not isinstance(df.index, pd.DatetimeIndex):
            report['warnings'].append(
                "Index is not DatetimeIndex. Consider using parse_dates=True when loading."
            )
        
        # Check data order
        if isinstance(df.index, pd.DatetimeIndex) and not df.index.is_monotonic_increasing:
            report['warnings'].append(
                "Data is not sorted by date. Consider df.sort_index()"
            )
        
        # Check for duplicates
        if df.index.duplicated().any():
            report['warnings'].append(
                f"Found {df.index.duplicated().sum()} duplicate index values"
            )
        
        return report


class DebugMode:
    """Context manager for debug mode with detailed logging."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.original_warnings = warnings.filters[:]
        
    def __enter__(self):
        if self.verbose:
            # Show all warnings
            warnings.filterwarnings('always')
            print("🐛 Debug mode enabled")
            print("-" * 50)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original warnings
        warnings.filters[:] = self.original_warnings
        
        if exc_type is not None and self.verbose:
            print("\n" + "="*50)
            print("❌ ERROR OCCURRED")
            print("="*50)
            print(f"Type: {exc_type.__name__}")
            print(f"Message: {exc_val}")
            print("\nTraceback:")
            traceback.print_tb(exc_tb)
            print("\n💡 Debug tips:")
            print("1. Check your input data format")
            print("2. Verify column names match exactly")
            print("3. Ensure sufficient data for calculations")
            print("4. Check parameter values are valid")
            
        if self.verbose:
            print("-" * 50)
            print("Debug mode disabled")
        
        # Don't suppress the exception
        return False


def validate_and_clean(df: pd.DataFrame, 
                      fix_issues: bool = True) -> pd.DataFrame:
    """Validate and optionally fix common data issues."""
    
    # Run validation
    validator = DataValidator()
    report = validator.validate_ohlcv(df)
    
    if not report['valid']:
        error_msg = "Data validation failed:\n"
        for error in report['errors']:
            error_msg += f"  • {error}\n"
        raise DataError(error_msg)
    
    # Show warnings
    if report['warnings']:
        print("⚠️  Data warnings:")
        for warning in report['warnings']:
            print(f"  • {warning}")
    
    if fix_issues:
        df_clean = df.copy()
        
        # Sort by index if needed
        if isinstance(df.index, pd.DatetimeIndex) and not df.index.is_monotonic_increasing:
            df_clean = df_clean.sort_index()
            print("✓ Sorted data by date")
        
        # Remove duplicates
        if df.index.duplicated().any():
            df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
            print("✓ Removed duplicate dates")
        
        # Forward fill NaN values (optional)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if df_clean[numeric_cols].isna().any().any():
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill')
            print("✓ Forward-filled NaN values")
        
        return df_clean
    
    return df


# Helpful error messages for common mistakes
ERROR_MESSAGES = {
    'no_volume': """
❌ Volume data not found!

Your data doesn't have a 'Volume' column, which is required for volume indicators.

Solutions:
1. Use indicators that don't require volume (RSI, SMA, MACD, etc.)
2. Add volume data to your DataFrame
3. Skip volume indicators: df.add_indicators('momentum', 'trend')
""",
    
    'insufficient_data': """
❌ Not enough data for calculation!

The indicator period is larger than your data length.

Solutions:
1. Use more historical data
2. Reduce the indicator period
3. Check that your data loaded correctly

Example: For RSI(14), you need at least 15 data points.
""",
    
    'wrong_input_type': """
❌ Wrong input type!

Expected: pandas Series or DataFrame
Received: {received_type}

Solutions:
1. Convert to pandas: pd.Series(your_data) or pd.DataFrame(your_data)
2. Load data properly: pd.read_csv('file.csv')
3. Check you're passing the right variable
""",
    
    'missing_ohlc': """
❌ Missing OHLC data!

This indicator requires High, Low, and Close prices.
Your data only has: {available_columns}

Solutions:
1. Use a different data source with OHLC data
2. Use indicators that only need Close prices (RSI, SMA, EMA)
3. Check your column names match exactly (case-sensitive)
"""
}


def show_helpful_error(error_key: str, **kwargs) -> None:
    """Display a helpful error message."""
    message = ERROR_MESSAGES.get(error_key, "Unknown error")
    print(message.format(**kwargs))


# Example usage in indicator functions
def safe_indicator_wrapper(indicator_func: Callable) -> Callable:
    """Wrapper that adds safety checks to indicator functions."""
    
    @wraps(indicator_func)
    def wrapper(data, *args, **kwargs):
        # Pre-validation
        if isinstance(data, pd.DataFrame):
            report = DataValidator.validate_ohlcv(data)
            if report['warnings']:
                for warning in report['warnings']:
                    warnings.warn(warning, UserWarning)
        
        # Call original function with error handling
        try:
            return indicator_func(data, *args, **kwargs)
        except Exception as e:
            # Add context to error
            func_name = indicator_func.__name__
            raise IndicatorError(
                f"Failed to calculate {func_name}: {str(e)}\n"
                f"Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}\n"
                f"Parameters: {kwargs}"
            ) from e
    
    return wrapper