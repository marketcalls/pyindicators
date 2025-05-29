"""Jupyter notebook widgets for interactive indicator analysis.

This module provides IPython widgets for Jupyter notebooks to make
indicator analysis interactive and visual.
"""

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Callable
import warnings

from pyindicators.easy import SmartDataFrame, analyze
from pyindicators.pipeline import IndicatorPipeline
from pyindicators import pandas_wrapper as ta


if not WIDGETS_AVAILABLE:
    warnings.warn(
        "IPython widgets not available. Install with: pip install ipywidgets",
        ImportWarning
    )


class IndicatorExplorer:
    """Interactive widget for exploring indicators in Jupyter notebooks."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize the explorer with data.
        
        Parameters
        ----------
        data : DataFrame
            OHLCV data
        """
        if not WIDGETS_AVAILABLE:
            raise ImportError("ipywidgets required. Install with: pip install ipywidgets")
            
        self.data = data
        self.sdf = SmartDataFrame(data)
        self.current_indicators = {}
        
        # Create widgets
        self._create_widgets()
        self._setup_layout()
        
    def _create_widgets(self):
        """Create all the widgets."""
        # Indicator selection
        self.indicator_dropdown = widgets.Dropdown(
            options=['RSI', 'MACD', 'Bollinger Bands', 'SMA', 'EMA', 'ATR', 'Stochastic'],
            description='Indicator:',
            value='RSI'
        )
        
        # Period slider
        self.period_slider = widgets.IntSlider(
            value=14,
            min=5,
            max=100,
            step=1,
            description='Period:',
            continuous_update=False
        )
        
        # Additional parameters
        self.param2_slider = widgets.FloatSlider(
            value=2.0,
            min=1.0,
            max=3.0,
            step=0.5,
            description='Std Dev:',
            continuous_update=False
        )
        
        # Date range slider
        date_range = (0, len(self.data) - 1)
        self.date_slider = widgets.IntRangeSlider(
            value=[max(0, len(self.data) - 252), len(self.data) - 1],
            min=date_range[0],
            max=date_range[1],
            step=1,
            description='Date Range:',
            continuous_update=False
        )
        
        # Strategy selection
        self.strategy_dropdown = widgets.Dropdown(
            options=['None', 'Simple RSI', 'MACD Cross', 'BB Bounce'],
            description='Strategy:',
            value='None'
        )
        
        # Backtest button
        self.backtest_button = widgets.Button(
            description='Run Backtest',
            button_style='success',
            icon='check'
        )
        
        # Output areas
        self.plot_output = widgets.Output()
        self.stats_output = widgets.Output()
        self.backtest_output = widgets.Output()
        
        # Connect events
        self.indicator_dropdown.observe(self._on_indicator_change, 'value')
        self.period_slider.observe(self._update_plot, 'value')
        self.param2_slider.observe(self._update_plot, 'value')
        self.date_slider.observe(self._update_plot, 'value')
        self.strategy_dropdown.observe(self._update_plot, 'value')
        self.backtest_button.on_click(self._run_backtest)
        
    def _setup_layout(self):
        """Setup the widget layout."""
        # Control panel
        controls = widgets.VBox([
            widgets.HTML('<h3>Indicator Settings</h3>'),
            self.indicator_dropdown,
            self.period_slider,
            self.param2_slider,
            widgets.HTML('<h3>Display Settings</h3>'),
            self.date_slider,
            widgets.HTML('<h3>Strategy Testing</h3>'),
            self.strategy_dropdown,
            self.backtest_button
        ])
        
        # Main display
        display_area = widgets.VBox([
            self.plot_output,
            widgets.HBox([self.stats_output, self.backtest_output])
        ])
        
        # Full layout
        self.layout = widgets.HBox([
            controls,
            display_area
        ])
        
        # Initial plot
        self._update_plot()
        
    def _on_indicator_change(self, change):
        """Handle indicator selection change."""
        indicator = change['new']
        
        # Update parameter visibility
        if indicator == 'Bollinger Bands':
            self.param2_slider.description = 'Std Dev:'
            self.param2_slider.min = 1.0
            self.param2_slider.max = 3.0
            self.param2_slider.value = 2.0
            self.param2_slider.layout.visibility = 'visible'
        elif indicator == 'MACD':
            self.param2_slider.layout.visibility = 'hidden'
            self.period_slider.layout.visibility = 'hidden'
        elif indicator == 'Stochastic':
            self.param2_slider.description = 'D Period:'
            self.param2_slider.min = 1
            self.param2_slider.max = 10
            self.param2_slider.value = 3
            self.param2_slider.layout.visibility = 'visible'
        else:
            self.param2_slider.layout.visibility = 'hidden'
            self.period_slider.layout.visibility = 'visible'
        
        self._update_plot()
        
    def _update_plot(self, change=None):
        """Update the plot with current settings."""
        with self.plot_output:
            clear_output(wait=True)
            
            # Get date range
            start_idx, end_idx = self.date_slider.value
            plot_data = self.data.iloc[start_idx:end_idx + 1]
            
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Price plot
            ax1 = axes[0]
            close = plot_data[self.sdf.columns['close']]
            ax1.plot(plot_data.index, close, 'k-', linewidth=1.5, label='Close')
            
            # Add indicator to price chart if applicable
            indicator = self.indicator_dropdown.value
            if indicator == 'Bollinger Bands':
                upper, middle, lower = ta.bollinger_bands(
                    self.data[self.sdf.columns['close']], 
                    self.period_slider.value,
                    self.param2_slider.value
                )
                upper = upper.iloc[start_idx:end_idx + 1]
                middle = middle.iloc[start_idx:end_idx + 1]
                lower = lower.iloc[start_idx:end_idx + 1]
                
                ax1.plot(plot_data.index, upper, 'r--', alpha=0.7, label='Upper BB')
                ax1.plot(plot_data.index, middle, 'b--', alpha=0.7, label='Middle BB')
                ax1.plot(plot_data.index, lower, 'r--', alpha=0.7, label='Lower BB')
                ax1.fill_between(plot_data.index, upper, lower, alpha=0.1, color='gray')
                
            elif indicator in ['SMA', 'EMA']:
                if indicator == 'SMA':
                    ma = ta.sma(self.data[self.sdf.columns['close']], self.period_slider.value)
                else:
                    ma = ta.ema(self.data[self.sdf.columns['close']], self.period_slider.value)
                
                ma = ma.iloc[start_idx:end_idx + 1]
                ax1.plot(plot_data.index, ma, 'orange', linewidth=2, 
                        label=f'{indicator}({self.period_slider.value})')
            
            # Add strategy signals
            strategy = self.strategy_dropdown.value
            if strategy != 'None':
                signals = self._get_strategy_signals(plot_data, strategy)
                buy_signals = plot_data.index[signals == 1]
                sell_signals = plot_data.index[signals == -1]
                
                for idx in buy_signals:
                    ax1.scatter(idx, close[idx], color='green', marker='^', 
                               s=100, zorder=5, label='Buy' if idx == buy_signals[0] else '')
                for idx in sell_signals:
                    ax1.scatter(idx, close[idx], color='red', marker='v', 
                               s=100, zorder=5, label='Sell' if idx == sell_signals[0] else '')
            
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Indicator subplot
            ax2 = axes[1]
            if indicator == 'RSI':
                rsi = ta.rsi(self.data[self.sdf.columns['close']], self.period_slider.value)
                rsi = rsi.iloc[start_idx:end_idx + 1]
                ax2.plot(plot_data.index, rsi, 'purple', linewidth=1.5)
                ax2.axhline(70, color='r', linestyle='--', alpha=0.5)
                ax2.axhline(30, color='g', linestyle='--', alpha=0.5)
                ax2.fill_between(plot_data.index, 30, 70, alpha=0.1, color='gray')
                ax2.set_ylabel('RSI')
                ax2.set_ylim(0, 100)
                
            elif indicator == 'MACD':
                macd_line, signal_line, histogram = ta.macd(self.data[self.sdf.columns['close']])
                macd_line = macd_line.iloc[start_idx:end_idx + 1]
                signal_line = signal_line.iloc[start_idx:end_idx + 1]
                histogram = histogram.iloc[start_idx:end_idx + 1]
                
                ax2.plot(plot_data.index, macd_line, 'b-', label='MACD', linewidth=1.5)
                ax2.plot(plot_data.index, signal_line, 'r-', label='Signal', linewidth=1.5)
                ax2.bar(plot_data.index, histogram, alpha=0.3, color='gray', label='Histogram')
                ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
                ax2.set_ylabel('MACD')
                ax2.legend()
                
            elif indicator == 'ATR':
                if all(col in self.sdf.columns for col in ['high', 'low']):
                    atr = ta.atr(
                        self.data[self.sdf.columns['high']],
                        self.data[self.sdf.columns['low']],
                        self.data[self.sdf.columns['close']],
                        self.period_slider.value
                    )
                    atr = atr.iloc[start_idx:end_idx + 1]
                    ax2.plot(plot_data.index, atr, 'orange', linewidth=1.5)
                    ax2.set_ylabel('ATR')
                else:
                    ax2.text(0.5, 0.5, 'ATR requires High/Low data', 
                            ha='center', va='center', transform=ax2.transAxes)
                    
            elif indicator == 'Stochastic':
                if all(col in self.sdf.columns for col in ['high', 'low']):
                    k, d = ta.stochastic(
                        self.data[self.sdf.columns['high']],
                        self.data[self.sdf.columns['low']],
                        self.data[self.sdf.columns['close']],
                        self.period_slider.value,
                        int(self.param2_slider.value)
                    )
                    k = k.iloc[start_idx:end_idx + 1]
                    d = d.iloc[start_idx:end_idx + 1]
                    ax2.plot(plot_data.index, k, 'b-', label=f'%K', linewidth=1.5)
                    ax2.plot(plot_data.index, d, 'r-', label=f'%D', linewidth=1.5)
                    ax2.axhline(80, color='r', linestyle='--', alpha=0.5)
                    ax2.axhline(20, color='g', linestyle='--', alpha=0.5)
                    ax2.set_ylabel('Stochastic')
                    ax2.set_ylim(0, 100)
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'Stochastic requires High/Low data', 
                            ha='center', va='center', transform=ax2.transAxes)
            else:
                # For indicators shown on price chart
                ax2.text(0.5, 0.5, f'{indicator} shown on price chart', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_visible(False)
                
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        # Update statistics
        self._update_stats()
        
    def _update_stats(self):
        """Update statistics display."""
        with self.stats_output:
            clear_output(wait=True)
            
            indicator = self.indicator_dropdown.value
            close = self.data[self.sdf.columns['close']]
            
            stats_html = f"<h4>{indicator} Statistics</h4>"
            stats_html += "<table style='width:100%'>"
            
            if indicator == 'RSI':
                rsi = ta.rsi(close, self.period_slider.value)
                current_rsi = rsi.iloc[-1]
                avg_rsi = rsi.mean()
                
                stats_html += f"<tr><td>Current RSI:</td><td>{current_rsi:.2f}</td></tr>"
                stats_html += f"<tr><td>Average RSI:</td><td>{avg_rsi:.2f}</td></tr>"
                stats_html += f"<tr><td>Overbought days:</td><td>{(rsi > 70).sum()}</td></tr>"
                stats_html += f"<tr><td>Oversold days:</td><td>{(rsi < 30).sum()}</td></tr>"
                
            elif indicator == 'MACD':
                macd_line, signal_line, histogram = ta.macd(close)
                stats_html += f"<tr><td>Current MACD:</td><td>{macd_line.iloc[-1]:.4f}</td></tr>"
                stats_html += f"<tr><td>Current Signal:</td><td>{signal_line.iloc[-1]:.4f}</td></tr>"
                stats_html += f"<tr><td>Current Histogram:</td><td>{histogram.iloc[-1]:.4f}</td></tr>"
                
                # Count crossovers
                crosses = ((macd_line > signal_line) != (macd_line.shift(1) > signal_line.shift(1))).sum()
                stats_html += f"<tr><td>Total Crossovers:</td><td>{crosses}</td></tr>"
                
            stats_html += "</table>"
            display(HTML(stats_html))
            
    def _get_strategy_signals(self, data: pd.DataFrame, strategy: str) -> pd.Series:
        """Get trading signals for a strategy."""
        signals = pd.Series(0, index=data.index)
        
        if strategy == 'Simple RSI':
            rsi = ta.rsi(self.data[self.sdf.columns['close']], self.period_slider.value)
            rsi = rsi.reindex(data.index)
            signals[rsi < 30] = 1
            signals[rsi > 70] = -1
            
        elif strategy == 'MACD Cross':
            macd_line, signal_line, _ = ta.macd(self.data[self.sdf.columns['close']])
            macd_line = macd_line.reindex(data.index)
            signal_line = signal_line.reindex(data.index)
            
            # Detect crossovers
            macd_above = macd_line > signal_line
            signals[macd_above & ~macd_above.shift(1)] = 1
            signals[~macd_above & macd_above.shift(1)] = -1
            
        elif strategy == 'BB Bounce':
            upper, _, lower = ta.bollinger_bands(
                self.data[self.sdf.columns['close']], 
                self.period_slider.value,
                self.param2_slider.value
            )
            close = self.data[self.sdf.columns['close']].reindex(data.index)
            upper = upper.reindex(data.index)
            lower = lower.reindex(data.index)
            
            signals[close <= lower] = 1
            signals[close >= upper] = -1
            
        return signals
        
    def _run_backtest(self, button):
        """Run backtest on selected strategy."""
        with self.backtest_output:
            clear_output(wait=True)
            
            strategy = self.strategy_dropdown.value
            if strategy == 'None':
                display(HTML("<p>Please select a strategy first!</p>"))
                return
                
            # Get signals for full dataset
            signals = self._get_strategy_signals(self.data, strategy)
            
            # Simple backtest
            close = self.data[self.sdf.columns['close']]
            returns = close.pct_change()
            
            # Calculate strategy returns
            strategy_returns = signals.shift(1) * returns
            
            # Performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            
            # Max drawdown
            cumulative = (1 + strategy_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Trade statistics
            trades = signals[signals != 0]
            n_trades = len(trades)
            win_trades = strategy_returns[signals.shift(1) != 0] > 0
            win_rate = win_trades.sum() / len(win_trades) if len(win_trades) > 0 else 0
            
            # Display results
            results_html = f"<h4>Backtest Results - {strategy}</h4>"
            results_html += "<table style='width:100%'>"
            results_html += f"<tr><td>Total Return:</td><td>{total_return:.2%}</td></tr>"
            results_html += f"<tr><td>Sharpe Ratio:</td><td>{sharpe:.2f}</td></tr>"
            results_html += f"<tr><td>Max Drawdown:</td><td>{max_drawdown:.2%}</td></tr>"
            results_html += f"<tr><td>Total Trades:</td><td>{n_trades}</td></tr>"
            results_html += f"<tr><td>Win Rate:</td><td>{win_rate:.2%}</td></tr>"
            results_html += "</table>"
            
            display(HTML(results_html))
            
    def display(self):
        """Display the widget."""
        display(self.layout)


class QuickAnalyzer:
    """Simple widget for quick analysis."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize quick analyzer."""
        if not WIDGETS_AVAILABLE:
            raise ImportError("ipywidgets required. Install with: pip install ipywidgets")
            
        self.data = data
        self.sdf = SmartDataFrame(data)
        
        # Create simple interface
        self.analyze_button = widgets.Button(
            description='Analyze',
            button_style='primary',
            icon='bar-chart'
        )
        
        self.indicator_select = widgets.SelectMultiple(
            options=['RSI', 'MACD', 'Bollinger Bands', 'SMA(20)', 'SMA(50)', 'Volume'],
            value=['RSI', 'MACD'],
            description='Indicators:',
            rows=6
        )
        
        self.output = widgets.Output()
        
        self.analyze_button.on_click(self._analyze)
        
        self.layout = widgets.VBox([
            widgets.HTML('<h3>Quick Technical Analysis</h3>'),
            self.indicator_select,
            self.analyze_button,
            self.output
        ])
        
    def _analyze(self, button):
        """Run analysis."""
        with self.output:
            clear_output(wait=True)
            
            # Add selected indicators
            indicators = list(self.indicator_select.value)
            
            # Create pipeline
            pipeline = IndicatorPipeline(self.data)
            
            for indicator in indicators:
                if indicator == 'RSI':
                    pipeline = pipeline.rsi()
                elif indicator == 'MACD':
                    pipeline = pipeline.macd()
                elif indicator == 'Bollinger Bands':
                    pipeline = pipeline.bollinger_bands()
                elif indicator == 'SMA(20)':
                    pipeline = pipeline.sma(20)
                elif indicator == 'SMA(50)':
                    pipeline = pipeline.sma(50)
                    
            result = pipeline.get()
            
            # Plot
            self._plot_analysis(result, indicators)
            
            # Show summary
            self._show_summary(result)
            
    def _plot_analysis(self, data: pd.DataFrame, indicators: List[str]):
        """Plot the analysis results."""
        n_plots = 2 if any(ind in ['RSI', 'MACD'] for ind in indicators) else 1
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
            
        # Price plot
        ax1 = axes[0]
        close = data[self.sdf.columns['close']]
        ax1.plot(data.index, close, 'k-', linewidth=1.5, label='Close')
        
        # Add overlays
        if 'Bollinger Bands' in indicators:
            ax1.plot(data.index, data['BB_Upper'], 'r--', alpha=0.7)
            ax1.plot(data.index, data['BB_Lower'], 'r--', alpha=0.7)
            ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], 
                           alpha=0.1, color='gray')
                           
        if 'SMA(20)' in indicators:
            ax1.plot(data.index, data['SMA_20'], 'blue', alpha=0.7, label='SMA(20)')
            
        if 'SMA(50)' in indicators:
            ax1.plot(data.index, data['SMA_50'], 'orange', alpha=0.7, label='SMA(50)')
            
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Indicator subplot
        if n_plots > 1:
            ax2 = axes[1]
            
            if 'RSI' in indicators and 'RSI_14' in data.columns:
                ax2.plot(data.index, data['RSI_14'], 'purple', linewidth=1.5, label='RSI')
                ax2.axhline(70, color='r', linestyle='--', alpha=0.5)
                ax2.axhline(30, color='g', linestyle='--', alpha=0.5)
                ax2.set_ylabel('RSI')
                ax2.set_ylim(0, 100)
                
            elif 'MACD' in indicators and 'MACD' in data.columns:
                ax2.plot(data.index, data['MACD'], 'b-', label='MACD')
                ax2.plot(data.index, data['MACD_Signal'], 'r-', label='Signal')
                ax2.bar(data.index, data['MACD_Histogram'], alpha=0.3, color='gray')
                ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
                ax2.set_ylabel('MACD')
                
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
    def _show_summary(self, data: pd.DataFrame):
        """Show analysis summary."""
        summary_html = "<h4>Analysis Summary</h4>"
        summary_html += "<table>"
        
        # Current values
        close = data[self.sdf.columns['close']].iloc[-1]
        summary_html += f"<tr><td>Current Price:</td><td>${close:.2f}</td></tr>"
        
        if 'RSI_14' in data.columns:
            rsi = data['RSI_14'].iloc[-1]
            summary_html += f"<tr><td>RSI(14):</td><td>{rsi:.2f}"
            if rsi > 70:
                summary_html += " <span style='color:red'>(Overbought)</span>"
            elif rsi < 30:
                summary_html += " <span style='color:green'>(Oversold)</span>"
            summary_html += "</td></tr>"
            
        if 'MACD' in data.columns:
            macd = data['MACD'].iloc[-1]
            signal = data['MACD_Signal'].iloc[-1]
            summary_html += f"<tr><td>MACD:</td><td>{macd:.4f}</td></tr>"
            summary_html += f"<tr><td>Signal:</td><td>{signal:.4f}</td></tr>"
            if macd > signal:
                summary_html += "<tr><td>MACD Status:</td><td style='color:green'>Bullish</td></tr>"
            else:
                summary_html += "<tr><td>MACD Status:</td><td style='color:red'>Bearish</td></tr>"
                
        summary_html += "</table>"
        display(HTML(summary_html))
        
    def display(self):
        """Display the widget."""
        display(self.layout)


# Convenience functions for Jupyter notebooks
def interactive_analysis(data: Union[pd.DataFrame, str]) -> IndicatorExplorer:
    """Launch interactive indicator explorer.
    
    Parameters
    ----------
    data : DataFrame or filepath
        Price data to analyze
        
    Returns
    -------
    IndicatorExplorer
        The widget instance
        
    Example
    -------
    >>> explorer = interactive_analysis('AAPL.csv')
    >>> # or
    >>> explorer = interactive_analysis(df)
    """
    if isinstance(data, str):
        sdf = SmartDataFrame(data)
        data = sdf.df
        
    explorer = IndicatorExplorer(data)
    explorer.display()
    return explorer


def quick_analysis(data: Union[pd.DataFrame, str]) -> QuickAnalyzer:
    """Launch quick analysis widget.
    
    Parameters
    ----------
    data : DataFrame or filepath
        Price data to analyze
        
    Returns
    -------
    QuickAnalyzer
        The widget instance
    """
    if isinstance(data, str):
        sdf = SmartDataFrame(data)
        data = sdf.df
        
    analyzer = QuickAnalyzer(data)
    analyzer.display()
    return analyzer


# For environments without widgets, provide alternative
def notebook_analysis(data: Union[pd.DataFrame, str], 
                     indicators: List[str] = None) -> None:
    """Non-interactive analysis for notebooks without widget support.
    
    This provides static plots and analysis when widgets are not available.
    """
    if isinstance(data, str):
        sdf = SmartDataFrame(data)
        data = sdf.df
    else:
        sdf = SmartDataFrame(data)
        
    if indicators is None:
        indicators = ['rsi', 'macd', 'bb']
        
    # Add indicators
    sdf.add_indicators(*indicators)
    
    # Create plots
    sdf.plot(indicators)
    
    # Show summary
    summary = sdf.summary()
    print("\nAnalysis Summary:")
    print("-" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")
        
    # Run backtest
    backtest_results = sdf.backtest()
    print("\nBacktest Results:")
    print("-" * 50)
    for key, value in backtest_results.items():
        print(f"{key}: {value}")