"""Visual indicator builder and interactive playground.

This module provides interactive visualization tools for exploring indicators
and building trading strategies visually.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.patches import Rectangle
from typing import Dict, List, Any, Tuple, Optional, Callable
import warnings
from pyindicators import pandas_wrapper as ta
from pyindicators.easy import SmartDataFrame


class InteractiveChart:
    """Interactive chart for exploring indicators with sliders."""
    
    def __init__(self, data: pd.DataFrame, figsize: Tuple[int, int] = (14, 10)):
        """Initialize interactive chart.
        
        Parameters
        ----------
        data : DataFrame
            OHLCV data
        figsize : tuple
            Figure size (width, height)
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.figsize = figsize
        
        # Detect columns
        self.sdf = SmartDataFrame(data)
        self.columns = self.sdf.columns
        
        # Setup figure
        self.fig = plt.figure(figsize=figsize)
        self.setup_layout()
        
        # Track active indicators
        self.active_indicators = {}
        self.indicator_lines = {}
        self.signal_markers = {'buy': [], 'sell': []}
        
        # Initial plot
        self.plot_price()
        self.setup_controls()
        
    def setup_layout(self):
        """Setup the figure layout with subplots."""
        # Main price chart
        self.ax_price = plt.subplot2grid((6, 4), (0, 0), rowspan=3, colspan=3)
        
        # Indicator charts
        self.ax_momentum = plt.subplot2grid((6, 4), (3, 0), colspan=3, sharex=self.ax_price)
        self.ax_volume = plt.subplot2grid((6, 4), (4, 0), colspan=3, sharex=self.ax_price)
        
        # Control panel
        self.ax_controls = plt.subplot2grid((6, 4), (0, 3), rowspan=6)
        self.ax_controls.axis('off')
        
        # Info panel
        self.ax_info = plt.subplot2grid((6, 4), (5, 0), colspan=3)
        self.ax_info.axis('off')
        
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
    def plot_price(self):
        """Plot the main price chart."""
        close = self.data[self.columns['close']]
        
        # Candlestick-style plot
        self.ax_price.clear()
        self.ax_price.plot(self.data.index, close, 'k-', linewidth=1, label='Close')
        
        # Add volume bars
        if 'volume' in self.columns:
            volume = self.data[self.columns.get('volume', 'Volume')]
            self.ax_volume.clear()
            self.ax_volume.bar(self.data.index, volume, alpha=0.3, color='steelblue')
            self.ax_volume.set_ylabel('Volume')
            self.ax_volume.grid(True, alpha=0.3)
        
        self.ax_price.set_ylabel('Price')
        self.ax_price.grid(True, alpha=0.3)
        self.ax_price.legend(loc='upper left')
        
    def setup_controls(self):
        """Setup interactive controls."""
        # Control positions
        y_start = 0.95
        y_step = 0.06
        
        # Title
        self.ax_controls.text(0.5, y_start, 'Indicator Controls', 
                             ha='center', fontsize=14, weight='bold')
        
        # RSI controls
        y_pos = y_start - y_step * 2
        self.ax_controls.text(0.1, y_pos, 'RSI:', fontsize=10)
        ax_rsi_toggle = plt.axes([0.8, y_pos - 0.01, 0.15, 0.04])
        self.btn_rsi = widgets.Button(ax_rsi_toggle, 'OFF', color='lightgray')
        self.btn_rsi.on_clicked(lambda x: self.toggle_indicator('RSI'))
        
        ax_rsi_period = plt.axes([0.8, y_pos - 0.05, 0.15, 0.03])
        self.slider_rsi = widgets.Slider(ax_rsi_period, 'Period', 5, 50, valinit=14, valstep=1)
        self.slider_rsi.on_changed(lambda x: self.update_indicator('RSI'))
        
        # MACD controls
        y_pos -= y_step * 2
        self.ax_controls.text(0.1, y_pos, 'MACD:', fontsize=10)
        ax_macd_toggle = plt.axes([0.8, y_pos - 0.01, 0.15, 0.04])
        self.btn_macd = widgets.Button(ax_macd_toggle, 'OFF', color='lightgray')
        self.btn_macd.on_clicked(lambda x: self.toggle_indicator('MACD'))
        
        # Bollinger Bands controls
        y_pos -= y_step * 2
        self.ax_controls.text(0.1, y_pos, 'BB:', fontsize=10)
        ax_bb_toggle = plt.axes([0.8, y_pos - 0.01, 0.15, 0.04])
        self.btn_bb = widgets.Button(ax_bb_toggle, 'OFF', color='lightgray')
        self.btn_bb.on_clicked(lambda x: self.toggle_indicator('BB'))
        
        ax_bb_period = plt.axes([0.8, y_pos - 0.05, 0.15, 0.03])
        self.slider_bb_period = widgets.Slider(ax_bb_period, 'Period', 10, 50, valinit=20, valstep=1)
        ax_bb_std = plt.axes([0.8, y_pos - 0.09, 0.15, 0.03])
        self.slider_bb_std = widgets.Slider(ax_bb_std, 'Std', 1, 3, valinit=2, valstep=0.5)
        self.slider_bb_period.on_changed(lambda x: self.update_indicator('BB'))
        self.slider_bb_std.on_changed(lambda x: self.update_indicator('BB'))
        
        # Moving Averages
        y_pos -= y_step * 2.5
        self.ax_controls.text(0.1, y_pos, 'SMA:', fontsize=10)
        ax_sma_toggle = plt.axes([0.8, y_pos - 0.01, 0.15, 0.04])
        self.btn_sma = widgets.Button(ax_sma_toggle, 'OFF', color='lightgray')
        self.btn_sma.on_clicked(lambda x: self.toggle_indicator('SMA'))
        
        ax_sma_period = plt.axes([0.8, y_pos - 0.05, 0.15, 0.03])
        self.slider_sma = widgets.Slider(ax_sma_period, 'Period', 5, 200, valinit=20, valstep=5)
        self.slider_sma.on_changed(lambda x: self.update_indicator('SMA'))
        
        # Strategy selector
        y_pos -= y_step * 2
        self.ax_controls.text(0.1, y_pos, 'Strategy:', fontsize=10)
        ax_strategy = plt.axes([0.75, y_pos - 0.05, 0.2, 0.1])
        self.radio_strategy = widgets.RadioButtons(
            ax_strategy, 
            ('None', 'RSI', 'MACD', 'BB'),
            active=0
        )
        self.radio_strategy.on_clicked(self.update_strategy)
        
        # Backtest button
        y_pos -= y_step * 3
        ax_backtest = plt.axes([0.75, y_pos, 0.2, 0.04])
        self.btn_backtest = widgets.Button(ax_backtest, 'Backtest', color='lightblue')
        self.btn_backtest.on_clicked(self.run_backtest)
        
    def toggle_indicator(self, indicator: str):
        """Toggle an indicator on/off."""
        if indicator in self.active_indicators:
            # Turn off
            self.active_indicators.pop(indicator)
            self.remove_indicator_plot(indicator)
            
            # Update button
            if indicator == 'RSI':
                self.btn_rsi.label.set_text('OFF')
                self.btn_rsi.color = 'lightgray'
            elif indicator == 'MACD':
                self.btn_macd.label.set_text('OFF')
                self.btn_macd.color = 'lightgray'
            elif indicator == 'BB':
                self.btn_bb.label.set_text('OFF')
                self.btn_bb.color = 'lightgray'
            elif indicator == 'SMA':
                self.btn_sma.label.set_text('OFF')
                self.btn_sma.color = 'lightgray'
        else:
            # Turn on
            self.active_indicators[indicator] = True
            self.update_indicator(indicator)
            
            # Update button
            if indicator == 'RSI':
                self.btn_rsi.label.set_text('ON')
                self.btn_rsi.color = 'lightgreen'
            elif indicator == 'MACD':
                self.btn_macd.label.set_text('ON')
                self.btn_macd.color = 'lightgreen'
            elif indicator == 'BB':
                self.btn_bb.label.set_text('ON')
                self.btn_bb.color = 'lightgreen'
            elif indicator == 'SMA':
                self.btn_sma.label.set_text('ON')
                self.btn_sma.color = 'lightgreen'
        
        plt.draw()
        
    def update_indicator(self, indicator: str):
        """Update indicator calculation and plot."""
        if indicator not in self.active_indicators:
            return
            
        close = self.data[self.columns['close']]
        
        if indicator == 'RSI':
            period = int(self.slider_rsi.val)
            rsi_values = ta.rsi(close, period)
            self.plot_rsi(rsi_values, period)
            
        elif indicator == 'MACD':
            macd_line, signal_line, histogram = ta.macd(close)
            self.plot_macd(macd_line, signal_line, histogram)
            
        elif indicator == 'BB':
            period = int(self.slider_bb_period.val)
            std_dev = self.slider_bb_std.val
            upper, middle, lower = ta.bollinger_bands(close, period, std_dev)
            self.plot_bb(upper, middle, lower)
            
        elif indicator == 'SMA':
            period = int(self.slider_sma.val)
            sma_values = ta.sma(close, period)
            self.plot_sma(sma_values, period)
            
        plt.draw()
        
    def plot_rsi(self, rsi_values: pd.Series, period: int):
        """Plot RSI indicator."""
        # Remove old RSI plot
        if 'RSI' in self.indicator_lines:
            self.indicator_lines['RSI'].remove()
            
        self.ax_momentum.clear()
        line, = self.ax_momentum.plot(self.data.index, rsi_values, 
                                      'purple', label=f'RSI({period})')
        self.indicator_lines['RSI'] = line
        
        # Add overbought/oversold lines
        self.ax_momentum.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        self.ax_momentum.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        self.ax_momentum.fill_between(self.data.index, 30, 70, alpha=0.1, color='gray')
        
        self.ax_momentum.set_ylabel('RSI')
        self.ax_momentum.set_ylim(0, 100)
        self.ax_momentum.legend()
        self.ax_momentum.grid(True, alpha=0.3)
        
    def plot_macd(self, macd_line: pd.Series, signal_line: pd.Series, histogram: pd.Series):
        """Plot MACD indicator."""
        self.ax_momentum.clear()
        
        self.ax_momentum.plot(self.data.index, macd_line, 'b-', label='MACD', linewidth=1.5)
        self.ax_momentum.plot(self.data.index, signal_line, 'r-', label='Signal', linewidth=1.5)
        self.ax_momentum.bar(self.data.index, histogram, alpha=0.3, color='gray', label='Histogram')
        
        self.ax_momentum.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        self.ax_momentum.set_ylabel('MACD')
        self.ax_momentum.legend()
        self.ax_momentum.grid(True, alpha=0.3)
        
    def plot_bb(self, upper: pd.Series, middle: pd.Series, lower: pd.Series):
        """Plot Bollinger Bands."""
        # Remove old BB plots
        for key in ['BB_upper', 'BB_middle', 'BB_lower', 'BB_fill']:
            if key in self.indicator_lines:
                if key == 'BB_fill':
                    self.indicator_lines[key].remove()
                else:
                    self.indicator_lines[key].remove()
        
        # Plot on price chart
        line_upper, = self.ax_price.plot(self.data.index, upper, 'r--', alpha=0.7, label='BB Upper')
        line_middle, = self.ax_price.plot(self.data.index, middle, 'b--', alpha=0.7, label='BB Middle')
        line_lower, = self.ax_price.plot(self.data.index, lower, 'r--', alpha=0.7, label='BB Lower')
        fill = self.ax_price.fill_between(self.data.index, upper, lower, alpha=0.1, color='gray')
        
        self.indicator_lines['BB_upper'] = line_upper
        self.indicator_lines['BB_middle'] = line_middle
        self.indicator_lines['BB_lower'] = line_lower
        self.indicator_lines['BB_fill'] = fill
        
        self.ax_price.legend()
        
    def plot_sma(self, sma_values: pd.Series, period: int):
        """Plot Simple Moving Average."""
        # Remove old SMA plot
        if 'SMA' in self.indicator_lines:
            self.indicator_lines['SMA'].remove()
            
        line, = self.ax_price.plot(self.data.index, sma_values, 
                                   'orange', label=f'SMA({period})', linewidth=2)
        self.indicator_lines['SMA'] = line
        self.ax_price.legend()
        
    def remove_indicator_plot(self, indicator: str):
        """Remove indicator from plot."""
        if indicator == 'RSI':
            self.ax_momentum.clear()
            self.ax_momentum.grid(True, alpha=0.3)
        elif indicator == 'MACD':
            self.ax_momentum.clear()
            self.ax_momentum.grid(True, alpha=0.3)
        elif indicator == 'BB':
            for key in ['BB_upper', 'BB_middle', 'BB_lower', 'BB_fill']:
                if key in self.indicator_lines:
                    if key == 'BB_fill':
                        self.indicator_lines[key].remove()
                    else:
                        self.indicator_lines[key].remove()
                    self.indicator_lines.pop(key)
        elif indicator == 'SMA':
            if 'SMA' in self.indicator_lines:
                self.indicator_lines['SMA'].remove()
                self.indicator_lines.pop('SMA')
                
        # Redraw legend
        self.ax_price.legend()
        
    def update_strategy(self, strategy: str):
        """Update trading strategy signals."""
        # Clear previous signals
        for marker in self.signal_markers['buy'] + self.signal_markers['sell']:
            marker.remove()
        self.signal_markers = {'buy': [], 'sell': []}
        
        if strategy == 'None':
            plt.draw()
            return
            
        close = self.data[self.columns['close']]
        buy_signals = pd.Series(False, index=self.data.index)
        sell_signals = pd.Series(False, index=self.data.index)
        
        if strategy == 'RSI':
            rsi_values = ta.rsi(close, int(self.slider_rsi.val))
            buy_signals = rsi_values < 30
            sell_signals = rsi_values > 70
            
        elif strategy == 'MACD':
            macd_line, signal_line, _ = ta.macd(close)
            buy_signals = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
            sell_signals = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
            
        elif strategy == 'BB':
            upper, _, lower = ta.bollinger_bands(close, 
                                                int(self.slider_bb_period.val), 
                                                self.slider_bb_std.val)
            buy_signals = close <= lower
            sell_signals = close >= upper
        
        # Plot signals
        buy_points = self.data.index[buy_signals]
        sell_points = self.data.index[sell_signals]
        
        for point in buy_points:
            marker = self.ax_price.scatter(point, close[point], 
                                         color='green', marker='^', s=100, zorder=5)
            self.signal_markers['buy'].append(marker)
            
        for point in sell_points:
            marker = self.ax_price.scatter(point, close[point], 
                                         color='red', marker='v', s=100, zorder=5)
            self.signal_markers['sell'].append(marker)
            
        plt.draw()
        
    def run_backtest(self, event):
        """Run backtest on current strategy."""
        strategy = self.radio_strategy.value_selected
        
        if strategy == 'None':
            self.show_info("Please select a strategy first!")
            return
            
        # Use SmartDataFrame for backtesting
        sdf = SmartDataFrame(self.data)
        
        # Map strategy names
        strategy_map = {
            'RSI': 'simple',
            'MACD': 'macd_cross',
            'BB': 'bb_bounce'
        }
        
        # Add required indicators
        if strategy == 'RSI':
            sdf.add_indicators('rsi')
        elif strategy == 'MACD':
            sdf.add_indicators('macd')
        elif strategy == 'BB':
            sdf.add_indicators('bb')
            
        # Run backtest
        results = sdf.backtest(strategy=strategy_map[strategy])
        
        # Display results
        info_text = f"Backtest Results ({strategy} Strategy):\n"
        info_text += "-" * 40 + "\n"
        for key, value in results.items():
            info_text += f"{key}: {value}\n"
            
        self.show_info(info_text)
        
    def show_info(self, text: str):
        """Display information in the info panel."""
        self.ax_info.clear()
        self.ax_info.text(0.05, 0.5, text, fontsize=10, 
                         verticalalignment='center', 
                         fontfamily='monospace')
        self.ax_info.axis('off')
        plt.draw()
        
    def show(self):
        """Display the interactive chart."""
        plt.show()


class IndicatorPlayground:
    """Visual playground for building and testing indicator combinations."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize the playground.
        
        Parameters
        ----------
        data : DataFrame
            OHLCV data
        """
        self.data = data
        self.sdf = SmartDataFrame(data)
        self.fig, self.axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Initialize indicator formulas
        self.custom_indicators = {}
        self.setup_playground()
        
    def setup_playground(self):
        """Setup the playground interface."""
        # Price chart
        self.ax_price = self.axes[0]
        self.ax_indicator1 = self.axes[1]
        self.ax_indicator2 = self.axes[2]
        self.ax_combined = self.axes[3]
        
        # Initial plots
        close = self.data[self.sdf.columns['close']]
        self.ax_price.plot(self.data.index, close, 'k-', linewidth=1)
        self.ax_price.set_ylabel('Price')
        self.ax_price.grid(True, alpha=0.3)
        
        # Add control buttons
        self.setup_controls()
        
    def setup_controls(self):
        """Setup control buttons and inputs."""
        # Add indicator combo button
        ax_combo = plt.axes([0.81, 0.02, 0.18, 0.04])
        self.btn_combo = widgets.Button(ax_combo, 'Create Combo Indicator')
        self.btn_combo.on_clicked(self.create_combo_indicator)
        
        # Formula input (text annotation for now)
        self.ax_price.text(0.02, 0.95, 'Indicator 1: RSI(14)', 
                          transform=self.ax_price.transAxes,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.ax_price.text(0.02, 0.88, 'Indicator 2: MACD Signal', 
                          transform=self.ax_price.transAxes,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    def create_combo_indicator(self, event):
        """Create a combination indicator."""
        # Calculate base indicators
        close = self.data[self.sdf.columns['close']]
        
        # RSI
        rsi = ta.rsi(close, 14)
        self.ax_indicator1.clear()
        self.ax_indicator1.plot(self.data.index, rsi, 'purple', label='RSI(14)')
        self.ax_indicator1.axhline(50, color='gray', linestyle='--', alpha=0.5)
        self.ax_indicator1.set_ylabel('RSI')
        self.ax_indicator1.legend()
        self.ax_indicator1.grid(True, alpha=0.3)
        
        # MACD
        macd_line, signal_line, _ = ta.macd(close)
        self.ax_indicator2.clear()
        self.ax_indicator2.plot(self.data.index, macd_line, 'b-', label='MACD')
        self.ax_indicator2.plot(self.data.index, signal_line, 'r-', label='Signal')
        self.ax_indicator2.axhline(0, color='gray', linestyle='-', alpha=0.5)
        self.ax_indicator2.set_ylabel('MACD')
        self.ax_indicator2.legend()
        self.ax_indicator2.grid(True, alpha=0.3)
        
        # Combined indicator: RSI-MACD Confluence
        # Normalize both to 0-100 scale
        macd_normalized = 50 + (macd_line / macd_line.std() * 10)
        macd_normalized = macd_normalized.clip(0, 100)
        
        # Confluence score
        confluence = (rsi + macd_normalized) / 2
        
        self.ax_combined.clear()
        self.ax_combined.plot(self.data.index, confluence, 'green', 
                            label='RSI-MACD Confluence', linewidth=2)
        self.ax_combined.axhline(70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        self.ax_combined.axhline(30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        self.ax_combined.fill_between(self.data.index, 30, 70, alpha=0.1, color='gray')
        self.ax_combined.set_ylabel('Confluence Score')
        self.ax_combined.set_ylim(0, 100)
        self.ax_combined.legend()
        self.ax_combined.grid(True, alpha=0.3)
        
        plt.draw()
        
    def show(self):
        """Display the playground."""
        plt.tight_layout()
        plt.show()


class StrategyBuilder:
    """Visual strategy builder with drag-and-drop components."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize strategy builder."""
        self.data = data
        self.sdf = SmartDataFrame(data)
        
        # Strategy components
        self.conditions = []
        self.actions = []
        
        self.setup_builder()
        
    def setup_builder(self):
        """Setup the visual builder interface."""
        self.fig = plt.figure(figsize=(14, 8))
        
        # Layout: [Components | Canvas | Preview]
        self.ax_components = plt.subplot2grid((4, 3), (0, 0), rowspan=4)
        self.ax_canvas = plt.subplot2grid((4, 3), (0, 1), rowspan=3)
        self.ax_preview = plt.subplot2grid((4, 3), (0, 2), rowspan=3)
        self.ax_results = plt.subplot2grid((4, 3), (3, 1), colspan=2)
        
        self._setup_components()
        self._setup_canvas()
        self._setup_preview()
        
    def _setup_components(self):
        """Setup component library."""
        self.ax_components.set_title('Components', fontsize=12, weight='bold')
        self.ax_components.set_xlim(0, 1)
        self.ax_components.set_ylim(0, 10)
        self.ax_components.axis('off')
        
        # Component categories
        components = {
            'Indicators': [
                ('RSI < 30', 'rsi_oversold'),
                ('RSI > 70', 'rsi_overbought'),
                ('MACD Cross Up', 'macd_bull'),
                ('MACD Cross Down', 'macd_bear'),
                ('Price < BB Lower', 'bb_lower'),
                ('Price > BB Upper', 'bb_upper'),
                ('SMA Cross', 'sma_cross')
            ],
            'Actions': [
                ('Buy Signal', 'buy'),
                ('Sell Signal', 'sell'),
                ('Stop Loss', 'stop_loss'),
                ('Take Profit', 'take_profit')
            ]
        }
        
        y_pos = 9.5
        for category, items in components.items():
            self.ax_components.text(0.1, y_pos, category, fontsize=11, weight='bold')
            y_pos -= 0.5
            
            for label, code in items:
                rect = Rectangle((0.1, y_pos - 0.3), 0.8, 0.25, 
                               facecolor='lightblue', edgecolor='black')
                self.ax_components.add_patch(rect)
                self.ax_components.text(0.5, y_pos - 0.15, label, 
                                      ha='center', va='center', fontsize=9)
                y_pos -= 0.5
                
    def _setup_canvas(self):
        """Setup strategy canvas."""
        self.ax_canvas.set_title('Strategy Logic', fontsize=12, weight='bold')
        self.ax_canvas.set_xlim(0, 10)
        self.ax_canvas.set_ylim(0, 10)
        self.ax_canvas.grid(True, alpha=0.3)
        
        # Example strategy flow
        self.ax_canvas.text(5, 9, 'IF', ha='center', fontsize=11, weight='bold')
        
        # Condition boxes
        rect1 = Rectangle((1, 7), 3, 1, facecolor='lightgreen', edgecolor='black')
        self.ax_canvas.add_patch(rect1)
        self.ax_canvas.text(2.5, 7.5, 'RSI < 30', ha='center', va='center')
        
        self.ax_canvas.text(5, 7.5, 'AND', ha='center', fontsize=10)
        
        rect2 = Rectangle((6, 7), 3, 1, facecolor='lightgreen', edgecolor='black')
        self.ax_canvas.add_patch(rect2)
        self.ax_canvas.text(7.5, 7.5, 'Price < BB Lower', ha='center', va='center')
        
        # Arrow
        self.ax_canvas.arrow(5, 6.5, 0, -1, head_width=0.3, head_length=0.2, fc='black')
        
        # Action
        self.ax_canvas.text(5, 5, 'THEN', ha='center', fontsize=11, weight='bold')
        rect3 = Rectangle((3.5, 3), 3, 1, facecolor='lightcoral', edgecolor='black')
        self.ax_canvas.add_patch(rect3)
        self.ax_canvas.text(5, 3.5, 'BUY SIGNAL', ha='center', va='center', weight='bold')
        
    def _setup_preview(self):
        """Setup results preview."""
        self.ax_preview.set_title('Signal Preview', fontsize=12, weight='bold')
        
        # Plot price with example signals
        close = self.data[self.sdf.columns['close']]
        self.ax_preview.plot(self.data.index[-100:], close.iloc[-100:], 'k-', linewidth=1)
        
        # Add example signals
        rsi = ta.rsi(close, 14)
        bb_upper, _, bb_lower = ta.bollinger_bands(close, 20, 2)
        
        buy_signals = (rsi < 30) & (close < bb_lower)
        sell_signals = (rsi > 70) & (close > bb_upper)
        
        buy_points = self.data.index[-100:][buy_signals.iloc[-100:]]
        sell_points = self.data.index[-100:][sell_signals.iloc[-100:]]
        
        for point in buy_points:
            self.ax_preview.scatter(point, close[point], color='green', 
                                  marker='^', s=100, zorder=5)
        
        for point in sell_points:
            self.ax_preview.scatter(point, close[point], color='red', 
                                  marker='v', s=100, zorder=5)
        
        self.ax_preview.set_ylabel('Price')
        self.ax_preview.grid(True, alpha=0.3)
        
        # Results summary
        self.ax_results.axis('off')
        results_text = "Strategy Performance:\n"
        results_text += f"Total Signals: {len(buy_points) + len(sell_points)}\n"
        results_text += f"Buy Signals: {len(buy_points)}\n"
        results_text += f"Sell Signals: {len(sell_points)}\n"
        results_text += f"Signal Frequency: {(len(buy_points) + len(sell_points))/100:.1%}"
        
        self.ax_results.text(0.1, 0.5, results_text, fontsize=10, 
                           verticalalignment='center', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    def show(self):
        """Display the strategy builder."""
        plt.tight_layout()
        plt.show()


# Helper functions for quick visualization
def plot_indicators(data: Union[pd.DataFrame, str], 
                   indicators: List[str] = None,
                   start_date: str = None,
                   end_date: str = None) -> None:
    """Quick plot of price data with indicators.
    
    Parameters
    ----------
    data : DataFrame or filepath
        Price data
    indicators : list
        List of indicators to plot (default: ['rsi', 'macd', 'bb'])
    start_date : str
        Start date for plot
    end_date : str
        End date for plot
    """
    # Load data if filepath
    if isinstance(data, str):
        sdf = SmartDataFrame(data)
        data = sdf.df
    
    # Filter date range
    if start_date:
        data = data[data.index >= start_date]
    if end_date:
        data = data[data.index <= end_date]
    
    # Default indicators
    if indicators is None:
        indicators = ['close', 'volume', 'rsi', 'macd', 'bb']
    
    # Create subplots
    n_plots = len([i for i in indicators if i not in ['bb', 'volume']])
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Price plot
    if 'close' in indicators:
        ax = axes[plot_idx]
        close = data['Close'] if 'Close' in data else data.iloc[:, 0]
        ax.plot(data.index, close, 'k-', linewidth=1.5, label='Close')
        
        # Add Bollinger Bands if requested
        if 'bb' in indicators:
            upper, middle, lower = ta.bollinger_bands(close)
            ax.plot(data.index, upper, 'r--', alpha=0.5, label='BB Upper')
            ax.plot(data.index, lower, 'r--', alpha=0.5, label='BB Lower')
            ax.fill_between(data.index, upper, lower, alpha=0.1, color='gray')
        
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Volume
    if 'volume' in indicators and 'Volume' in data:
        ax = axes[plot_idx]
        ax.bar(data.index, data['Volume'], alpha=0.5, color='steelblue')
        ax.set_ylabel('Volume')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # RSI
    if 'rsi' in indicators:
        ax = axes[plot_idx]
        close = data['Close'] if 'Close' in data else data.iloc[:, 0]
        rsi_values = ta.rsi(close)
        ax.plot(data.index, rsi_values, 'purple', linewidth=1.5)
        ax.axhline(70, color='r', linestyle='--', alpha=0.5)
        ax.axhline(30, color='g', linestyle='--', alpha=0.5)
        ax.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
        ax.set_ylabel('RSI')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # MACD
    if 'macd' in indicators:
        ax = axes[plot_idx]
        close = data['Close'] if 'Close' in data else data.iloc[:, 0]
        macd_line, signal_line, histogram = ta.macd(close)
        ax.plot(data.index, macd_line, 'b-', label='MACD', linewidth=1.5)
        ax.plot(data.index, signal_line, 'r-', label='Signal', linewidth=1.5)
        ax.bar(data.index, histogram, alpha=0.3, color='gray', label='Histogram')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylabel('MACD')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Main function to launch visual tools
def launch_visual_tools(data: Union[pd.DataFrame, str], 
                       tool: str = 'interactive') -> None:
    """Launch visual indicator tools.
    
    Parameters
    ----------
    data : DataFrame or filepath
        Price data
    tool : str
        Tool to launch: 'interactive', 'playground', 'builder'
    """
    # Load data if needed
    if isinstance(data, str):
        sdf = SmartDataFrame(data)
        data = sdf.df
    
    if tool == 'interactive':
        chart = InteractiveChart(data)
        chart.show()
    elif tool == 'playground':
        playground = IndicatorPlayground(data)
        playground.show()
    elif tool == 'builder':
        builder = StrategyBuilder(data)
        builder.show()
    else:
        raise ValueError(f"Unknown tool: {tool}")