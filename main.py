"""
app.py

EVK4 Event Camera Dashboard - Main Application

A NiceGUI-based desktop application for visualizing and analyzing event-based camera data
from Prophesee EVK4 sensors. Provides interactive visualization of event histograms,
temporal analysis, power spectrum analysis, and frame generation capabilities.

Key Features:
    - Load and visualize .raw and .npz event data files
    - Interactive ROI selection for focused analysis
    - Multiple polarity modes (BOTH, ON, OFF, SIGNED)
    - Inter-event interval and power spectrum analysis
    - Frame generation with adjustable temporal resolution
    - Export frames as TIFF sequences
    - Dark/light theme support
"""

from nicegui import ui, app
from pathlib import Path
from utils import (
    raw_to_npz, 
    compute_event_histogram, 
    generate_frames,
    get_polarity_mode_from_string,
    filter_events_by_polarity,
    filter_events_by_roi,
    create_signed_heatmap,
    create_regular_heatmap,
)
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import asyncio
import imageio
import numpy as np
import numpy.typing as npt
import os
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import traceback

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

@dataclass
class AppState:
    """
    Centralized application state container.
    
    Attributes:
        current_file: Path to the currently loaded data file
        current_data: Dictionary containing loaded event data and metadata
        recording_duration_ms: Duration of the recording in milliseconds
        generated_frames: Array of generated frames for visualization
        generated_timestamps: Timestamps corresponding to each generated frame
        updating: Flag to prevent recursive updates during UI changes
        current_roi: Current region of interest as (x_min, x_max, y_min, y_max)
    """
    current_file: Optional[Path] = None
    current_data: Optional[Dict[str, Any]] = None
    recording_duration_ms: float = 0.0
    generated_frames: Optional[npt.NDArray] = None
    generated_timestamps: Optional[npt.NDArray] = None
    updating: bool = False
    current_roi: Optional[Tuple[int, int, int, int]] = None

# ============================================================================
# CONSTANTS
# ============================================================================

# Threading
executor = ThreadPoolExecutor(max_workers=1)

# UI Options
POLARITY_OPTIONS: List[str] = ['BOTH', 'CD ON (polarity=1)', 'CD OFF (polarity=0)', 'SIGNED (ON - OFF)']
BIAS_NAMES: List[str] = ['bias_diff', 'bias_diff_off', 'bias_diff_on', 'bias_fo', 'bias_hpf', 'bias_refr']

# Display limits
MAX_TIMETRACE_POINTS: int = 10000  # Maximum points to display in time trace (downsampled if exceeded)
MAX_IEI_POINTS: int = 10000  # Maximum points for inter-event interval histogram
MAX_DISPLAY_FRAMES: int = 1000  # Maximum frames to display in viewer (downsampled if exceeded)
RECONNECT_TIMEOUT: int = 120  # Reconnect timeout in seconds for native app

# Analysis parameters
IEI_HISTOGRAM_NBINS: int = 100  # Number of bins for inter-event interval histogram
POWER_SPECTRUM_BIN_WIDTH_US: int = 100  # Temporal bin width in microseconds for power spectrum
MIN_FREQUENCY_HZ: float = 0.1  # Minimum frequency to display in power spectrum
MAX_FREQUENCY_HZ: float = 2000  # Maximum frequency to display in power spectrum

# Visualization parameters
FRAME_PERCENTILE_ZMAX: int = 99  # Use 99th percentile for frame colorscale max
TIMETRACE_JITTER: float = 0.5  # Vertical jitter range for time trace scatter plot
TIMETRACE_MARGIN_RATIO: float = 0.01  # Margin as ratio of duration for time axis

# ============================================================================
# PLOT CONFIGURATION
# ============================================================================

@dataclass
class PlotConfig:
    """
    Centralized plot styling configuration.
    
    Provides consistent theming and styling across all plots including colors,
    margins, and ROI visualization settings. Supports both light and dark modes.
    
    Attributes:
        color_on: Color for ON polarity events (orange)
        color_off: Color for OFF polarity events (blue)
        color_spectrum: Color for spectrum/analysis plots (pink)
        signed_colorscale_light: Diverging colorscale for signed mode in light theme
        signed_colorscale_dark: Diverging colorscale for signed mode in dark theme
        roi_line_color: Color for ROI rectangle border
        roi_line_width: Width of ROI rectangle border
        roi_fill_color: Fill color for ROI rectangle (semi-transparent)
        histogram_marker_line_color: Color for histogram bar borders
        histogram_marker_line_width: Width of histogram bar borders
        default_margin_l/r/t/b: Default plot margins (left/right/top/bottom)
        timetrace_margin_l/r/t/b: Time trace specific margins
        timetrace_marker_size: Marker size for time trace scatter plot
    """
    
    # Event polarity colors
    color_on: str = '#E69F00'  # Orange
    color_off: str = '#56B4E9'  # Blue
    color_spectrum: str = '#CC79A7'  # Pink
    
    # Signed colorscale (light mode)
    signed_colorscale_light: List[List] = None
    # Signed colorscale (dark mode)
    signed_colorscale_dark: List[List] = None
    
    # ROI styling
    roi_line_color: str = 'cyan'
    roi_line_width: int = 2
    roi_fill_color: str = 'rgba(0,255,255,0.2)'
    
    # Histogram styling
    histogram_marker_line_color: str = 'white'
    histogram_marker_line_width: float = 0.5
    
    # Margins
    default_margin_l: int = 50
    default_margin_r: int = 50
    default_margin_t: int = 50
    default_margin_b: int = 50
    
    # Timetrace specific
    timetrace_margin_l: int = 0
    timetrace_margin_r: int = 0
    timetrace_margin_t: int = 50
    timetrace_margin_b: int = 50
    timetrace_marker_size: int = 3
    
    def __post_init__(self):
        """Initialize colorscales after dataclass creation."""
        if self.signed_colorscale_light is None:
            self.signed_colorscale_light = [
                [0, f'rgb(86, 180, 233)'],  # Blue (OFF)
                [0.5, 'rgba(0, 0, 0, 0)'],  # Transparent black center
                [1, f'rgb(230, 159, 0)']    # Orange (ON)
            ]
        
        if self.signed_colorscale_dark is None:
            self.signed_colorscale_dark = [
                [0, f'rgb(86, 180, 233)'],  # Blue (OFF)
                [0.5, 'rgba(255, 255, 255, 0)'],  # Transparent white center
                [1, f'rgb(230, 159, 0)']    # Orange (ON)
            ]
    
    def get_signed_colorscale(self, dark_mode: bool) -> List[List]:
        """
        Get appropriate signed colorscale based on theme.
        
        Args:
            dark_mode: Whether dark mode is active
        
        Returns:
            Colorscale list suitable for Plotly heatmaps
        """
        return self.signed_colorscale_dark if dark_mode else self.signed_colorscale_light
    
    def get_default_margin(self) -> Dict[str, int]:
        """
        Get default margin dictionary for Plotly layouts.
        
        Returns:
            Dictionary with keys 'l', 'r', 't', 'b' for margins
        """
        return dict(
            l=self.default_margin_l,
            r=self.default_margin_r,
            t=self.default_margin_t,
            b=self.default_margin_b
        )
    
    def get_timetrace_margin(self) -> Dict[str, int]:
        """
        Get timetrace-specific margin dictionary.
        
        Time trace uses minimal left/right margins for better horizontal space usage.
        
        Returns:
            Dictionary with keys 'l', 'r', 't', 'b' for margins
        """
        return dict(
            l=self.timetrace_margin_l,
            r=self.timetrace_margin_r,
            t=self.timetrace_margin_t,
            b=self.timetrace_margin_b
        )


# Create global plot configuration instance
PLOT_CONFIG = PlotConfig()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

@ui.page('/')
def main_page() -> None:
    """
    Main application page with all UI components and event handlers.
    
    Constructs the complete dashboard interface including:
    - File loading controls
    - Recording statistics display
    - Interactive event histogram with ROI selection
    - Analysis plots (IEI, power spectrum, time trace)
    - Frame generation and export controls
    """
    dark = ui.dark_mode(True)
    state = AppState()

    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================

    def get_polarity_mode() -> str:
        """
        Get current polarity mode from main selector.
        
        Returns:
            Mode string: 'all', 'on', 'off', or 'signed'
        """
        return get_polarity_mode_from_string(polarity_select.value)

    def get_frame_polarity_mode() -> str:
        """
        Get polarity mode from frame generation selector.
        
        Returns:
            Mode string: 'all', 'on', 'off', or 'signed'
        """
        return get_polarity_mode_from_string(frame_polarity_select.value)

    def apply_heatmap_layout(fig: go.Figure, dark_mode: bool) -> go.Figure:
        """
        Apply common layout configuration to event histogram heatmap.
        
        Configures axis labels, aspect ratio, theme, and ROI drawing tools.
        
        Args:
            fig: Plotly figure to configure
            dark_mode: Whether dark mode is active
        
        Returns:
            Configured figure
        """
        template = 'plotly_dark' if dark_mode else 'plotly'
        fig.update_layout(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            yaxis=dict(scaleanchor='x', scaleratio=1, autorange='reversed'),
            margin=PLOT_CONFIG.get_default_margin(),
            template=template,
            dragmode='drawrect',
            newshape=dict(
                line=dict(color=PLOT_CONFIG.roi_line_color, width=PLOT_CONFIG.roi_line_width),
                fillcolor=PLOT_CONFIG.roi_fill_color
            ),
            modebar_add=['drawrect', 'eraseshape']
        )
        return fig

    def get_plot_template() -> str:
        """
        Get current Plotly template based on dark mode setting.
        
        Returns:
            'plotly_dark' if dark mode is active, 'plotly' otherwise
        """
        return 'plotly_dark' if dark.value else 'plotly'

    def get_base_layout(**kwargs: Any) -> Dict[str, Any]:
        """
        Get base Plotly layout with common settings.
        
        Provides consistent margins, template, and grid settings across plots.
        Additional keyword arguments override or extend base settings.
        
        Args:
            **kwargs: Additional layout parameters to merge with base config
        
        Returns:
            Dictionary of layout parameters suitable for fig.update_layout()
        """
        base: Dict[str, Any] = dict(
            margin=PLOT_CONFIG.get_default_margin(),
            template=get_plot_template(),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
        )
        base.update(kwargs)
        return base

    @asynccontextmanager
    async def loading_overlay(message: str = 'Loading...'):
        """
        Context manager for displaying loading overlays during async operations.
        
        Shows a persistent dialog with spinner and message while work is in progress.
        Automatically closes when context exits.
        
        Args:
            message: Message to display in the overlay
        
        Yields:
            None
        
        Example:
            async with loading_overlay('Processing data...'):
                result = await some_async_function()
        """
        overlay = ui.dialog().props('persistent')
        with overlay, ui.card().classes('items-center p-8'):
            ui.spinner(size='xl')
            ui.label(message).classes('mt-4')
        overlay.open()
        await asyncio.sleep(0.05)
        
        try:
            yield
        finally:
            overlay.close()

    # ========================================================================
    # VALIDATION HELPERS
    # ========================================================================

    def validate_positive_number(
        value: Optional[float], 
        name: str, 
        min_value: float = 0.0,
        exclusive_min: bool = True
    ) -> bool:
        """
        Validate that a number is positive and within bounds.
        
        Args:
            value: The value to validate
            name: Name of the parameter (for error messages)
            min_value: Minimum allowed value
            exclusive_min: If True, value must be > min_value; if False, >= min_value
        
        Returns:
            True if valid, False otherwise (displays notification on failure)
        """
        if value is None:
            ui.notify(f'{name} is required', type='negative')
            return False
        
        if exclusive_min and value <= min_value:
            ui.notify(f'{name} must be greater than {min_value}', type='negative')
            return False
        
        if not exclusive_min and value < min_value:
            ui.notify(f'{name} must be at least {min_value}', type='negative')
            return False
        
        return True

    def validate_dimensions(width: int, height: int) -> bool:
        """
        Validate sensor dimensions are positive.
        
        Args:
            width: Sensor width in pixels
            height: Sensor height in pixels
        
        Returns:
            True if valid, False otherwise (displays notification on failure)
        """
        if width <= 0 or height <= 0:
            ui.notify(f'Invalid sensor dimensions: {width}x{height}', type='negative')
            return False
        return True

    def validate_events_not_empty(events: npt.NDArray[np.void], context: str = 'operation') -> bool:
        """
        Validate that event array is not empty.
        
        Args:
            events: Event array to validate
            context: Context description for error message
        
        Returns:
            True if valid, False otherwise (displays notification on failure)
        """
        if len(events) == 0:
            ui.notify(f'No events available for {context}', type='warning')
            return False
        return True

    def validate_roi_bounds(
        roi: Tuple[int, int, int, int],
        width: int,
        height: int
    ) -> bool:
        """
        Validate ROI bounds are within sensor dimensions.
        
        Args:
            roi: Tuple of (x_min, x_max, y_min, y_max)
            width: Sensor width in pixels
            height: Sensor height in pixels
        
        Returns:
            True if valid, False otherwise (displays notification on failure)
        """
        x_min, x_max, y_min, y_max = roi
        
        if x_min < 0 or x_max >= width or y_min < 0 or y_max >= height:
            ui.notify(f'ROI ({x_min},{y_min})-({x_max},{y_max}) is outside sensor bounds (0,0)-({width-1},{height-1})', type='warning')
            return False
        
        if x_min >= x_max or y_min >= y_max:
            ui.notify('ROI has zero or negative area', type='warning')
            return False
        
        return True

    def validate_array_length(
        arr: npt.NDArray,
        min_length: int,
        name: str
    ) -> bool:
        """
        Validate array has minimum required length.
        
        Args:
            arr: Array to validate
            min_length: Minimum required length
            name: Name of the array (for error messages)
        
        Returns:
            True if valid, False otherwise (displays notification on failure)
        """
        if len(arr) < min_length:
            ui.notify(f'Not enough data in {name}: need at least {min_length}, got {len(arr)}', type='warning')
            return False
        return True

    # ========================================================================
    # UI CALLBACKS
    # ========================================================================

    def toggle_dark() -> None:
        """
        Toggle between dark and light mode themes.
        
        Updates all plots with new theme, resets ROI (since shapes are theme-dependent),
        and updates signed mode colorscales if applicable.
        """
        dark.toggle()
        icon.set_name('light_mode' if dark.value else 'dark_mode')
        template = get_plot_template()

        if state.current_data is not None:
            # Reset ROI since shapes will be lost
            state.current_roi = None
            roi_label.text = ''
            
            # Update signed colorscale if in signed mode
            if get_polarity_mode() == 'signed':
                new_colorscale = PLOT_CONFIG.get_signed_colorscale(dark.value)
                histogram_plot.figure.data[0].colorscale = new_colorscale
            
            # Update histogram plot
            histogram_plot.figure.update_layout(template=template)
            histogram_plot.update()
            
            # Update all other plots (with whole sensor data since ROI was reset)
            update_iei_histogram()
            update_power_spectrum()
            update_timetrace()
            
            # Update plot templates
            if timetrace_plot.visible:
                timetrace_plot.figure.update_layout(template=template)
                timetrace_plot.update()
            
            if iei_plot.figure and hasattr(iei_plot.figure, 'update_layout'):
                iei_plot.figure.update_layout(template=template)
                iei_plot.update()
                spectrum_plot.figure.update_layout(template=template)
                spectrum_plot.update()

        # Update frame viewer if frames are generated
        if state.generated_frames is not None:
            # Update frame colorscale if in signed mode
            if get_frame_polarity_mode() == 'signed':
                new_colorscale = PLOT_CONFIG.get_signed_colorscale(dark.value)
                frame_plot.figure.data[0].colorscale = new_colorscale
            
            frame_plot.figure.update_layout(template=template)
            frame_plot.update()

    async def update_plots() -> None:
        """
        Update all plots with current data and settings.
        
        Recomputes event histogram based on current polarity mode and updates
        all analysis plots (IEI histogram, power spectrum, time trace).
        """
        if state.current_data is None:
            return
        
        async with loading_overlay('Updating plots...'):
            try:
                events = state.current_data['events']
                width, height = int(state.current_data['width']), int(state.current_data['height'])
                mode = get_polarity_mode()
                
                # Validate inputs
                if not validate_dimensions(width, height):
                    return
                
                if not validate_events_not_empty(events, 'plotting'):
                    return
                
                histogram = compute_event_histogram(events, width, height, mode)
                
                # Create appropriate heatmap based on mode
                if mode == 'signed':
                    signed_colorscale = PLOT_CONFIG.get_signed_colorscale(dark.value)
                    fig = create_signed_heatmap(histogram, signed_colorscale)
                else:
                    fig = create_regular_heatmap(histogram)
                
                # Apply layout and update
                fig = apply_heatmap_layout(fig, dark.value)
                histogram_plot.figure = fig
                histogram_plot.update()

                # Update analysis plots
                update_iei_histogram()
                update_power_spectrum()
                update_timetrace()
                
            except KeyError as e:
                ui.notify(f'Missing required data field: {str(e)}', type='negative')
                print(f'Data error: {e}')
                traceback.print_exc()
            except ValueError as e:
                ui.notify(f'Invalid data values: {str(e)}', type='negative')
                print(f'Value error: {e}')
                traceback.print_exc()
            except Exception as e:
                ui.notify(f'Failed to update plots: {str(e)}', type='negative')
                print(f'Plot update error: {e}')
                traceback.print_exc()

    def update_iei_histogram() -> None:
        """
        Update inter-event interval (IEI) histogram.
        
        Computes time differences between consecutive events and displays distribution
        on both linear (ms) and logarithmic (Hz) scales. Applies current ROI and
        polarity filters. Downsamples if data exceeds MAX_IEI_POINTS.
        """
        if state.current_data is None:
            return
        
        try:
            events = state.current_data['events']
            mode = get_polarity_mode()

            # Apply filters
            events = filter_events_by_roi(events, state.current_roi)
            events = filter_events_by_polarity(events, mode)
            
            if not validate_array_length(events, 2, 'events for IEI histogram'):
                return
            
            # Compute inter-event intervals
            iei = np.diff(events['t'])
            iei_ms = iei / 1000
            iei_ms = iei_ms[iei_ms > 0]
            
            if len(iei_ms) == 0:
                return
            
            # Downsample if needed
            if len(iei_ms) > MAX_IEI_POINTS:
                indices = np.random.choice(len(iei_ms), MAX_IEI_POINTS, replace=False)
                iei_ms = iei_ms[indices]
            
            iei_min, iei_max = float(iei_ms.min()), float(iei_ms.max())
            
            if iei_min <= 0 or iei_max <= 0:
                print(f'Warning: Invalid IEI range: {iei_min} to {iei_max}')
                return
            
            # Create histogram
            fig = go.Figure(go.Histogram(
                x=iei_ms.tolist(),
                nbinsx=IEI_HISTOGRAM_NBINS,
                marker_color=PLOT_CONFIG.color_spectrum,
                marker_line=dict(
                    color=PLOT_CONFIG.histogram_marker_line_color,
                    width=PLOT_CONFIG.histogram_marker_line_width
                )
            ))
            
            # Apply layout with dual axes (ms and Hz)
            fig.update_layout(**get_base_layout(
                xaxis_title='Inter-event interval (ms)',
                yaxis_title='Count',
                yaxis_type='log',
                xaxis2=dict(
                    title='Frequency (Hz)',
                    overlaying='x',
                    side='top',
                    range=[1000 / iei_max, 1000 / iei_min],
                    showgrid=False,
                ),
            ))
            fig.add_trace(go.Scatter(x=[], y=[], xaxis='x2'))
            
            iei_plot.figure = fig
            iei_plot.update()
            
        except Exception as e:
            print(f'Error updating IEI histogram: {e}')
            traceback.print_exc()

    def update_power_spectrum() -> None:
        """
        Update power spectrum plot via FFT analysis.
        
        Bins events into temporal windows, computes FFT of event rate time series,
        and displays power spectrum. For signed mode, uses difference between ON
        and OFF event rates. Applies current ROI filter.
        """
        if state.current_data is None:
            return
        
        try:
            events = state.current_data['events']
            mode = get_polarity_mode()

            # Apply ROI filter
            events = filter_events_by_roi(events, state.current_roi)
            
            if not validate_events_not_empty(events, 'power spectrum'):
                return
            
            # Create time bins
            times_us = events['t']
            
            if not validate_array_length(times_us, 2, 'timestamps for power spectrum'):
                return
            
            bin_width_us = POWER_SPECTRUM_BIN_WIDTH_US
            bins = np.arange(times_us.min(), times_us.max() + bin_width_us, bin_width_us)
            
            if not validate_array_length(bins, 2, 'bins for power spectrum'):
                return
            
            # Compute event counts per bin
            if mode == 'signed':
                on_events = filter_events_by_polarity(events, 'on')
                off_events = filter_events_by_polarity(events, 'off')
                on_counts, _ = np.histogram(on_events['t'], bins=bins)
                off_counts, _ = np.histogram(off_events['t'], bins=bins)
                counts = on_counts.astype(np.float64) - off_counts.astype(np.float64)
            else:
                events = filter_events_by_polarity(events, mode)
                times_us = events['t']
                counts, _ = np.histogram(times_us, bins=bins)
            
            # Compute FFT
            if not validate_array_length(counts, 2, 'counts for FFT'):
                return
                
            fft = np.fft.rfft(counts - counts.mean())
            power = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(len(counts), d=bin_width_us / 1e6)
            
            # Filter frequency range
            mask = (freqs >= MIN_FREQUENCY_HZ) & (freqs <= MAX_FREQUENCY_HZ)
            freqs = freqs[mask]
            power = power[mask]
            
            if len(freqs) == 0 or len(power) == 0:
                print('Warning: No frequencies in valid range')
                return
            
            # Create plot
            fig = go.Figure(go.Scatter(
                x=freqs,
                y=power,
                mode='lines',
                line=dict(color=PLOT_CONFIG.color_spectrum, width=1)
            ))
            
            # Apply layout
            fig.update_layout(**get_base_layout(
                xaxis_title='Frequency (Hz)',
                yaxis_title='Power',
                yaxis_type='log',
            ))
            
            spectrum_plot.figure = fig
            spectrum_plot.update()
            
        except Exception as e:
            print(f'Error updating power spectrum: {e}')
            traceback.print_exc()

    def update_timetrace() -> None:
        """
        Update time trace scatter plot.
        
        Displays individual events as colored points over time, with ON events in
        orange and OFF events in blue. Vertical position is randomized (jittered)
        for visual clarity. Applies current ROI and polarity filters.
        Downsamples if data exceeds MAX_TIMETRACE_POINTS.
        """
        if state.current_data is None:
            return
        
        try:
            events = state.current_data['events']
            mode = get_polarity_mode()
            
            # Apply filters
            events = filter_events_by_roi(events, state.current_roi)
            events = filter_events_by_polarity(events, mode)
            
            if not validate_events_not_empty(events, 'time trace'):
                timetrace_plot.visible = False
                return
            
            # Downsample if needed
            if len(events) > MAX_TIMETRACE_POINTS:
                indices = np.random.choice(len(events), MAX_TIMETRACE_POINTS, replace=False)
                indices.sort()
                events = events[indices]
                ui.notify(f'Downsampled to {MAX_TIMETRACE_POINTS:,} points', type='info')
            
            # Prepare data
            times = events['t'] / 1e6
            
            if not validate_array_length(times, 2, 'timestamps for time trace'):
                timetrace_plot.visible = False
                return
            
            duration = float(times.max() - times.min())
            
            if not validate_positive_number(duration, 'time duration', min_value=0.0, exclusive_min=True):
                timetrace_plot.visible = False
                return
            
            polarities = events['p']
            colors = np.where(polarities == 1, PLOT_CONFIG.color_on, PLOT_CONFIG.color_off)
            jitter = np.random.uniform(-TIMETRACE_JITTER, TIMETRACE_JITTER, len(times))

            # Create plot
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=times,
                y=jitter,
                mode='markers',
                marker=dict(size=PLOT_CONFIG.timetrace_marker_size, color=colors),
            ))
            
            # Apply layout
            fig.update_layout(**get_base_layout(
                xaxis_title='Time (s)',
                xaxis_range=[
                    times.min() - TIMETRACE_MARGIN_RATIO * duration, 
                    times.max() + TIMETRACE_MARGIN_RATIO * duration
                ],
                yaxis=dict(visible=False, range=[-0.6, 0.6]),
                margin=PLOT_CONFIG.get_timetrace_margin(),
            ))
            
            timetrace_plot.visible = True
            timetrace_plot.figure = fig
            timetrace_plot.update()
            
        except Exception as e:
            print(f'Error updating time trace: {e}')
            traceback.print_exc()
            timetrace_plot.visible = False

    async def on_shape_drawn(e: Any) -> None:
        """
        Handle ROI shape drawing on histogram.
        
        Extracts ROI bounds from drawn rectangle, validates them, and updates
        all analysis plots to use the new ROI. If shapes are cleared, resets
        to full sensor view.
        
        Args:
            e: Event object containing shape data from Plotly
        """
        if e.args is None or state.current_data is None:
            return
        
        try:
            args = e.args
            if 'shapes' not in args:
                return
            
            shapes = args['shapes']
            
            # If no shapes or shapes cleared, reset ROI
            if not shapes or len(shapes) == 0:
                state.current_roi = None
                roi_label.text = ''
                await update_plots()
                return
            
            # Extract ROI bounds from last drawn shape
            shape = shapes[-1]
            
            # Validate shape has required fields
            if not all(key in shape for key in ['x0', 'x1', 'y0', 'y1']):
                print('Warning: Invalid shape format')
                return
            
            x_min, x_max = int(shape['x0']), int(shape['x1'])
            y_min, y_max = int(shape['y0']), int(shape['y1'])

            # Ensure min < max
            if x_min > x_max:
                x_min, x_max = x_max, x_min
            if y_min > y_max:
                y_min, y_max = y_max, y_min
            
            # Validate bounds
            width = int(state.current_data['width'])
            height = int(state.current_data['height'])
            
            if not validate_roi_bounds((x_min, x_max, y_min, y_max), width, height):
                return

            state.current_roi = (x_min, x_max, y_min, y_max)
            roi_label.text = f'ROI: ({x_min}, {y_min}) → ({x_max}, {y_max})'
            
            async with loading_overlay('Updating ROI plots...'):
                try:
                    # Update plots with ROI filter
                    update_power_spectrum()
                    update_iei_histogram()
                    update_timetrace()
                except Exception as e:
                    ui.notify(f'Failed to update ROI plots: {str(e)}', type='negative')
                    print(f'ROI update error: {e}')
                    traceback.print_exc()
                
        except Exception as e:
            print(f'Error handling shape draw: {e}')
            traceback.print_exc()

    def on_delta_t_change() -> None:
        """
        Handle delta T input change.
        
        Updates frame count to maintain consistency with recording duration.
        Uses updating flag to prevent recursive updates.
        """
        if state.updating or state.recording_duration_ms == 0 or delta_t_input.value is None:
            return
        state.updating = True
        delta_t = max(0.01, float(delta_t_input.value))
        frames = int(state.recording_duration_ms / delta_t)
        frames_input.value = max(1, frames)
        state.updating = False

    def on_frames_change() -> None:
        """
        Handle frame count input change.
        
        Updates delta T to maintain consistency with recording duration.
        Uses updating flag to prevent recursive updates.
        """
        if state.updating or state.recording_duration_ms == 0 or frames_input.value is None:
            return
        state.updating = True
        frames = max(1, int(frames_input.value))
        delta_t = state.recording_duration_ms / frames
        delta_t_input.value = round(delta_t, 2)
        state.updating = False

    # ========================================================================
    # FILE HANDLING
    # ========================================================================

    async def pick_file() -> None:
        """
        Open native file picker dialog.
        
        Allows user to select .raw or .npz event data files for loading.
        """
        result = await app.native.main_window.create_file_dialog(allow_multiple=False)
        if result and len(result) > 0:
            path = Path(result[0])
            if path.suffix.lower() in ('.raw', '.npz'):
                await process_file(path)
            else:
                ui.notify('Please select a .raw or .npz file', type='negative')

    async def process_file(path: Path) -> None:
        """
        Load and process event data file.
        
        Handles both .raw (converts to .npz first) and .npz files. Loads event data,
        extracts metadata and bias settings, computes statistics, and initializes
        all visualizations.
        
        Args:
            path: Path to .raw or .npz file
        """
        state.current_roi = None
        roi_label.text = ''
        suffix = path.suffix.lower()
        
        # Handle .raw files (convert to .npz first)
        if suffix == '.raw':
            npz_path = path.with_suffix('.npz')
            if npz_path.exists() and not overwrite_toggle.value:
                ui.notify(f'{npz_path.name} already exists, loading existing file.', type='warning')
            else:
                async with loading_overlay(f'Converting {path.name}...'):
                    try:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(executor, raw_to_npz, path, overwrite_toggle.value)
                        ui.notify(f'Saved: {npz_path}')
                    except FileNotFoundError:
                        ui.notify(f'File not found: {path}', type='negative')
                        return
                    except PermissionError:
                        ui.notify(f'Permission denied: {path}', type='negative')
                        return
                    except Exception as e:
                        ui.notify(f'Failed to convert file: {str(e)}', type='negative')
                        print(f'Conversion error: {e}')
                        traceback.print_exc()
                        return
                        
            state.current_file = npz_path
        elif suffix == '.npz':
            state.current_file = path
            ui.notify(f'Loaded: {path.name}')
        
        file_label.text = f'Loaded: {state.current_file.name}'
        file_label.visible = True
        
        # Load data with error handling
        try:
            state.current_data = dict(np.load(state.current_file))
            events = state.current_data['events']
            width, height = int(state.current_data['width']), int(state.current_data['height'])
        except KeyError as e:
            ui.notify(f'Invalid file format: missing {str(e)}', type='negative')
            file_label.visible = False
            state.current_file = None
            state.current_data = None
            return
        except ValueError as e:
            ui.notify(f'Invalid data in file: {str(e)}', type='negative')
            file_label.visible = False
            state.current_file = None
            state.current_data = None
            return
        except Exception as e:
            ui.notify(f'Failed to load file: {str(e)}', type='negative')
            file_label.visible = False
            state.current_file = None
            state.current_data = None
            print(f'Load error: {e}')
            traceback.print_exc()
            return
        
        # Validate data
        if not validate_events_not_empty(events, 'file loading'):
            file_label.visible = False
            state.current_file = None
            state.current_data = None
            return
        
        # Update bias table if bias data exists
        bias_columns: List[Dict[str, str]] = []
        bias_row: Dict[str, int] = {}
        for name in BIAS_NAMES:
            if name in state.current_data:
                try:
                    bias_columns.append({'name': name, 'label': name, 'field': name})
                    bias_row[name] = int(state.current_data[name])
                except (ValueError, TypeError) as e:
                    print(f'Warning: Could not parse bias {name}: {e}')
                    continue
        
        if bias_columns:
            bias_table.columns = bias_columns
            bias_table.rows = [bias_row]
            bias_table.visible = True
        else:
            bias_table.visible = False

        # Compute statistics with validation
        try:
            duration = float((events['t'][-1] - events['t'][0]) / 1e6)
            if not validate_positive_number(duration, 'recording duration', min_value=0.0, exclusive_min=True):
                return
                
            event_count = len(events)
            event_rate = event_count / duration
        except Exception as e:
            ui.notify(f'Failed to compute statistics: {str(e)}', type='negative')
            print(f'Statistics error: {e}')
            traceback.print_exc()
            return
        
        # Update stats table
        stats_table.columns = [
            {'name': 'events', 'label': 'Events', 'field': 'events'},
            {'name': 'duration', 'label': 'Duration', 'field': 'duration'},
            {'name': 'rate', 'label': 'Event rate', 'field': 'rate'},
            {'name': 'resolution', 'label': 'Resolution', 'field': 'resolution'},
        ]
        stats_table.rows = [{
            'events': f'{event_count:,}',
            'duration': f'{duration:.2f} s',
            'rate': f'{event_rate:,.0f} ev/s',
            'resolution': f'{width} x {height}',
        }]

        # Initialize frame generation parameters
        state.recording_duration_ms = duration * 1000
        frames_input.value = 100
        on_frames_change()

        # Update all plots
        try:
            await update_plots()
            data_section.visible = True
        except Exception as e:
            ui.notify(f'Failed to generate plots: {str(e)}', type='negative')
            print(f'Plot generation error: {e}')
            traceback.print_exc()

    # ========================================================================
    # FRAME GENERATION
    # ========================================================================

    def update_frame_display() -> None:
        """
        Update displayed frame based on slider position.
        
        Updates the frame heatmap and index label when user moves the slider.
        """
        if state.generated_frames is None or len(state.generated_frames) == 0:
            return
        
        try:
            idx = int(frame_slider.value)
            
            # Validate index
            if idx < 0 or idx >= len(state.generated_frames):
                print(f'Warning: Invalid frame index {idx}')
                return
            
            frame_plot.figure.data[0].z = state.generated_frames[idx]
            frame_plot.update()
            frame_index_label.text = f'Frame {idx + 1} / {len(state.generated_frames)}'
            
        except Exception as e:
            print(f'Error updating frame display: {e}')
            traceback.print_exc()

    async def generate_frames_callback() -> None:
        """
        Generate frames from event data for visualization.
        
        Creates frame sequence based on current delta T and polarity settings.
        Downsamples to MAX_DISPLAY_FRAMES if necessary. Updates frame viewer
        with appropriate colorscale for the selected polarity mode.
        """
        if state.current_data is None:
            ui.notify('No data loaded', type='warning')
            return
        
        events = state.current_data['events']
        width, height = int(state.current_data['width']), int(state.current_data['height'])
        delta_t = delta_t_input.value
        mode = get_frame_polarity_mode()
        
        # Validate inputs
        if not validate_positive_number(delta_t, 'ΔT', min_value=0.0, exclusive_min=True):
            return
        
        async with loading_overlay('Generating frames...'):
            try:
                loop = asyncio.get_event_loop()
                frames, timestamps = await loop.run_in_executor(
                    executor, generate_frames, events, width, height, delta_t, mode
                )
            except Exception as e:
                ui.notify(f'Failed to generate frames: {str(e)}', type='negative')
                print(f'Frame generation error: {e}')
                traceback.print_exc()
                return
        
        if len(frames) == 0:
            ui.notify('No frames generated', type='warning')
            return
        
        n_frames = len(frames)
        original_n_frames = n_frames
        
        # Downsample for display if needed
        if n_frames > MAX_DISPLAY_FRAMES:
            indices = np.linspace(0, n_frames - 1, MAX_DISPLAY_FRAMES, dtype=int)
            frames = frames[indices]
            timestamps = timestamps[indices]
            n_frames = MAX_DISPLAY_FRAMES
            ui.notify(f'Downsampled from {original_n_frames} to {MAX_DISPLAY_FRAMES} frames for display', type='info')
        
        state.generated_frames = frames
        state.generated_timestamps = timestamps
        
        try:
            # Create frame plot with appropriate colorscale
            if mode == 'signed':
                max_abs = max(abs(frames.min()), abs(frames.max()), 1)
                
                frame_colorscale = PLOT_CONFIG.get_signed_colorscale(dark.value)
                
                fig = go.Figure(go.Heatmap(
                    z=frames[0],
                    colorscale=frame_colorscale,
                    zmid=0,
                    zmin=-max_abs,
                    zmax=max_abs,
                    colorbar=dict(title='ON - OFF')
                ))
            else:
                zmin = 0
                zmax = max(1, np.percentile(frames[frames > 0], FRAME_PERCENTILE_ZMAX)) if np.any(frames > 0) else 1
                fig = go.Figure(go.Heatmap(
                    z=frames[0],
                    colorscale='Viridis',
                    zmin=zmin,
                    zmax=zmax,
                    colorbar=dict(title='Count')
                ))
            
            fig.update_layout(**get_base_layout(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                yaxis=dict(scaleanchor='x', scaleratio=1, autorange='reversed'),
            ))
            
            # Update slider and display
            frame_slider._props['max'] = n_frames - 1
            frame_slider.value = 0
            frame_slider.update()
            
            frame_plot.figure = fig
            frame_viewer.visible = True
            frame_plot.update()
            frame_index_label.text = f'Frame 1 / {n_frames}'
            
            ui.notify(f'Generated {original_n_frames} frames')
            
        except Exception as e:
            ui.notify(f'Failed to display frames: {str(e)}', type='negative')
            print(f'Frame display error: {e}')
            traceback.print_exc()

    async def export_frames_callback() -> None:
        """
        Export frames to multi-page TIFF file.
        
        Generates full frame sequence (not downsampled) and exports as 16-bit TIFF.
        Output filename includes delta T parameter for identification.
        """
        if state.current_data is None:
            ui.notify('No data loaded', type='warning')
            return
        
        events = state.current_data['events']
        width, height = int(state.current_data['width']), int(state.current_data['height'])
        delta_t = delta_t_input.value
        mode = get_frame_polarity_mode()
        
        # Validate inputs
        if not validate_positive_number(delta_t, 'ΔT', min_value=0.0, exclusive_min=True):
            return
        
        async with loading_overlay('Exporting frames...'):
            try:
                loop = asyncio.get_event_loop()
                frames, _ = await loop.run_in_executor(
                    executor, generate_frames, events, width, height, delta_t, mode
                )
            except Exception as e:
                ui.notify(f'Failed to generate frames: {str(e)}', type='negative')
                print(f'Frame generation error: {e}')
                traceback.print_exc()
                return
        
        if len(frames) == 0:
            ui.notify('No frames to export', type='warning')
            return
        
        try:
            delta_t_us = int(delta_t * 1000)
            output_path = state.current_file.with_name(f'{state.current_file.stem}_frames_{delta_t_us}us.tif')
            
            # Check if output directory is writable
            if not output_path.parent.exists():
                ui.notify(f'Output directory does not exist: {output_path.parent}', type='negative')
                return
                
            imageio.mimwrite(str(output_path), frames.astype(np.uint16))
            ui.notify(f'Exported {len(frames)} frames to {output_path.name}')
            
        except PermissionError:
            ui.notify(f'Permission denied writing to {output_path}', type='negative')
        except OSError as e:
            ui.notify(f'Failed to write file: {str(e)}', type='negative')
        except Exception as e:
            ui.notify(f'Failed to export frames: {str(e)}', type='negative')
            print(f'Export error: {e}')
            traceback.print_exc()

    # ========================================================================
    # UI LAYOUT
    # ========================================================================

    with ui.header().classes('justify-between items-center'):
        ui.label('EVK4 Dashboard').classes('text-xl font-bold')
        icon = ui.icon('light_mode', size='md').classes('cursor-pointer')
        icon.on('click', toggle_dark)

    with ui.row().classes('w-full items-center gap-4'):
        ui.button('Open File', on_click=pick_file).classes('w-64')
        overwrite_toggle = ui.switch('OVERWRITE')
        file_label = ui.label().classes('text-gray-400')
        file_label.visible = False

    ui.separator()

    with ui.column().classes('w-full gap-4') as data_section:
        ui.label('Recording Info').classes('text-lg font-bold')
        with ui.row().classes('w-full items-center gap-4'):
            stats_table = ui.table(
                columns=[], 
                rows=[], 
                column_defaults={'align': 'center', 'headerClasses': 'uppercase text-primary'}
            )
            bias_table = ui.table(
                columns=[], 
                rows=[], 
                column_defaults={'align': 'center', 'headerClasses': 'uppercase text-primary'}
            )

        ui.separator()

        ui.label('Event Visualization').classes('text-lg font-bold')
        with ui.row().classes('w-full items-center'):
            polarity_select = ui.select(
                options=POLARITY_OPTIONS,
                value='BOTH',
                label='MODE',
                on_change=update_plots
            ).classes('w-48')
            roi_label = ui.label('').classes('text-gray-400')
            ui.space()
            cd_on_badge = ui.badge('CD ON (polarity=1)').style(f'background-color: {PLOT_CONFIG.color_on} !important')
            cd_off_badge = ui.badge('CD OFF (polarity=0)').style(f'background-color: {PLOT_CONFIG.color_off} !important')
        
        with ui.row().classes('w-full flex-nowrap gap-0'):
            histogram_plot = ui.plotly({})
            histogram_plot.on('plotly_relayout', on_shape_drawn)
            timetrace_plot = ui.plotly({}).classes('flex-grow')
            timetrace_plot.visible = False
        
        with ui.row().classes('w-full flex-nowrap gap-0'):
            spectrum_plot = ui.plotly({}).classes('flex-grow')
            iei_plot = ui.plotly({})

        ui.separator()

        ui.label('Frame Generation').classes('text-lg font-bold')
        with ui.row().classes('w-full items-center gap-4'):
            generate_button = ui.button('Generate Frames')
            export_button = ui.button('Export TIFF')
            delta_t_input = ui.number(
                label='ΔT (ms)', 
                value=33, 
                min=0.01, 
                step=1, 
                format='%.2f'
            ).classes('w-40')
            frames_input = ui.number(
                label='Frames', 
                value=100, 
                min=1, 
                step=1
            ).classes('w-40')
            frame_polarity_select = ui.select(
                options=POLARITY_OPTIONS, 
                value='BOTH', 
                label='MODE'
            ).classes('w-48')
        
        delta_t_input.on('update:model-value', on_delta_t_change)
        frames_input.on('update:model-value', on_frames_change)

        with ui.column().classes('w-full') as frame_viewer:
            frame_plot = ui.plotly({})
            with ui.row().classes('items-center gap-4').style('width: 25%'):
                frame_slider = ui.slider(min=0, max=99, value=0, step=1).classes('flex-grow')
                frame_index_label = ui.label('Frame 0 / 0')
        frame_viewer.visible = False

    data_section.visible = False

    # Connect frame generation callbacks
    frame_slider.on('update:model-value', update_frame_display, throttle=0.1)
    generate_button.on('click', generate_frames_callback)
    export_button.on('click', export_frames_callback)


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

def shutdown() -> None:
    """
    Clean shutdown handler.
    
    Ensures thread pool executor is properly closed before application exit.
    """
    print('Shutting down...')
    executor.shutdown(wait=False)
    os._exit(0)


app.on_startup(lambda: app.native.main_window.maximize())
app.on_shutdown(shutdown)
ui.run(native=True, reconnect_timeout=RECONNECT_TIMEOUT)