"""
core/state.py

Application state management and plot configuration.

Contains centralized state container and styling configuration for the dashboard.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy.typing as npt


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
        current_time_range: Current time range filter as (t_min_s, t_max_s) in seconds,
            or None to use the full recording duration
    """
    current_file: Optional[Path] = None
    current_data: Optional[Dict[str, Any]] = None
    recording_duration_ms: float = 0.0
    generated_frames: Optional[npt.NDArray] = None
    generated_timestamps: Optional[npt.NDArray] = None
    updating: bool = False
    current_roi: Optional[Tuple[int, int, int, int]] = None
    current_time_range: Optional[Tuple[float, float]] = None


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
    timetrace_marker_size: int = 5
    
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