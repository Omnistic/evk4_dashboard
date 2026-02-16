"""
core package

Core application components including state management, constants, and validation.
"""

from .state import AppState, PlotConfig, PLOT_CONFIG
from .constants import (
    POLARITY_OPTIONS,
    BIAS_NAMES,
    MAX_TIMETRACE_POINTS,
    MAX_IEI_POINTS,
    MAX_DISPLAY_FRAMES,
    RECONNECT_TIMEOUT,
    IEI_HISTOGRAM_NBINS,
    POWER_SPECTRUM_BIN_WIDTH_US,
    MIN_FREQUENCY_HZ,
    MAX_FREQUENCY_HZ,
    FRAME_PERCENTILE_ZMAX,
    TIMETRACE_JITTER,
    TIMETRACE_MARGIN_RATIO,
)
from .validation import (
    validate_positive_number,
    validate_dimensions,
    validate_events_not_empty,
    validate_roi_bounds,
    validate_array_length,
)

__all__ = [
    # State
    'AppState',
    'PlotConfig',
    'PLOT_CONFIG',
    # Constants
    'POLARITY_OPTIONS',
    'BIAS_NAMES',
    'MAX_TIMETRACE_POINTS',
    'MAX_IEI_POINTS',
    'MAX_DISPLAY_FRAMES',
    'RECONNECT_TIMEOUT',
    'IEI_HISTOGRAM_NBINS',
    'POWER_SPECTRUM_BIN_WIDTH_US',
    'MIN_FREQUENCY_HZ',
    'MAX_FREQUENCY_HZ',
    'FRAME_PERCENTILE_ZMAX',
    'TIMETRACE_JITTER',
    'TIMETRACE_MARGIN_RATIO',
    # Validation
    'validate_positive_number',
    'validate_dimensions',
    'validate_events_not_empty',
    'validate_roi_bounds',
    'validate_array_length',
]
