"""
core/constants.py

Application-wide constants and configuration values.

Contains all magic numbers, display limits, analysis parameters, and UI options.
"""

from typing import List

# ============================================================================
# UI OPTIONS
# ============================================================================

POLARITY_OPTIONS: List[str] = [
    'BOTH',
    'CD ON (polarity=1)',
    'CD OFF (polarity=0)',
    'SIGNED (ON - OFF)'
]

BIAS_NAMES: List[str] = [
    'bias_diff',
    'bias_diff_off',
    'bias_diff_on',
    'bias_fo',
    'bias_hpf',
    'bias_refr'
]

# ============================================================================
# DISPLAY LIMITS
# ============================================================================

MAX_TIMETRACE_POINTS: int = 10000  # Maximum points to display in time trace (downsampled if exceeded)
MAX_IEI_POINTS: int = 10000  # Maximum points for inter-event interval histogram
MAX_DISPLAY_FRAMES: int = 1000  # Maximum frames to display in viewer (downsampled if exceeded)
RECONNECT_TIMEOUT: int = 120  # Reconnect timeout in seconds for native app

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

IEI_HISTOGRAM_NBINS: int = 100  # Number of bins for inter-event interval histogram
POWER_SPECTRUM_BIN_WIDTH_US: int = 100  # Temporal bin width in microseconds for power spectrum
MIN_FREQUENCY_HZ: float = 0.1  # Minimum frequency to display in power spectrum
MAX_FREQUENCY_HZ: float = 2000  # Maximum frequency to display in power spectrum

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

TIMETRACE_JITTER: float = 0.5  # Vertical jitter range for time trace scatter plot
TIMETRACE_MARGIN_RATIO: float = 0.01  # Margin as ratio of duration for time axis
