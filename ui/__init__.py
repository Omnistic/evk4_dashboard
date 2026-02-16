"""
ui package

User interface components including layout and event handlers.
"""

from .layout import build_main_layout, UIComponents
from .callbacks import (
    create_toggle_dark_callback,
    create_update_plots_callback,
    create_on_shape_drawn_callback,
    create_delta_t_change_callback,
    create_frames_change_callback,
    create_pick_file_callback,
    create_update_frame_display_callback,
    create_generate_frames_callback,
    create_export_frames_callback,
)
from .plots import (
    update_histogram_plot,
    update_iei_histogram,
    update_power_spectrum,
    update_timetrace,
)

__all__ = [
    # Layout
    'build_main_layout',
    'UIComponents',
    # Callbacks
    'create_toggle_dark_callback',
    'create_update_plots_callback',
    'create_on_shape_drawn_callback',
    'create_delta_t_change_callback',
    'create_frames_change_callback',
    'create_pick_file_callback',
    'create_update_frame_display_callback',
    'create_generate_frames_callback',
    'create_export_frames_callback',
    # Plots
    'update_histogram_plot',
    'update_iei_histogram',
    'update_power_spectrum',
    'update_timetrace',
]
