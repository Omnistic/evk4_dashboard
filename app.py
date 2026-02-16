"""
app.py

EVK4 Event Camera Dashboard - Main Application Entry Point

A NiceGUI-based desktop application for visualizing and analyzing event-based camera data
from Prophesee EVK4 sensors. Provides interactive visualization of event histograms,
temporal analysis, power spectrum analysis, and frame generation capabilities.

This is the main entry point that wires together UI components, callbacks, and business logic.
"""

from nicegui import ui, app
import os

from core import AppState, RECONNECT_TIMEOUT
from ui import (
    build_main_layout,
    create_toggle_dark_callback,
    create_update_plots_callback,
    create_on_shape_drawn_callback,
    create_delta_t_change_callback,
    create_frames_change_callback,
    create_pick_file_callback,
    create_update_frame_display_callback,
    create_generate_frames_callback,
    create_export_frames_callback,
    update_histogram_plot,
    update_iei_histogram,
    update_power_spectrum,
    update_timetrace,
)
from services import shutdown_executor


@ui.page('/')
def main_page() -> None:
    """
    Main application page with all UI components and event handlers.
    
    Constructs the complete dashboard interface and wires up all callbacks.
    """
    # Initialize state and UI mode
    dark = ui.dark_mode(True)
    state = AppState()
    
    # We need a mutable container to hold components reference
    class ComponentsHolder:
        components = None
    holder = ComponentsHolder()
    
    # Create polarity change callback that will use holder
    def on_polarity_change_callback():
        print(f"DEBUG: Polarity changed! state.current_data={state.current_data is not None}, holder.components={holder.components is not None}")
        if state.current_data is None or holder.components is None:
            print("DEBUG: Returning early")
            return
        polarity_mode = holder.components.polarity_select.value
        print(f"DEBUG: Updating plots with mode: {polarity_mode}")
        update_histogram_plot(state, dark.value, polarity_mode, holder.components.histogram_plot)
        update_iei_histogram(state, dark.value, polarity_mode, holder.components.iei_plot)
        update_power_spectrum(state, dark.value, polarity_mode, holder.components.spectrum_plot)
        update_timetrace(state, dark.value, polarity_mode, holder.components.timetrace_plot)
        print("DEBUG: Plots updated")
    
    # Build UI layout with callback
    components = build_main_layout(dark, on_polarity_change=on_polarity_change_callback)
    holder.components = components
    
    # Create callback functions
    toggle_dark = create_toggle_dark_callback(state, dark, components)
    update_plots = create_update_plots_callback(state, dark, components)
    on_shape_drawn = create_on_shape_drawn_callback(state, dark, components)
    on_delta_t_change = create_delta_t_change_callback(state, components)
    on_frames_change = create_frames_change_callback(state, components)
    handle_pick_file = create_pick_file_callback(state, dark, components)
    update_frame_display = create_update_frame_display_callback(state, components)
    generate_frames = create_generate_frames_callback(state, dark, components)
    export_frames = create_export_frames_callback(state, components)
    
    # Wire up callbacks to UI components
    components.icon.on('click', toggle_dark)
    # Note: polarity_select callback is set during layout construction
    components.histogram_plot.on('plotly_relayout', on_shape_drawn)
    components.delta_t_input.on('update:model-value', on_delta_t_change)
    components.frames_input.on('update:model-value', on_frames_change)
    components.frame_slider.on('update:model-value', update_frame_display, throttle=0.1)
    
    # Wire up button callbacks
    components.open_file_btn.on('click', handle_pick_file)
    components.generate_frames_btn.on('click', generate_frames)
    components.export_frames_btn.on('click', export_frames)


def shutdown() -> None:
    """
    Clean shutdown handler.
    
    Ensures thread pool executor is properly closed before application exit.
    """
    print('Shutting down...')
    shutdown_executor()
    os._exit(0)


# Application lifecycle hooks
app.on_startup(lambda: app.native.main_window.maximize())
app.on_shutdown(shutdown)

# Run the application
ui.run(native=True, reconnect_timeout=RECONNECT_TIMEOUT)