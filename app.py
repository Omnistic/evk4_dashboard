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
import asyncio
from contextlib import asynccontextmanager

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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    with overlay:
        # Card with strong shadow for visibility
        with ui.card().classes('items-center p-8 shadow-2xl border-2'):
            # Large, thick spinner in primary color (blue - visible in both modes)
            ui.spinner(size='xl', color='primary', thickness=7)
            # Bold, large text
            ui.label(message).classes('mt-4 text-xl font-bold')
    overlay.open()
    await asyncio.sleep(0.05)
    
    try:
        yield
    finally:
        overlay.close()


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
    async def on_polarity_change_callback():
        if state.current_data is None or holder.components is None:
            return
        
        async with loading_overlay('Updating polarity mode...'):
            polarity_mode = holder.components.polarity_select.value
            update_histogram_plot(state, dark.value, polarity_mode, holder.components.histogram_plot)
            update_iei_histogram(state, dark.value, polarity_mode, holder.components.iei_plot)
            update_power_spectrum(state, dark.value, polarity_mode, holder.components.spectrum_plot)
            update_timetrace(state, dark.value, polarity_mode, holder.components.timetrace_plot)
            
            # Small delay to show the spinner
            await asyncio.sleep(0.05)
    
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