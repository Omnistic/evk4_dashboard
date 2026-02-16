"""
ui/callbacks.py

Event handlers and callbacks for UI interactions.

Handles all user interactions including theme toggling, plot updates,
file loading, frame generation, and ROI selection.
"""

import asyncio
from contextlib import asynccontextmanager
import numpy as np
import plotly.graph_objects as go
from nicegui import ui
import traceback
import imageio

from utils import generate_frames, get_polarity_mode_from_string
from core import (
    PLOT_CONFIG,
    MAX_DISPLAY_FRAMES,
    FRAME_PERCENTILE_ZMAX,
)
from core.validation import validate_positive_number, validate_roi_bounds, validate_events_not_empty
from services import (
    pick_file,
    convert_raw_file,
    load_npz_data,
    extract_bias_data,
    compute_statistics,
)
from ui.plots import (
    update_histogram_plot,
    update_iei_histogram,
    update_power_spectrum,
    update_timetrace,
    get_base_layout,
)

# Thread pool executor for async operations
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)


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


# ============================================================================
# CALLBACK FUNCTIONS
# ============================================================================

def create_toggle_dark_callback(state, dark, components):
    """
    Create callback for theme toggle button.
    
    Args:
        state: AppState instance
        dark: Dark mode instance
        components: UIComponents namedtuple
    
    Returns:
        Async callback function
    """
    async def toggle_dark() -> None:
        """Toggle between dark and light mode themes."""
        async with loading_overlay('Switching theme...'):
            dark.toggle()
            components.icon.set_name('light_mode' if dark.value else 'dark_mode')
            
            # Small delay to show the spinner
            await asyncio.sleep(0.1)
            
            if state.current_data is not None:
                # Reset ROI since shapes will be lost
                state.current_roi = None
                components.roi_label.text = ''
                
                # Update signed colorscale if in signed mode
                polarity_mode = components.polarity_select.value
                if get_polarity_mode_from_string(polarity_mode) == 'signed':
                    new_colorscale = PLOT_CONFIG.get_signed_colorscale(dark.value)
                    components.histogram_plot.figure.data[0].colorscale = new_colorscale
                
                # Update all plots
                template = 'plotly_dark' if dark.value else 'plotly'
                components.histogram_plot.figure.update_layout(template=template)
                components.histogram_plot.update()
                
                update_iei_histogram(state, dark.value, polarity_mode, components.iei_plot)
                update_power_spectrum(state, dark.value, polarity_mode, components.spectrum_plot)
                update_timetrace(state, dark.value, polarity_mode, components.timetrace_plot)
                
                if components.timetrace_plot.visible:
                    components.timetrace_plot.figure.update_layout(template=template)
                    components.timetrace_plot.update()
                
                if components.iei_plot.figure and hasattr(components.iei_plot.figure, 'update_layout'):
                    components.iei_plot.figure.update_layout(template=template)
                    components.iei_plot.update()
                    components.spectrum_plot.figure.update_layout(template=template)
                    components.spectrum_plot.update()
            
            # Update frame viewer if frames are generated
            if state.generated_frames is not None:
                frame_polarity_mode = components.frame_polarity_select.value
                if get_polarity_mode_from_string(frame_polarity_mode) == 'signed':
                    new_colorscale = PLOT_CONFIG.get_signed_colorscale(dark.value)
                    components.frame_plot.figure.data[0].colorscale = new_colorscale
                
                template = 'plotly_dark' if dark.value else 'plotly'
                components.frame_plot.figure.update_layout(template=template)
                components.frame_plot.update()
    
    return toggle_dark


def create_update_plots_callback(state, dark, components):
    """
    Create callback for updating all plots.
    
    Args:
        state: AppState instance
        dark: Dark mode instance
        components: UIComponents namedtuple
    
    Returns:
        Async callback function
    """
    async def update_plots() -> None:
        """Update all plots with current data and settings."""
        if state.current_data is None:
            return
        
        async with loading_overlay('Updating plots...'):
            polarity_mode = components.polarity_select.value
            update_histogram_plot(state, dark.value, polarity_mode, components.histogram_plot)
            update_iei_histogram(state, dark.value, polarity_mode, components.iei_plot)
            update_power_spectrum(state, dark.value, polarity_mode, components.spectrum_plot)
            update_timetrace(state, dark.value, polarity_mode, components.timetrace_plot)
    
    return update_plots


def create_on_shape_drawn_callback(state, dark, components):
    """
    Create callback for ROI shape drawing.
    
    Args:
        state: AppState instance
        dark: Dark mode instance
        components: UIComponents namedtuple
    
    Returns:
        Async callback function
    """
    async def on_shape_drawn(e) -> None:
        """Handle ROI shape drawing on histogram."""
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
                components.roi_label.text = ''
                await create_update_plots_callback(state, dark, components)()
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
            components.roi_label.text = f'ROI: ({x_min}, {y_min}) → ({x_max}, {y_max})'
            
            async with loading_overlay('Updating ROI plots...'):
                try:
                    polarity_mode = components.polarity_select.value
                    update_power_spectrum(state, dark.value, polarity_mode, components.spectrum_plot)
                    update_iei_histogram(state, dark.value, polarity_mode, components.iei_plot)
                    update_timetrace(state, dark.value, polarity_mode, components.timetrace_plot)
                except Exception as e:
                    ui.notify(f'Failed to update ROI plots: {str(e)}', type='negative')
                    print(f'ROI update error: {e}')
                    traceback.print_exc()
                
        except Exception as e:
            print(f'Error handling shape draw: {e}')
            traceback.print_exc()
    
    return on_shape_drawn


def create_delta_t_change_callback(state, components):
    """Create callback for delta T input changes."""
    def on_delta_t_change() -> None:
        """Handle delta T input change."""
        if state.updating or state.recording_duration_ms == 0 or components.delta_t_input.value is None:
            return
        state.updating = True
        delta_t = max(0.01, float(components.delta_t_input.value))
        frames = int(state.recording_duration_ms / delta_t)
        components.frames_input.value = max(1, frames)
        state.updating = False
    
    return on_delta_t_change


def create_frames_change_callback(state, components):
    """Create callback for frame count input changes."""
    def on_frames_change() -> None:
        """Handle frame count input change."""
        if state.updating or state.recording_duration_ms == 0 or components.frames_input.value is None:
            return
        state.updating = True
        frames = max(1, int(components.frames_input.value))
        delta_t = state.recording_duration_ms / frames
        components.delta_t_input.value = round(delta_t, 2)
        state.updating = False
    
    return on_frames_change


def create_pick_file_callback(state, dark, components):
    """
    Create callback for file picker button.
    
    Args:
        state: AppState instance
        dark: Dark mode instance
        components: UIComponents namedtuple
    
    Returns:
        Async callback function
    """
    async def handle_pick_file() -> None:
        """Open file picker and process selected file."""
        path = await pick_file()
        if path is None:
            return
        
        await process_file_full(path, state, dark, components)
    
    return handle_pick_file


async def process_file_full(path, state, dark, components):
    """
    Complete file processing pipeline.
    
    Args:
        path: Path to file
        state: AppState instance
        dark: Dark mode instance
        components: UIComponents namedtuple
    """
    state.current_roi = None
    components.roi_label.text = ''
    suffix = path.suffix.lower()
    
    # Handle .raw files (convert to .npz first)
    if suffix == '.raw':
        async with loading_overlay(f'Converting {path.name}...'):
            npz_path = await convert_raw_file(path, components.overwrite_toggle.value)
            if npz_path is None:
                return
        state.current_file = npz_path
    elif suffix == '.npz':
        state.current_file = path
        ui.notify(f'Loaded: {path.name}')
    
    components.file_label.text = f'Loaded: {state.current_file.name}'
    components.file_label.visible = True
    
    # Load data
    state.current_data = load_npz_data(state.current_file)
    if state.current_data is None:
        components.file_label.visible = False
        state.current_file = None
        return
    
    events = state.current_data['events']
    width, height = int(state.current_data['width']), int(state.current_data['height'])
    
    # Validate data
    if not validate_events_not_empty(events, 'file loading'):
        components.file_label.visible = False
        state.current_file = None
        state.current_data = None
        return
    
    # Update bias table
    bias_columns, bias_row = extract_bias_data(state.current_data)
    if bias_columns:
        components.bias_table.columns = bias_columns
        components.bias_table.rows = [bias_row]
        components.bias_table.visible = True
    else:
        components.bias_table.visible = False
    
    # Compute statistics
    stats = compute_statistics(events)
    if stats is None:
        return
    
    # Update stats table
    components.stats_table.columns = [
        {'name': 'events', 'label': 'Events', 'field': 'events'},
        {'name': 'duration', 'label': 'Duration', 'field': 'duration'},
        {'name': 'rate', 'label': 'Event rate', 'field': 'rate'},
        {'name': 'resolution', 'label': 'Resolution', 'field': 'resolution'},
    ]
    components.stats_table.rows = [{
        'events': f'{stats["event_count"]:,}',
        'duration': f'{stats["duration"]:.2f} s',
        'rate': f'{stats["event_rate"]:,.0f} ev/s',
        'resolution': f'{width} x {height}',
    }]
    
    # Initialize frame generation parameters
    state.recording_duration_ms = stats['duration'] * 1000
    components.frames_input.value = 100
    create_frames_change_callback(state, components)()
    
    # Update all plots
    try:
        await create_update_plots_callback(state, dark, components)()
        components.data_section.visible = True
    except Exception as e:
        ui.notify(f'Failed to generate plots: {str(e)}', type='negative')
        print(f'Plot generation error: {e}')
        traceback.print_exc()


def create_update_frame_display_callback(state, components):
    """Create callback for frame slider."""
    def update_frame_display() -> None:
        """Update displayed frame based on slider position."""
        if state.generated_frames is None or len(state.generated_frames) == 0:
            return
        
        try:
            idx = int(components.frame_slider.value)
            
            # Validate index
            if idx < 0 or idx >= len(state.generated_frames):
                print(f'Warning: Invalid frame index {idx}')
                return
            
            components.frame_plot.figure.data[0].z = state.generated_frames[idx]
            components.frame_plot.update()
            components.frame_index_label.text = f'Frame {idx + 1} / {len(state.generated_frames)}'
            
        except Exception as e:
            print(f'Error updating frame display: {e}')
            traceback.print_exc()
    
    return update_frame_display


def create_generate_frames_callback(state, dark, components):
    """Create callback for generate frames button."""
    async def generate_frames_callback() -> None:
        """Generate frames from event data for visualization."""
        if state.current_data is None:
            ui.notify('No data loaded', type='warning')
            return
        
        events = state.current_data['events']
        width, height = int(state.current_data['width']), int(state.current_data['height'])
        delta_t = components.delta_t_input.value
        mode = get_polarity_mode_from_string(components.frame_polarity_select.value)
        
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
                dark.value,
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                yaxis=dict(scaleanchor='x', scaleratio=1, autorange='reversed'),
            ))
            
            # Update slider and display
            components.frame_slider._props['max'] = n_frames - 1
            components.frame_slider.value = 0
            components.frame_slider.update()
            
            components.frame_plot.figure = fig
            components.frame_viewer.visible = True
            components.frame_plot.update()
            components.frame_index_label.text = f'Frame 1 / {n_frames}'
            
            ui.notify(f'Generated {original_n_frames} frames')
            
        except Exception as e:
            ui.notify(f'Failed to display frames: {str(e)}', type='negative')
            print(f'Frame display error: {e}')
            traceback.print_exc()
    
    return generate_frames_callback


def create_export_frames_callback(state, components):
    """Create callback for export frames button."""
    async def export_frames_callback() -> None:
        """Export frames to multi-page TIFF file."""
        if state.current_data is None:
            ui.notify('No data loaded', type='warning')
            return
        
        events = state.current_data['events']
        width, height = int(state.current_data['width']), int(state.current_data['height'])
        delta_t = components.delta_t_input.value
        mode = get_polarity_mode_from_string(components.frame_polarity_select.value)
        
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
    
    return export_frames_callback