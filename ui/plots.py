"""
ui/plots.py

Plot update functions for event visualization and analysis.

Contains all functions for updating histograms, IEI analysis, power spectrum,
and time trace visualizations.
"""

import numpy as np
import plotly.graph_objects as go
from nicegui import ui
import traceback

from utils import (
    compute_event_histogram,
    filter_events_by_polarity,
    filter_events_by_roi,
    filter_events_by_time_range,
    create_signed_heatmap,
    create_regular_heatmap,
    get_polarity_mode_from_string,
    profile,
)
from core import (
    PLOT_CONFIG,
    MAX_TIMETRACE_POINTS,
    MAX_IEI_POINTS,
    IEI_HISTOGRAM_NBINS,
    POWER_SPECTRUM_BIN_WIDTH_US,
    MIN_FREQUENCY_HZ,
    MAX_FREQUENCY_HZ,
    TIMETRACE_JITTER,
    TIMETRACE_MARGIN_RATIO,
)
from core.validation import (
    validate_dimensions,
    validate_events_not_empty,
    validate_array_length,
    validate_positive_number,
)


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


def get_plot_template(dark_mode: bool) -> str:
    """
    Get current Plotly template based on dark mode setting.
    
    Args:
        dark_mode: Whether dark mode is active
    
    Returns:
        'plotly_dark' if dark mode is active, 'plotly' otherwise
    """
    return 'plotly_dark' if dark_mode else 'plotly'


def get_base_layout(dark_mode: bool, **kwargs) -> dict:
    """
    Get base Plotly layout with common settings.
    
    Provides consistent margins, template, and grid settings across plots.
    Additional keyword arguments override or extend base settings.
    
    Args:
        dark_mode: Whether dark mode is active
        **kwargs: Additional layout parameters to merge with base config
    
    Returns:
        Dictionary of layout parameters suitable for fig.update_layout()
    """
    base = dict(
        margin=PLOT_CONFIG.get_default_margin(),
        template=get_plot_template(dark_mode),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
    )
    base.update(kwargs)
    return base


@profile
def update_histogram_plot(state, dark_mode, polarity_mode, histogram_plot, zmin=None, zmax=None):
    """
    Update event histogram plot.
    
    Recomputes the 2D spatial histogram and redraws the heatmap only when the
    underlying data has changed (polarity mode or time range). If only display
    parameters changed (e.g. colorscale limits or dark mode), the cached histogram
    is reused to avoid unnecessary recomputation.

    If zmin/zmax are provided, the colorscale is clamped to those values — useful
    for revealing low-activity pixels when bright hotspots dominate. If either is
    None, that limit is auto-scaled from the data.
    
    Args:
        state: AppState instance
        dark_mode: Whether dark mode is active
        polarity_mode: Current polarity mode string
        histogram_plot: NiceGUI plotly component
        zmin: Optional minimum colorscale value. None = auto.
        zmax: Optional maximum colorscale value. None = auto.
    """
    if state.current_data is None:
        return
    
    try:
        events = state.current_data['events']
        width, height = int(state.current_data['width']), int(state.current_data['height'])
        mode = get_polarity_mode_from_string(polarity_mode)
        
        if not validate_dimensions(width, height):
            return
        
        if not validate_events_not_empty(events, 'plotting'):
            return

        # Check if we can reuse the cached histogram — only recompute if the
        # polarity mode or time range has changed since the last call
        cache_key = (polarity_mode, state.current_time_range)
        if state.cached_histogram is not None and state.cached_histogram_key == cache_key:
            histogram = state.cached_histogram
        else:
            events = filter_events_by_time_range(events, state.current_time_range)

            if not validate_events_not_empty(events, 'plotting'):
                return

            histogram = compute_event_histogram(events, width, height, mode)
            state.cached_histogram = histogram
            state.cached_histogram_key = cache_key
        
        # Pass user-defined colorscale limits; None values fall back to auto-scaling
        if mode == 'signed':
            signed_colorscale = PLOT_CONFIG.get_signed_colorscale(dark_mode)
            fig = create_signed_heatmap(histogram, signed_colorscale, zmin=zmin, zmax=zmax)
        else:
            fig = create_regular_heatmap(histogram, zmin=zmin, zmax=zmax)
        
        fig = apply_heatmap_layout(fig, dark_mode)
        histogram_plot.figure = fig
        histogram_plot.update()
        
    except KeyError as e:
        ui.notify(f'Missing required data field: {str(e)}', type='negative')
        traceback.print_exc()
    except ValueError as e:
        ui.notify(f'Invalid data values: {str(e)}', type='negative')
        traceback.print_exc()
    except Exception as e:
        ui.notify(f'Failed to update histogram: {str(e)}', type='negative')
        traceback.print_exc()


@profile
def update_iei_histogram(state, dark_mode, polarity_mode, iei_plot):
    """
    Update inter-event interval (IEI) histogram.
    
    Computes time differences between consecutive events and displays distribution
    on both linear (ms) and logarithmic (Hz) scales. Applies current ROI and
    polarity filters. Downsamples if data exceeds MAX_IEI_POINTS.

    Caches the computed IEI data and reuses it when only display parameters
    (e.g. dark mode) have changed.
    
    Args:
        state: AppState instance
        dark_mode: Whether dark mode is active
        polarity_mode: Current polarity mode string
        iei_plot: NiceGUI plotly component
    """
    if state.current_data is None:
        return
    
    try:
        cache_key = (polarity_mode, state.current_time_range, state.current_roi)
        if state.cached_iei is not None and state.cached_iei_key == cache_key:
            iei_ms = state.cached_iei
        else:
            events = state.current_data['events']
            mode = get_polarity_mode_from_string(polarity_mode)

            events = filter_events_by_roi(events, state.current_roi)
            events = filter_events_by_time_range(events, state.current_time_range)
            events = filter_events_by_polarity(events, mode)
            
            if not validate_array_length(events, 2, 'events for IEI histogram'):
                return
            
            iei = np.diff(events['t'])
            iei_ms = iei / 1000
            iei_ms = iei_ms[iei_ms > 0]
            
            if len(iei_ms) == 0:
                return
            
            if len(iei_ms) > MAX_IEI_POINTS:
                rng = np.random.default_rng()
                indices = rng.choice(len(iei_ms), MAX_IEI_POINTS, replace=False, shuffle=False)
                iei_ms = iei_ms[indices]

            state.cached_iei = iei_ms
            state.cached_iei_key = cache_key
        
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
            dark_mode,
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


@profile
def update_power_spectrum(state, dark_mode, polarity_mode, spectrum_plot):
    """
    Update power spectrum plot via FFT analysis.
    
    Bins events into temporal windows, computes FFT of event rate time series,
    and displays power spectrum. For signed mode, uses difference between ON
    and OFF event rates. Applies current ROI filter.

    Caches the computed (freqs, power) result and reuses it when only display
    parameters (e.g. dark mode) have changed.
    
    Args:
        state: AppState instance
        dark_mode: Whether dark mode is active
        polarity_mode: Current polarity mode string
        spectrum_plot: NiceGUI plotly component
    """
    if state.current_data is None:
        return
    
    try:
        cache_key = (polarity_mode, state.current_time_range, state.current_roi)
        if state.cached_power_spectrum is not None and state.cached_power_spectrum_key == cache_key:
            freqs, power = state.cached_power_spectrum
        else:
            events = state.current_data['events']
            mode = get_polarity_mode_from_string(polarity_mode)

            events = filter_events_by_roi(events, state.current_roi)
            events = filter_events_by_time_range(events, state.current_time_range)
            
            if not validate_events_not_empty(events, 'power spectrum'):
                return
            
            times_us = events['t']
            
            if not validate_array_length(times_us, 2, 'timestamps for power spectrum'):
                return
            
            bin_width_us = POWER_SPECTRUM_BIN_WIDTH_US
            t_min = int(times_us.min())
            t_max = int(times_us.max())
            n_bins = int((t_max - t_min) // bin_width_us) + 1

            if n_bins < 2:
                return

            if mode == 'signed':
                on_events = filter_events_by_polarity(events, 'on')
                off_events = filter_events_by_polarity(events, 'off')
                on_indices = ((on_events['t'] - t_min) // bin_width_us).astype(np.int32)
                off_indices = ((off_events['t'] - t_min) // bin_width_us).astype(np.int32)
                on_counts = np.bincount(on_indices, minlength=n_bins)
                off_counts = np.bincount(off_indices, minlength=n_bins)
                counts = on_counts.astype(np.float64) - off_counts.astype(np.float64)
            else:
                events = filter_events_by_polarity(events, mode)
                bin_indices = ((events['t'] - t_min) // bin_width_us).astype(np.int32)
                counts = np.bincount(bin_indices, minlength=n_bins).astype(np.float64)
            
            fft = np.fft.rfft(counts - counts.mean())
            power = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(len(counts), d=bin_width_us / 1e6)
            
            mask = (freqs >= MIN_FREQUENCY_HZ) & (freqs <= MAX_FREQUENCY_HZ)
            freqs = freqs[mask]
            power = power[mask]

            if len(freqs) == 0 or len(power) == 0:
                return

            state.cached_power_spectrum = (freqs, power)
            state.cached_power_spectrum_key = cache_key
        
        # Create plot
        fig = go.Figure(go.Scatter(
            x=freqs,
            y=power,
            mode='lines',
            line=dict(color=PLOT_CONFIG.color_spectrum, width=1)
        ))
        
        # Apply layout
        fig.update_layout(**get_base_layout(
            dark_mode,
            xaxis_title='Frequency (Hz)',
            yaxis_title='Power',
            yaxis_type='log',
        ))
        
        spectrum_plot.figure = fig
        spectrum_plot.update()
        
    except Exception as e:
        print(f'Error updating power spectrum: {e}')
        traceback.print_exc()


@profile
def update_timetrace(state, dark_mode, polarity_mode, timetrace_plot):
    """
    Update time trace scatter plot.
    
    Displays individual events as colored points over time, with ON events in
    orange and OFF events in blue. Vertical position is randomized (jittered)
    for visual clarity. Applies current ROI and polarity filters.
    Downsamples if data exceeds MAX_TIMETRACE_POINTS using a fast RNG.

    Caches the computed (times, colors, jitter) arrays and reuses them when
    only display parameters (e.g. dark mode) have changed.
    
    Args:
        state: AppState instance
        dark_mode: Whether dark mode is active
        polarity_mode: Current polarity mode string
        timetrace_plot: NiceGUI plotly component
    """
    if state.current_data is None:
        return
    
    try:
        # Check if we can reuse cached timetrace data — only recompute if polarity,
        # time range, or ROI has changed since the last call
        cache_key = (polarity_mode, state.current_time_range, state.current_roi)
        if state.cached_timetrace is not None and state.cached_timetrace_key == cache_key:
            times, colors, jitter = state.cached_timetrace
        else:
            events = state.current_data['events']
            mode = get_polarity_mode_from_string(polarity_mode)
            
            # Apply filters
            events = filter_events_by_roi(events, state.current_roi)
            events = filter_events_by_time_range(events, state.current_time_range)
            events = filter_events_by_polarity(events, mode)
            
            if not validate_events_not_empty(events, 'time trace'):
                timetrace_plot.visible = False
                return
            
            # Downsample if needed — use default_rng for significantly faster sampling
            if len(events) > MAX_TIMETRACE_POINTS:
                rng = np.random.default_rng()
                indices = rng.choice(len(events), MAX_TIMETRACE_POINTS, replace=False, shuffle=False)
                indices.sort()
                events = events[indices]
                ui.notify(f'Downsampled to {MAX_TIMETRACE_POINTS:,} points', type='info')
            
            # Prepare data
            times = events['t'] / 1e6
            
            if not validate_array_length(times, 2, 'timestamps for time trace'):
                timetrace_plot.visible = False
                return
            
            polarities = events['p']
            colors = np.where(polarities == 1, PLOT_CONFIG.color_on, PLOT_CONFIG.color_off)
            jitter = np.random.uniform(-TIMETRACE_JITTER, TIMETRACE_JITTER, len(times))

            state.cached_timetrace = (times, colors, jitter)
            state.cached_timetrace_key = cache_key

        duration = float(times.max() - times.min())
        
        if not validate_positive_number(duration, 'time duration', min_value=0.0, exclusive_min=True):
            timetrace_plot.visible = False
            return

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
            dark_mode,
            xaxis_title='Time (s)',
            xaxis=dict(
                range=[
                    times.min() - TIMETRACE_MARGIN_RATIO * duration, 
                    times.max() + TIMETRACE_MARGIN_RATIO * duration
                ],
                showgrid=True,
                tickmode='linear',
            ),
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