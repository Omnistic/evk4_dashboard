"""
utils.py

Utility functions for event-based camera data processing.

This module provides core functionality for working with event camera data from
Prophesee EVK4 sensors, including:
    - File format conversion (.raw to .npz)
    - Event filtering by polarity and spatial region
    - Histogram computation for spatial analysis
    - Frame generation with temporal binning
    - Plotly figure creation for visualization

Event Data Format:
    Events are stored as structured NumPy arrays with fields:
        - 't': timestamp in microseconds (uint64)
        - 'x': horizontal pixel coordinate (uint16)
        - 'y': vertical pixel coordinate (uint16)
        - 'p': polarity (0=OFF, 1=ON) (uint8)
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional
import warnings
import plotly.graph_objects as go
from pathlib import Path
from tqdm import tqdm
from metavision_sdk_stream import Camera, CameraStreamSlicer

# ============================================================================
# FILE I/O
# ============================================================================

def raw_to_npz(file_path: Path, overwrite: bool = False) -> None:
    """
    Convert .raw event file to compressed .npz format.
    
    Reads event data using Prophesee SDK and saves to NumPy compressed format
    for faster loading. Also extracts and saves sensor dimensions and bias
    settings if available.
    
    Args:
        file_path: Path to input .raw file
        overwrite: If True, overwrite existing .npz file; if False, skip conversion
                  if .npz already exists
    
    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If input file is not a .raw file
    
    Note:
        Output .npz file is created in the same directory as input with .npz extension.
        Bias settings are read from companion .bias file if present.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(path)
    
    if path.suffix.lower() != '.raw':
        raise ValueError(f'Expected .raw file, got: {path.suffix}')

    npz_path = path.with_suffix('.npz')
    if npz_path.exists() and not overwrite:
        warnings.warn(f'{npz_path} already exists, skipping (use overwrite=True to replace)')
        return

    # Load event data using Prophesee SDK
    camera = Camera.from_file(str(path))
    slicer = CameraStreamSlicer(camera.move())
    width = slicer.camera().width()
    height = slicer.camera().height()
    chunks = []
    for slice in tqdm(slicer, desc='Converting'):
        if slice.events.size > 0:
            chunks.append(slice.events.copy())
    all_events = np.concatenate(chunks)

    # Load bias settings if available
    bias_path = path.with_suffix('.bias')
    biases = {}
    if bias_path.exists():
        with open(bias_path, 'r') as f:
            for line in f:
                parts = line.strip().split('%')
                if len(parts) == 2:
                    value = int(parts[0].strip())
                    name = parts[1].strip()
                    biases[name] = value

    # Save to compressed format
    np.savez(npz_path, events=all_events, width=width, height=height, **biases)

# ============================================================================
# POLARITY MODE CONVERSION
# ============================================================================

def get_polarity_mode_from_string(polarity_string: str) -> str:
    """
    Convert UI polarity selection string to internal mode identifier.
    
    Translates human-readable polarity options from the UI into standardized
    mode strings used throughout the application.
    
    Args:
        polarity_string: One of:
            - 'BOTH': Include all events
            - 'CD ON (polarity=1)': Only ON events
            - 'CD OFF (polarity=0)': Only OFF events
            - 'SIGNED (ON - OFF)': Signed difference between ON and OFF
    
    Returns:
        Mode string: 'all', 'on', 'off', or 'signed'
    
    Example:
        >>> get_polarity_mode_from_string('CD ON (polarity=1)')
        'on'
        >>> get_polarity_mode_from_string('BOTH')
        'all'
    """
    if polarity_string == 'CD ON (polarity=1)':
        return 'on'
    elif polarity_string == 'CD OFF (polarity=0)':
        return 'off'
    elif polarity_string == 'SIGNED (ON - OFF)':
        return 'signed'
    return 'all'

# ============================================================================
# EVENT FILTERING
# ============================================================================

def filter_events_by_polarity(
    events: npt.NDArray[np.void], 
    mode: str
) -> npt.NDArray[np.void]:
    """
    Filter events based on polarity mode.
    
    Args:
        events: Event array with 'p' field containing polarity values
        mode: Filter mode:
            - 'on': Return only ON events (p=1)
            - 'off': Return only OFF events (p=0)
            - 'all': Return all events unchanged
            - 'signed': Return all events unchanged (filtering happens in analysis)
    
    Returns:
        Filtered event array. For 'all' and 'signed' modes, returns input unchanged.
    
    Example:
        >>> events = load_events()  # Load some events
        >>> on_events = filter_events_by_polarity(events, 'on')
        >>> len(on_events) <= len(events)
        True
    """
    if mode == 'on':
        return events[events['p'] == 1]
    elif mode == 'off':
        return events[events['p'] == 0]
    return events

def filter_events_by_roi(
    events: npt.NDArray[np.void], 
    roi: Optional[Tuple[int, int, int, int]]
) -> npt.NDArray[np.void]:
    """
    Filter events to only those within region of interest (ROI) bounds.
    
    Args:
        events: Event array with 'x' and 'y' coordinate fields
        roi: ROI bounds as (x_min, x_max, y_min, y_max), or None for no filtering.
            Coordinates are inclusive on both ends.
    
    Returns:
        Filtered event array. If roi is None, returns input unchanged.
    
    Example:
        >>> events = load_events()
        >>> roi = (100, 200, 150, 250)  # Select 100x100 pixel region
        >>> roi_events = filter_events_by_roi(events, roi)
    """
    if roi is None:
        return events
    
    x_min, x_max, y_min, y_max = roi
    mask = (
        (events['x'] >= x_min) & (events['x'] <= x_max) &
        (events['y'] >= y_min) & (events['y'] <= y_max)
    )
    return events[mask]

def filter_events_by_time_range(
    events: npt.NDArray[np.void],
    time_range: Optional[Tuple[float, float]]
) -> npt.NDArray[np.void]:
    """
    Filter events to only those within a time range.

    Args:
        events: Event array with 't' timestamp field in microseconds
        time_range: Time range as (t_min_s, t_max_s) in seconds, or None for no filtering.

    Returns:
        Filtered event array. If time_range is None, returns input unchanged.
    """
    if time_range is None:
        return events
    t_min_us = int(time_range[0] * 1e6)
    t_max_us = int(time_range[1] * 1e6)
    mask = (events['t'] >= t_min_us) & (events['t'] <= t_max_us)
    return events[mask]

# ============================================================================
# HISTOGRAM COMPUTATION
# ============================================================================

def compute_event_histogram(
    events: npt.NDArray[np.void], 
    width: int, 
    height: int, 
    mode: str = 'all'
) -> npt.NDArray[np.int32] | npt.NDArray[np.uint32]:
    """
    Compute 2D histogram of event locations (spatial event density).
    
    Counts events at each pixel location. For signed mode, computes the difference
    between ON and OFF event counts at each pixel.
    
    Args:
        events: Event array with 'x', 'y', and 'p' fields
        width: Sensor width in pixels
        height: Sensor height in pixels
        mode: Processing mode:
            - 'all': Count all events
            - 'on': Count only ON events
            - 'off': Count only OFF events
            - 'signed': Compute ON count minus OFF count (signed difference)
    
    Returns:
        2D array of shape (height, width) containing event counts per pixel.
        For signed mode, returns int32 array (can be negative).
        For other modes, returns uint32 array (non-negative).
    
    Example:
        >>> events = load_events()
        >>> histogram = compute_event_histogram(events, 640, 480, mode='all')
        >>> histogram.shape
        (480, 640)
        >>> histogram.max()  # Pixel with most events
        1523
    """
    width = int(width)
    height = int(height)
    
    if mode == 'signed':
        # Signed mode: ON events - OFF events
        on_events = events[events['p'] == 1]
        off_events = events[events['p'] == 0]
        
        histogram = np.zeros((height, width), dtype=np.int32)
        
        # Add ON events
        if len(on_events) > 0:
            coords, counts = np.unique(
                on_events['y'].astype(np.int64) * width + on_events['x'].astype(np.int64), 
                return_counts=True
            )
            histogram.flat[coords] += counts.astype(np.int32)
        
        # Subtract OFF events
        if len(off_events) > 0:
            coords, counts = np.unique(
                off_events['y'].astype(np.int64) * width + off_events['x'].astype(np.int64), 
                return_counts=True
            )
            histogram.flat[coords] -= counts.astype(np.int32)
        
        return histogram
    
    # Regular modes: filter first, then count
    if mode == 'on':
        events = events[events['p'] == 1]
    elif mode == 'off':
        events = events[events['p'] == 0]
    
    histogram = np.zeros((height, width), dtype=np.uint32)
    coords, counts = np.unique(
        events['y'].astype(np.int64) * width + events['x'].astype(np.int64), 
        return_counts=True
    )
    histogram.flat[coords] = counts
    return histogram

# ============================================================================
# PLOTLY FIGURE CREATION
# ============================================================================

def create_signed_heatmap(
    histogram: npt.NDArray[np.int32],
    signed_colorscale: list,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> go.Figure:
    """
    Create Plotly heatmap for signed (ON - OFF) event data.
    
    Uses diverging colorscale centered at zero. By default, limits are set
    symmetrically around zero based on the data range. Manual zmin/zmax can
    be provided to clamp the colorscale, which is useful when bright hotspots
    would otherwise wash out the rest of the data.
    
    Args:
        histogram: 2D array with signed event counts (int32)
        signed_colorscale: Plotly colorscale list, typically diverging from
                          blue (negative) through transparent (zero) to orange (positive)
        zmin: Minimum value for colorscale. If None, uses -max_abs (auto symmetric).
        zmax: Maximum value for colorscale. If None, uses +max_abs (auto symmetric).
    
    Returns:
        Plotly Figure object ready for display or further layout configuration
    
    Example:
        >>> histogram = compute_event_histogram(events, 640, 480, mode='signed')
        >>> colorscale = [[0, 'blue'], [0.5, 'white'], [1, 'orange']]
        >>> # Auto scale
        >>> fig = create_signed_heatmap(histogram, colorscale)
        >>> # Manual clamp to [-50, 50]
        >>> fig = create_signed_heatmap(histogram, colorscale, zmin=-50, zmax=50)
    """
    max_abs = max(abs(histogram.min()), abs(histogram.max()), 1)
    return go.Figure(go.Heatmap(
        z=histogram,
        colorscale=signed_colorscale,
        zmid=0,
        zmin=zmin if zmin is not None else -max_abs,
        zmax=zmax if zmax is not None else max_abs,
        colorbar=dict(title='ON - OFF')
    ))

def create_regular_heatmap(
    histogram: npt.NDArray[np.uint32],
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> go.Figure:
    """
    Create Plotly heatmap for regular (unsigned) event count data.
    
    Uses standard Viridis colorscale for non-negative event counts. By default,
    the colorscale spans the full data range. Manual zmin/zmax can be provided
    to clamp the colorscale, which is useful when bright hotspots would otherwise
    wash out the rest of the data.
    
    Args:
        histogram: 2D array with event counts (uint32)
        zmin: Minimum value for colorscale. If None, Plotly auto-scales from data.
        zmax: Maximum value for colorscale. If None, Plotly auto-scales from data.
    
    Returns:
        Plotly Figure object ready for display or further layout configuration
    
    Example:
        >>> histogram = compute_event_histogram(events, 640, 480, mode='all')
        >>> # Auto scale
        >>> fig = create_regular_heatmap(histogram)
        >>> # Clamp to [0, 100] to reveal low-activity pixels
        >>> fig = create_regular_heatmap(histogram, zmin=0, zmax=100)
    """
    return go.Figure(go.Heatmap(
        z=histogram,
        colorscale='Viridis',
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title='Count')
    ))

def display_event_histogram(histogram: npt.NDArray) -> None:
    """
    Display event histogram in a browser window (standalone use).
    
    Convenience function for quick visualization during development or
    interactive analysis. Creates a basic Plotly figure and opens in browser.
    
    Args:
        histogram: 2D array of event counts
    
    Example:
        >>> histogram = compute_event_histogram(events, 640, 480)
        >>> display_event_histogram(histogram)  # Opens in browser
    """
    fig = go.Figure(go.Heatmap(z=histogram))
    fig.show()

# ============================================================================
# FRAME GENERATION
# ============================================================================

def generate_frames(
    events: npt.NDArray[np.void], 
    width: int, 
    height: int, 
    delta_t_ms: float, 
    mode: str = 'all',
    clip_value: Optional[int] = 65535,
    scale_to_fit: bool = False
) -> Tuple[npt.NDArray[np.int32] | npt.NDArray[np.uint16], npt.NDArray[np.float64]]:
    """
    Generate video frames from events by accumulating events in time windows.
    
    Events are binned into temporal windows of size delta_t_ms. For non-signed modes,
    event counts per pixel are accumulated. For signed mode, ON events are added and
    OFF events are subtracted to produce signed frames.
    
    Frame values can be clipped to uint16 range (default) for storage efficiency,
    scaled to preserve dynamic range, or kept as full uint32.
    
    Args:
        events: Structured array of events with fields 't', 'x', 'y', 'p'
        width: Frame width in pixels
        height: Frame height in pixels
        delta_t_ms: Time window for each frame in milliseconds
        mode: Polarity mode:
            - 'all': Accumulate all events
            - 'on': Only ON events
            - 'off': Only OFF events
            - 'signed': ON events minus OFF events (results in signed int32 frames)
        clip_value: Maximum value for clipping event counts. Default 65535 (uint16 max).
                   Pixels with more events are capped at this value.
                   Set to None to disable clipping and keep full uint32 range.
        scale_to_fit: If True, linearly scale all frames so max value equals clip_value.
                     Preserves relative event counts across all pixels but may reduce
                     contrast in individual frames. Only applies when clip_value is set.
    
    Returns:
        Tuple containing:
            - frames: Array of shape (n_frames, height, width)
                     dtype is int32 for signed mode (can be negative)
                     dtype is uint16 for clipped non-signed modes
                     dtype is uint32 for non-clipped non-signed modes
            - timestamps: Array of frame start times in microseconds (uint64)
    
    Examples:
        >>> events = load_events()
        >>> 
        >>> # Default: clip at 65535 for uint16 storage
        >>> frames, ts = generate_frames(events, 640, 480, 33.0)
        >>> frames.dtype
        dtype('uint16')
        >>> 
        >>> # Scale to preserve dynamic range without hard clipping
        >>> frames, ts = generate_frames(events, 640, 480, 33.0, scale_to_fit=True)
        >>> 
        >>> # No clipping, keep full uint32 range
        >>> frames, ts = generate_frames(events, 640, 480, 33.0, clip_value=None)
        >>> frames.dtype
        dtype('uint32')
        >>> 
        >>> # Signed mode for ON-OFF difference
        >>> frames, ts = generate_frames(events, 640, 480, 33.0, mode='signed')
        >>> frames.dtype
        dtype('int32')
    
    Note:
        For signed mode, clip_value and scale_to_fit are ignored since frames
        contain negative values. Signed frames are always returned as int32.
    """
    width = int(width)
    height = int(height)
    
    if len(events) == 0:
        return np.array([]), np.array([])
    
    delta_t_us = delta_t_ms * 1000
    t_start = events['t'][0]
    t_end = events['t'][-1]
    
    # Use floor (int()) instead of ceil to match UI calculation
    # This ensures that if user requests N frames, they get exactly N frames
    n_frames = int((t_end - t_start) / delta_t_us)
    
    # Ensure we have at least 1 frame
    if n_frames < 1:
        n_frames = 1
    
    timestamps = np.arange(n_frames) * delta_t_us + t_start
    
    if mode == 'signed':
        # Signed mode: ON events positive, OFF events negative
        on_events = events[events['p'] == 1]
        off_events = events[events['p'] == 0]
        
        frames = np.zeros((n_frames, height, width), dtype=np.int32)
        
        # Add ON events
        if len(on_events) > 0:
            frame_idx = ((on_events['t'] - t_start) / delta_t_us).astype(np.int64)
            frame_idx = np.clip(frame_idx, 0, n_frames - 1)
            flat_coords = (
                frame_idx * (height * width) +
                on_events['y'].astype(np.int64) * width +
                on_events['x'].astype(np.int64)
            )
            unique_coords, counts = np.unique(flat_coords, return_counts=True)
            frames.ravel()[unique_coords] += counts.astype(np.int32)
        
        # Subtract OFF events
        if len(off_events) > 0:
            frame_idx = ((off_events['t'] - t_start) / delta_t_us).astype(np.int64)
            frame_idx = np.clip(frame_idx, 0, n_frames - 1)
            flat_coords = (
                frame_idx * (height * width) +
                off_events['y'].astype(np.int64) * width +
                off_events['x'].astype(np.int64)
            )
            unique_coords, counts = np.unique(flat_coords, return_counts=True)
            frames.ravel()[unique_coords] -= counts.astype(np.int32)
        
        # Note: Clipping and scaling not applied to signed frames as they contain negative values
        return frames, timestamps
    else:
        # Regular modes (on, off, all)
        if mode == 'on':
            events = events[events['p'] == 1]
        elif mode == 'off':
            events = events[events['p'] == 0]
        
        if len(events) == 0:
            return np.array([]), np.array([])
        
        frame_idx = ((events['t'] - t_start) / delta_t_us).astype(np.int64)
        frame_idx = np.clip(frame_idx, 0, n_frames - 1)
        
        # Start with uint32 to avoid overflow during accumulation
        frames = np.zeros((n_frames, height, width), dtype=np.uint32)
        flat_coords = (
            frame_idx * (height * width) +
            events['y'].astype(np.int64) * width +
            events['x'].astype(np.int64)
        )
        
        unique_coords, counts = np.unique(flat_coords, return_counts=True)
        frames.ravel()[unique_coords] = counts
        
        # Apply scaling or clipping if requested
        if clip_value is not None:
            if scale_to_fit:
                # Scale linearly so maximum value equals clip_value
                # Preserves relative intensities but may reduce contrast
                max_val = frames.max()
                if max_val > clip_value:
                    frames = (frames.astype(np.float64) * clip_value / max_val).astype(np.uint16)
                else:
                    frames = frames.astype(np.uint16)
            else:
                # Hard clip: pixels exceeding clip_value are saturated at clip_value
                # This is the default behavior (clip_value=65535 for uint16 storage)
                frames = np.minimum(frames, clip_value).astype(np.uint16)
        else:
            # No clipping - keep as uint32 (not recommended for TIFF export)
            pass
        
        return frames, timestamps