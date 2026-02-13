import numpy as np
import os, warnings
import plotly.graph_objects as go

from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

from metavision_sdk_stream import Camera, CameraStreamSlicer

load_dotenv()

def raw_to_npz(file_path, overwrite=False):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(path)
    
    if path.suffix.lower() != '.raw':
        raise ValueError(f'Expected .raw file, got: {path.suffix}')

    npz_path = path.with_suffix('.npz')
    if npz_path.exists() and not overwrite:
        warnings.warn(f'{npz_path} already exists, skipping (use overwrite=True to replace)')
        return

    camera = Camera.from_file(str(path))
    slicer = CameraStreamSlicer(camera.move())
    width = slicer.camera().width()
    height = slicer.camera().height()
    chunks = []
    for slice in tqdm(slicer, desc='Converting'):
        if slice.events.size > 0:
            chunks.append(slice.events.copy())
    all_events = np.concatenate(chunks)

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

    np.savez(npz_path, events=all_events, width=width, height=height, **biases)

def compute_event_histogram(events, width, height, mode='all'):
    width = int(width)
    height = int(height)
    
    if mode == 'signed':
        # Signed mode: ON pixels get positive counts, OFF pixels get negative
        on_events = events[events['p'] == 1]
        off_events = events[events['p'] == 0]
        
        histogram = np.zeros((height, width), dtype=np.int32)
        
        if len(on_events) > 0:
            coords, counts = np.unique(
                on_events['y'].astype(np.int64) * width + on_events['x'].astype(np.int64), 
                return_counts=True
            )
            histogram.flat[coords] += counts.astype(np.int32)
        
        if len(off_events) > 0:
            coords, counts = np.unique(
                off_events['y'].astype(np.int64) * width + off_events['x'].astype(np.int64), 
                return_counts=True
            )
            histogram.flat[coords] -= counts.astype(np.int32)
        
        return histogram
    
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

def display_event_histogram(histogram):
    fig = go.Figure(go.Heatmap(z=histogram))
    fig.show()

def generate_frames(events, width, height, delta_t_ms, mode='all'):
    width = int(width)
    height = int(height)
    
    if mode == 'on':
        events = events[events['p'] == 1]
    elif mode == 'off':
        events = events[events['p'] == 0]
    
    if len(events) == 0:
        return np.array([]), np.array([])
    
    delta_t_us = delta_t_ms * 1000
    t_start = events['t'][0]
    t_end = events['t'][-1]
    
    n_frames = int(np.ceil((t_end - t_start) / delta_t_us))
    timestamps = np.arange(n_frames) * delta_t_us + t_start
    
    frame_idx = ((events['t'] - t_start) / delta_t_us).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, n_frames - 1)
    
    frames = np.zeros((n_frames, height, width), dtype=np.uint16)
    flat_coords = (
        frame_idx * (height * width) +
        events['y'].astype(np.int64) * width +
        events['x'].astype(np.int64)
    )
    
    unique_coords, counts = np.unique(flat_coords, return_counts=True)
    frames.ravel()[unique_coords] = np.minimum(counts, 65535).astype(np.uint16)
    
    return frames, timestamps

if __name__ == '__main__':
    file_path = os.getenv('FILE_PATH')
    if not file_path:
        raise ValueError('FILE_PATH environment variable not set')
    raw_to_npz(file_path)

    npz_path = Path(file_path).with_suffix('.npz')
    data = np.load(npz_path)
    events, width, height = data['events'], data['width'], data['height']

    histogram = compute_event_histogram(events, width, height)
    display_event_histogram(histogram)