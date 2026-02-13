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