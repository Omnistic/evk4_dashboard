"""
services/file_service.py

File loading and processing services.

Handles file selection, .raw to .npz conversion, data loading, and statistics computation.
"""

from pathlib import Path
from nicegui import ui, app
from typing import Dict, Any, List
import numpy as np
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor

from utils import raw_to_npz
from core import BIAS_NAMES
from core.validation import validate_positive_number

# Thread pool for async file operations
executor = ThreadPoolExecutor(max_workers=1)


async def pick_file() -> Path | None:
    """
    Open native file picker dialog.
    
    Allows user to select .raw or .npz event data files for loading.
    
    Returns:
        Path to selected file, or None if no valid file selected
    """
    result = await app.native.main_window.create_file_dialog(allow_multiple=False)
    if result and len(result) > 0:
        path = Path(result[0])
        if path.suffix.lower() in ('.raw', '.npz'):
            return path
        else:
            ui.notify('Please select a .raw or .npz file', type='negative')
    return None


async def convert_raw_file(path: Path, overwrite: bool) -> Path | None:
    """
    Convert .raw file to .npz format.
    
    Args:
        path: Path to .raw file
        overwrite: Whether to overwrite existing .npz file
    
    Returns:
        Path to .npz file, or None on failure
    """
    npz_path = path.with_suffix('.npz')
    
    if npz_path.exists() and not overwrite:
        ui.notify(f'{npz_path.name} already exists, loading existing file.', type='warning')
        return npz_path
    
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, raw_to_npz, path, overwrite)
        ui.notify(f'Saved: {npz_path}')
        return npz_path
    except FileNotFoundError:
        ui.notify(f'File not found: {path}', type='negative')
        return None
    except PermissionError:
        ui.notify(f'Permission denied: {path}', type='negative')
        return None
    except Exception as e:
        ui.notify(f'Failed to convert file: {str(e)}', type='negative')
        print(f'Conversion error: {e}')
        traceback.print_exc()
        return None


def load_npz_data(path: Path) -> Dict[str, Any] | None:
    """
    Load data from .npz file.
    
    Args:
        path: Path to .npz file
    
    Returns:
        Dictionary containing loaded data, or None on failure
    """
    try:
        return dict(np.load(path))
    except KeyError as e:
        ui.notify(f'Invalid file format: missing {str(e)}', type='negative')
        return None
    except ValueError as e:
        ui.notify(f'Invalid data in file: {str(e)}', type='negative')
        return None
    except Exception as e:
        ui.notify(f'Failed to load file: {str(e)}', type='negative')
        print(f'Load error: {e}')
        traceback.print_exc()
        return None


def extract_bias_data(data: Dict[str, Any]) -> tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Extract bias settings from loaded data.
    
    Args:
        data: Loaded data dictionary
    
    Returns:
        Tuple of (bias_columns, bias_row) for table display
    """
    bias_columns: List[Dict[str, str]] = []
    bias_row: Dict[str, int] = {}
    
    for name in BIAS_NAMES:
        if name in data:
            try:
                bias_columns.append({'name': name, 'label': name, 'field': name})
                bias_row[name] = int(data[name])
            except (ValueError, TypeError) as e:
                print(f'Warning: Could not parse bias {name}: {e}')
                continue
    
    return bias_columns, bias_row


def compute_statistics(events: np.ndarray) -> Dict[str, Any] | None:
    """
    Compute recording statistics from event data.
    
    Args:
        events: Event array
    
    Returns:
        Dictionary with 'duration', 'event_count', 'event_rate', or None on failure
    """
    try:
        duration = float((events['t'][-1] - events['t'][0]) / 1e6)
        
        if not validate_positive_number(duration, 'recording duration', min_value=0.0, exclusive_min=True):
            return None
        
        event_count = len(events)
        event_rate = event_count / duration
        on_count = int(np.sum(events['p'] == 1))
        off_count = int(np.sum(events['p'] == 0))
        
        return {
            'duration': duration,
            'event_count': event_count,
            'event_rate': event_rate,
            'on_count': on_count,
            'off_count': off_count,
        }
    except Exception as e:
        ui.notify(f'Failed to compute statistics: {str(e)}', type='negative')
        print(f'Statistics error: {e}')
        traceback.print_exc()
        return None


def shutdown_executor() -> None:
    """Shutdown the thread pool executor."""
    executor.shutdown(wait=False)