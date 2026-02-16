"""
core/validation.py

Input validation functions for data integrity checks.

Provides validation for numerical inputs, array dimensions, ROI bounds,
and other data constraints. All functions return bool and display user
notifications on validation failure.
"""

from nicegui import ui
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple


def validate_positive_number(
    value: Optional[float], 
    name: str, 
    min_value: float = 0.0,
    exclusive_min: bool = True
) -> bool:
    """
    Validate that a number is positive and within bounds.
    
    Args:
        value: The value to validate
        name: Name of the parameter (for error messages)
        min_value: Minimum allowed value
        exclusive_min: If True, value must be > min_value; if False, >= min_value
    
    Returns:
        True if valid, False otherwise (displays notification on failure)
    """
    if value is None:
        ui.notify(f'{name} is required', type='negative')
        return False
    
    if exclusive_min and value <= min_value:
        ui.notify(f'{name} must be greater than {min_value}', type='negative')
        return False
    
    if not exclusive_min and value < min_value:
        ui.notify(f'{name} must be at least {min_value}', type='negative')
        return False
    
    return True


def validate_dimensions(width: int, height: int) -> bool:
    """
    Validate sensor dimensions are positive.
    
    Args:
        width: Sensor width in pixels
        height: Sensor height in pixels
    
    Returns:
        True if valid, False otherwise (displays notification on failure)
    """
    if width <= 0 or height <= 0:
        ui.notify(f'Invalid sensor dimensions: {width}x{height}', type='negative')
        return False
    return True


def validate_events_not_empty(events: npt.NDArray[np.void], context: str = 'operation') -> bool:
    """
    Validate that event array is not empty.
    
    Args:
        events: Event array to validate
        context: Context description for error message
    
    Returns:
        True if valid, False otherwise (displays notification on failure)
    """
    if len(events) == 0:
        ui.notify(f'No events available for {context}', type='warning')
        return False
    return True


def validate_roi_bounds(
    roi: Tuple[int, int, int, int],
    width: int,
    height: int
) -> bool:
    """
    Validate ROI bounds are within sensor dimensions.
    
    Args:
        roi: Tuple of (x_min, x_max, y_min, y_max)
        width: Sensor width in pixels
        height: Sensor height in pixels
    
    Returns:
        True if valid, False otherwise (displays notification on failure)
    """
    x_min, x_max, y_min, y_max = roi
    
    if x_min < 0 or x_max >= width or y_min < 0 or y_max >= height:
        ui.notify(
            f'ROI ({x_min},{y_min})-({x_max},{y_max}) is outside sensor bounds '
            f'(0,0)-({width-1},{height-1})',
            type='warning'
        )
        return False
    
    if x_min >= x_max or y_min >= y_max:
        ui.notify('ROI has zero or negative area', type='warning')
        return False
    
    return True


def validate_array_length(
    arr: npt.NDArray,
    min_length: int,
    name: str
) -> bool:
    """
    Validate array has minimum required length.
    
    Args:
        arr: Array to validate
        min_length: Minimum required length
        name: Name of the array (for error messages)
    
    Returns:
        True if valid, False otherwise (displays notification on failure)
    """
    if len(arr) < min_length:
        ui.notify(
            f'Not enough data in {name}: need at least {min_length}, got {len(arr)}',
            type='warning'
        )
        return False
    return True
