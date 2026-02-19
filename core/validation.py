"""
core/validation.py

Input validation functions for data integrity checks.

Provides validation for numerical inputs, array dimensions, ROI bounds,
and other data constraints. All functions return bool and display user
notifications on validation failure.
"""


def validate_positive_number(value, name, min_value=0.0, exclusive_min=True):
    if value is None:
        print(f'[WARN] {name} is required')
        return False
    if exclusive_min and value <= min_value:
        print(f'[WARN] {name} must be greater than {min_value}')
        return False
    if not exclusive_min and value < min_value:
        print(f'[WARN] {name} must be at least {min_value}')
        return False
    return True

def validate_dimensions(width, height):
    if width <= 0 or height <= 0:
        print(f'[WARN] Invalid sensor dimensions: {width}x{height}')
        return False
    return True

def validate_events_not_empty(events, context='operation'):
    if len(events) == 0:
        print(f'[WARN] No events available for {context}')
        return False
    return True

def validate_roi_bounds(roi, width, height):
    x_min, x_max, y_min, y_max = roi
    if x_min < 0 or x_max >= width or y_min < 0 or y_max >= height:
        print(
            f'[WARN] ROI ({x_min},{y_min})-({x_max},{y_max}) is outside sensor bounds '
            f'(0,0)-({width-1},{height-1})'
        )
        return False
    if x_min >= x_max or y_min >= y_max:
        print('[WARN] ROI has zero or negative area')
        return False
    return True

def validate_array_length(arr, min_length, name):
    if len(arr) < min_length:
        print(f'[WARN] Not enough data in {name}: need at least {min_length}, got {len(arr)}')
        return False
    return True
