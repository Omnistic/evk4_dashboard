"""
ui/layout.py

UI layout construction for the dashboard.

Builds all visual components including header, file controls, statistics tables,
plots, and frame generation controls.
"""

from nicegui import ui
from typing import NamedTuple
from core import POLARITY_OPTIONS, PLOT_CONFIG


class UIComponents(NamedTuple):
    """Container for all UI components that need to be accessed by callbacks."""
    # Controls
    icon: ui.icon
    overwrite_toggle: ui.switch
    file_label: ui.label
    polarity_select: ui.select
    roi_label: ui.label
    frame_polarity_select: ui.select
    delta_t_input: ui.number
    frames_input: ui.number
    
    # Buttons
    open_file_btn: ui.button
    generate_frames_btn: ui.button
    export_frames_btn: ui.button
    
    # Tables
    stats_table: ui.table
    bias_table: ui.table
    
    # Plots
    histogram_plot: ui.plotly
    timetrace_plot: ui.plotly
    spectrum_plot: ui.plotly
    iei_plot: ui.plotly
    frame_plot: ui.plotly
    
    # Frame controls
    frame_slider: ui.slider
    frame_index_label: ui.label
    frame_viewer: ui.column
    
    # Sections
    data_section: ui.column


def build_main_layout(dark, on_polarity_change=None) -> UIComponents:
    """
    Build the complete dashboard UI layout.
    
    Args:
        dark: Dark mode instance from NiceGUI
    
    Returns:
        UIComponents namedtuple containing references to all interactive components
    """
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    with ui.header().classes('justify-between items-center'):
        ui.label('EVK4 Dashboard').classes('text-xl font-bold')
        icon = ui.icon('light_mode', size='md').classes('cursor-pointer')
    
    # ========================================================================
    # FILE CONTROLS
    # ========================================================================
    
    with ui.row().classes('w-full items-center gap-4'):
        open_file_btn = ui.button('Open File', icon='folder_open').classes('w-64')
        overwrite_toggle = ui.switch('OVERWRITE')
        file_label = ui.label().classes('text-gray-400')
        file_label.visible = False
    
    ui.separator()
    
    # ========================================================================
    # DATA SECTION
    # ========================================================================
    
    with ui.column().classes('w-full gap-4') as data_section:
        
        # Recording Info
        ui.label('Recording Info').classes('text-lg font-bold')
        with ui.row().classes('w-full items-center gap-4'):
            stats_table = ui.table(
                columns=[], 
                rows=[], 
                column_defaults={'align': 'center', 'headerClasses': 'uppercase text-primary'}
            )
            bias_table = ui.table(
                columns=[], 
                rows=[], 
                column_defaults={'align': 'center', 'headerClasses': 'uppercase text-primary'}
            )
        
        ui.separator()
        
        # Event Visualization
        ui.label('Event Visualization').classes('text-lg font-bold')
        with ui.row().classes('w-full items-center'):
            polarity_select = ui.select(
                options=POLARITY_OPTIONS,
                value='BOTH',
                label='MODE',
                on_change=on_polarity_change
            ).classes('w-48')
            roi_label = ui.label('').classes('text-gray-400')
            ui.space()
            ui.badge('CD ON (polarity=1)').style(f'background-color: {PLOT_CONFIG.color_on} !important')
            ui.badge('CD OFF (polarity=0)').style(f'background-color: {PLOT_CONFIG.color_off} !important')
        
        with ui.row().classes('w-full flex-nowrap gap-0'):
            histogram_plot = ui.plotly({})
            timetrace_plot = ui.plotly({}).classes('flex-grow')
            timetrace_plot.visible = False
        
        with ui.row().classes('w-full flex-nowrap gap-0'):
            spectrum_plot = ui.plotly({}).classes('flex-grow')
            iei_plot = ui.plotly({})
        
        ui.separator()
        
        # Frame Generation
        ui.label('Frame Generation').classes('text-lg font-bold')
        with ui.row().classes('w-full items-center gap-4'):
            generate_frames_btn = ui.button('Generate Frames', icon='play_arrow')
            export_frames_btn = ui.button('Export TIFF', icon='save')
            delta_t_input = ui.number(
                label='ΔT (ms)', 
                value=33, 
                min=0.01, 
                step=1, 
                format='%.2f'
            ).classes('w-40')
            frames_input = ui.number(
                label='Frames', 
                value=100, 
                min=1, 
                step=1
            ).classes('w-40')
            frame_polarity_select = ui.select(
                options=POLARITY_OPTIONS, 
                value='BOTH', 
                label='MODE'
            ).classes('w-48')
        
        with ui.column().classes('w-full') as frame_viewer:
            frame_plot = ui.plotly({})
            with ui.row().classes('items-center gap-4').style('width: 25%'):
                frame_slider = ui.slider(min=0, max=99, value=0, step=1).classes('flex-grow')
                frame_index_label = ui.label('Frame 0 / 0')
        frame_viewer.visible = False
    
    data_section.visible = False
    
    # ========================================================================
    # RETURN COMPONENTS
    # ========================================================================
    
    return UIComponents(
        icon=icon,
        overwrite_toggle=overwrite_toggle,
        file_label=file_label,
        polarity_select=polarity_select,
        roi_label=roi_label,
        frame_polarity_select=frame_polarity_select,
        delta_t_input=delta_t_input,
        frames_input=frames_input,
        open_file_btn=open_file_btn,
        generate_frames_btn=generate_frames_btn,
        export_frames_btn=export_frames_btn,
        stats_table=stats_table,
        bias_table=bias_table,
        histogram_plot=histogram_plot,
        timetrace_plot=timetrace_plot,
        spectrum_plot=spectrum_plot,
        iei_plot=iei_plot,
        frame_plot=frame_plot,
        frame_slider=frame_slider,
        frame_index_label=frame_index_label,
        frame_viewer=frame_viewer,
        data_section=data_section,
    )