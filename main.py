from nicegui import ui, app
from pathlib import Path
from utils import raw_to_npz, compute_event_histogram, generate_frames
from concurrent.futures import ThreadPoolExecutor
import asyncio
import imageio
import numpy as np
import os
import plotly.graph_objects as go

executor = ThreadPoolExecutor(max_workers=1)

POLARITY_OPTIONS = ['BOTH', 'CD ON (polarity=1)', 'CD OFF (polarity=0)']
BIAS_NAMES = ['bias_diff', 'bias_diff_off', 'bias_diff_on', 'bias_fo', 'bias_hpf', 'bias_refr']
MAX_TIMETRACE_POINTS = 50000
MAX_DISPLAY_FRAMES = 200


@ui.page('/')
def main_page():
    dark = ui.dark_mode(True)
    current_file = None
    current_data = None
    recording_duration_ms = 0
    generated_frames = None
    generated_timestamps = None
    updating = False

    # === HELPER FUNCTIONS ===

    def get_polarity_mode():
        polarity = polarity_select.value
        if polarity == 'CD ON (polarity=1)':
            return 'on'
        elif polarity == 'CD OFF (polarity=0)':
            return 'off'
        return 'all'

    def get_frame_polarity_mode():
        polarity = frame_polarity_select.value
        if polarity == 'CD ON (polarity=1)':
            return 'on'
        elif polarity == 'CD OFF (polarity=0)':
            return 'off'
        return 'all'

    # === UI CALLBACKS ===

    def toggle_dark():
        dark.toggle()
        icon.set_name('light_mode' if dark.value else 'dark_mode')
        template = 'plotly_dark' if dark.value else 'plotly'

        if current_data is not None:
            histogram_plot.figure.update_layout(template=template)
            histogram_plot.update()
            
            if timetrace_plot.visible:
                timetrace_plot.figure.update_layout(template=template)
                timetrace_plot.update()
            
            if iei_plot.figure and hasattr(iei_plot.figure, 'update_layout'):
                iei_plot.figure.update_layout(template=template)
                iei_plot.update()
                spectrum_plot.figure.update_layout(template=template)
                spectrum_plot.update()

        if generated_frames is not None:
            frame_plot.figure.update_layout(template=template)
            frame_plot.update()

    def update_histogram():
        if current_data is None:
            return
        
        events = current_data['events']
        width, height = int(current_data['width']), int(current_data['height'])
        mode = get_polarity_mode()
        histogram = compute_event_histogram(events, width, height, mode)
        
        fig = go.Figure(go.Heatmap(
            z=histogram,
            colorscale='Viridis',
            colorbar=dict(title='Count')
        ))
        fig.update_layout(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            yaxis=dict(scaleanchor='x', scaleratio=1, autorange='reversed'),
            margin=dict(l=50, r=50, t=50, b=50),
            template='plotly_dark' if dark.value else 'plotly',
            dragmode='drawrect',
            newshape=dict(line=dict(color='cyan', width=2), fillcolor='rgba(0,255,255,0.2)'),
            modebar_add=['drawrect', 'eraseshape']
        )
        histogram_plot.figure = fig
        histogram_plot.update()
    
        update_iei_histogram()
        update_power_spectrum()

    def update_iei_histogram():
        if current_data is None:
            return
        
        events = current_data['events']
        mode = get_polarity_mode()
        if mode == 'on':
            events = events[events['p'] == 1]
        elif mode == 'off':
            events = events[events['p'] == 0]
        
        if len(events) < 2:
            return
        
        iei = np.diff(events['t'])
        iei_ms = iei / 1000
        iei_ms = iei_ms[iei_ms > 0]
        
        if len(iei_ms) > 50000:
            indices = np.random.choice(len(iei_ms), 50000, replace=False)
            iei_ms = iei_ms[indices]
        
        iei_min, iei_max = iei_ms.min(), iei_ms.max()
        
        fig = go.Figure(go.Histogram(
            x=iei_ms.tolist(),
            nbinsx=100,
            marker_color='#CC79A7',
            marker_line=dict(color='white', width=0.5)
        ))
        fig.update_layout(
            xaxis_title='Inter-event interval (ms)',
            yaxis_title='Count',
            yaxis_type='log',
            xaxis=dict(showgrid=True, dtick=1),
            yaxis=dict(showgrid=True),
            margin=dict(l=50, r=50, t=50, b=50),
            template='plotly_dark' if dark.value else 'plotly',
            xaxis2=dict(
                title='Frequency (Hz)',
                overlaying='x',
                side='top',
                range=[1000 / iei_min, 1000 / iei_max],
                showgrid=False,
            ),
        )
        fig.add_trace(go.Scatter(x=[], y=[], xaxis='x2'))
        
        iei_plot.figure = fig
        iei_plot.update()

    def update_power_spectrum():
        if current_data is None:
            return
        
        events = current_data['events']
        mode = get_polarity_mode()
        if mode == 'on':
            events = events[events['p'] == 1]
        elif mode == 'off':
            events = events[events['p'] == 0]
        
        times_us = events['t']
        
        bin_width_us = 100
        bins = np.arange(times_us.min(), times_us.max() + bin_width_us, bin_width_us)
        counts, _ = np.histogram(times_us, bins=bins)
        
        fft = np.fft.rfft(counts - counts.mean())
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(counts), d=bin_width_us / 1e6)
        
        mask = (freqs >= 0.1) & (freqs <= 2000)
        freqs = freqs[mask]
        power = power[mask]
        
        fig = go.Figure(go.Scatter(
            x=freqs,
            y=power,
            mode='lines',
            line=dict(color='#CC79A7', width=1)
        ))
        fig.update_layout(
            xaxis_title='Frequency (Hz)',
            yaxis_title='Power',
            yaxis_type='log',
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            margin=dict(l=50, r=50, t=50, b=50),
            template='plotly_dark' if dark.value else 'plotly',
        )
        spectrum_plot.figure = fig
        spectrum_plot.update()

    def on_shape_drawn(e):
        if e.args is None or current_data is None:
            return
        
        args = e.args
        if 'shapes' not in args:
            return
        
        shapes = args['shapes']
        if not shapes or len(shapes) == 0:
            return
        
        shape = shapes[-1]
        x_min, x_max = int(shape['x0']), int(shape['x1'])
        y_min, y_max = int(shape['y0']), int(shape['y1'])

        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        plot_timetrace(x_min, x_max, y_min, y_max)

    def plot_timetrace(x_min, x_max, y_min, y_max):
        events = current_data['events']
        
        mode = get_polarity_mode()
        if mode == 'on':
            events = events[events['p'] == 1]
        elif mode == 'off':
            events = events[events['p'] == 0]
        
        mask = (
            (events['x'] >= x_min) & (events['x'] <= x_max) &
            (events['y'] >= y_min) & (events['y'] <= y_max)
        )
        selected_events = events[mask]
        
        if len(selected_events) == 0:
            ui.notify('No events in selection', type='warning')
            return
        
        if len(selected_events) > MAX_TIMETRACE_POINTS:
            indices = np.random.choice(len(selected_events), MAX_TIMETRACE_POINTS, replace=False)
            indices.sort()
            selected_events = selected_events[indices]
            ui.notify(f'Downsampled to {MAX_TIMETRACE_POINTS:,} points', type='info')
        
        times = selected_events['t'] / 1e6
        duration = times.max() - times.min()
        polarities = selected_events['p']
        colors = np.where(polarities == 1, '#E69F00', '#56B4E9')
        jitter = np.random.uniform(-0.5, 0.5, len(times))

        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=times,
            y=jitter,
            mode='markers',
            marker=dict(size=3, color=colors),
        ))
        fig.update_layout(
            xaxis_title='Time (s)',
            xaxis_range=[times.min() - 0.01 * duration, times.max() + 0.01 * duration],
            yaxis=dict(visible=False, range=[-0.6, 0.6]),
            margin=dict(l=0, r=0, t=50, b=50),
            template='plotly_dark' if dark.value else 'plotly',
        )
        timetrace_plot.visible = True
        timetrace_plot.figure = fig
        timetrace_plot.update()

    def on_delta_t_change():
        nonlocal updating
        if updating or recording_duration_ms == 0 or delta_t_input.value is None:
            return
        updating = True
        delta_t = max(0.01, delta_t_input.value)
        frames = int(recording_duration_ms / delta_t)
        frames_input.value = max(1, frames)
        updating = False

    def on_frames_change():
        nonlocal updating
        if updating or recording_duration_ms == 0 or frames_input.value is None:
            return
        updating = True
        frames = max(1, int(frames_input.value))
        delta_t = recording_duration_ms / frames
        delta_t_input.value = round(delta_t, 2)
        updating = False

    # === FILE HANDLING ===

    async def pick_file():
        result = await app.native.main_window.create_file_dialog(allow_multiple=False)
        if result and len(result) > 0:
            path = Path(result[0])
            if path.suffix.lower() in ('.raw', '.npz'):
                await process_file(path)
            else:
                ui.notify('Please select a .raw or .npz file', type='negative')

    async def process_file(path):
        nonlocal current_file, current_data, recording_duration_ms
        suffix = path.suffix.lower()
        
        if suffix == '.raw':
            npz_path = path.with_suffix('.npz')
            if npz_path.exists() and not overwrite_toggle.value:
                ui.notify(f'{npz_path.name} already exists, loading existing file.', type='warning')
            else:
                overlay = ui.dialog().props('persistent')
                with overlay, ui.card().classes('items-center p-8'):
                    ui.spinner(size='xl')
                    ui.label(f'Converting {path.name}...').classes('mt-4')
                overlay.open()
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(executor, raw_to_npz, path, overwrite_toggle.value)
                
                overlay.close()
                ui.notify(f'Saved: {npz_path}')
            current_file = npz_path
        elif suffix == '.npz':
            current_file = path
            ui.notify(f'Loaded: {path.name}')
        
        file_label.text = f'Loaded: {current_file.name}'
        file_label.visible = True
        
        try:
            current_data = np.load(current_file)
            events = current_data['events']
            width, height = int(current_data['width']), int(current_data['height'])
        except Exception as e:
            ui.notify(f'Failed to load file: {e}', type='negative')
            file_label.visible = False
            current_file = None
            current_data = None
            return
        
        bias_columns = []
        bias_row = {}
        for name in BIAS_NAMES:
            if name in current_data:
                bias_columns.append({'name': name, 'label': name, 'field': name})
                bias_row[name] = int(current_data[name])
        
        if bias_columns:
            bias_table.columns = bias_columns
            bias_table.rows = [bias_row]
            bias_table.visible = True
        else:
            bias_table.visible = False

        duration = (events['t'][-1] - events['t'][0]) / 1e6
        event_count = len(events)
        event_rate = event_count / duration if duration > 0 else 0
        
        stats_table.columns = [
            {'name': 'events', 'label': 'Events', 'field': 'events'},
            {'name': 'duration', 'label': 'Duration', 'field': 'duration'},
            {'name': 'rate', 'label': 'Event rate', 'field': 'rate'},
            {'name': 'resolution', 'label': 'Resolution', 'field': 'resolution'},
        ]
        stats_table.rows = [{
            'events': f'{event_count:,}',
            'duration': f'{duration:.2f} s',
            'rate': f'{event_rate:,.0f} ev/s',
            'resolution': f'{width} x {height}',
        }]

        recording_duration_ms = duration * 1000
        frames_input.value = 100
        on_frames_change()

        update_histogram()
        data_section.visible = True

    # === UI LAYOUT ===

    with ui.header().classes('justify-between items-center'):
        ui.label('EVK4 Dashboard').classes('text-xl font-bold')
        icon = ui.icon('light_mode', size='md').classes('cursor-pointer')
        icon.on('click', toggle_dark)

    with ui.row().classes('w-full items-center gap-4'):
        ui.button('Open File', on_click=pick_file).classes('w-64')
        overwrite_toggle = ui.switch('OVERWRITE')
        file_label = ui.label().classes('text-gray-400')
        file_label.visible = False

    ui.separator()

    with ui.column().classes('w-full gap-4') as data_section:
        ui.label('Recording Info').classes('text-lg font-bold')
        with ui.row().classes('w-full items-center gap-4'):
            stats_table = ui.table(columns=[], rows=[], column_defaults={'align': 'center', 'headerClasses': 'uppercase text-primary'})
            bias_table = ui.table(columns=[], rows=[], column_defaults={'align': 'center', 'headerClasses': 'uppercase text-primary'})

        ui.separator()

        ui.label('Event Visualization').classes('text-lg font-bold')
        with ui.row().classes('w-full items-center'):
            polarity_select = ui.select(
                options=POLARITY_OPTIONS,
                value='BOTH',
                label='MODE',
                on_change=lambda: update_histogram()
            ).classes('w-48')
            ui.space()
            cd_on_badge = ui.badge('CD ON (polarity=1)').style('background-color: #E69F00 !important')
            cd_off_badge = ui.badge('CD OFF (polarity=0)').style('background-color: #56B4E9 !important')
        with ui.row().classes('w-full flex-nowrap gap-0'):
            histogram_plot = ui.plotly({})
            histogram_plot.on('plotly_relayout', on_shape_drawn)
            timetrace_plot = ui.plotly({}).classes('flex-grow')
            timetrace_plot.visible = False
        with ui.row().classes('w-full flex-nowrap gap-0'):
            spectrum_plot = ui.plotly({}).classes('flex-grow')
            iei_plot = ui.plotly({})
        cd_on_badge.bind_visibility_from(timetrace_plot, 'visible')
        cd_off_badge.bind_visibility_from(timetrace_plot, 'visible')

        ui.separator()

        ui.label('Frame Generation').classes('text-lg font-bold')
        with ui.row().classes('w-full items-center gap-4'):
            generate_button = ui.button('Generate Frames')
            export_button = ui.button('Export TIFF')
            delta_t_input = ui.number(label='ΔT (ms)', value=33, min=0.01, step=1, format='%.2f').classes('w-40')
            frames_input = ui.number(label='Frames', value=100, min=1, step=1).classes('w-40')
            frame_polarity_select = ui.select(options=POLARITY_OPTIONS, value='BOTH', label='MODE').classes('w-48')
        delta_t_input.on('update:model-value', on_delta_t_change)
        frames_input.on('update:model-value', on_frames_change)

        with ui.column().classes('w-full') as frame_viewer:
            frame_plot = ui.plotly({})
            with ui.row().classes('items-center gap-4').style('width: 25%'):
                frame_slider = ui.slider(min=0, max=99, value=0, step=1).classes('flex-grow')
                frame_index_label = ui.label('Frame 0 / 0')
        frame_viewer.visible = False

    data_section.visible = False

    # === FRAME GENERATION CALLBACKS ===

    def update_frame_display():
        nonlocal generated_frames
        if generated_frames is None or len(generated_frames) == 0:
            return
        idx = int(frame_slider.value)
        frame_plot.figure.data[0].z = generated_frames[idx]
        frame_plot.update()
        frame_index_label.text = f'Frame {idx + 1} / {len(generated_frames)}'

    async def generate_frames_callback():
        nonlocal generated_frames, generated_timestamps
        if current_data is None:
            ui.notify('No data loaded', type='warning')
            return
        
        events = current_data['events']
        width, height = int(current_data['width']), int(current_data['height'])
        delta_t = delta_t_input.value
        mode = get_frame_polarity_mode()
        
        overlay = ui.dialog().props('persistent')
        with overlay, ui.card().classes('items-center p-8'):
            ui.spinner(size='xl')
            ui.label('Generating frames...').classes('mt-4')
        overlay.open()
        await asyncio.sleep(0.1)
        
        loop = asyncio.get_event_loop()
        frames, timestamps = await loop.run_in_executor(
            executor, generate_frames, events, width, height, delta_t, mode
        )
        
        overlay.close()
        
        if len(frames) == 0:
            ui.notify('No frames generated', type='warning')
            return
        
        n_frames = len(frames)
        original_n_frames = n_frames
        
        if n_frames > MAX_DISPLAY_FRAMES:
            indices = np.linspace(0, n_frames - 1, MAX_DISPLAY_FRAMES, dtype=int)
            frames = frames[indices]
            timestamps = timestamps[indices]
            n_frames = MAX_DISPLAY_FRAMES
            ui.notify(f'Downsampled from {original_n_frames} to {MAX_DISPLAY_FRAMES} frames for display', type='info')
        
        generated_frames = frames
        generated_timestamps = timestamps
        
        zmin = 0
        zmax = max(1, np.percentile(frames[frames > 0], 99)) if np.any(frames > 0) else 1
        
        fig = go.Figure(go.Heatmap(
            z=frames[0],
            colorscale='Viridis',
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title='Count')
        ))
        fig.update_layout(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            yaxis=dict(scaleanchor='x', scaleratio=1, autorange='reversed'),
            margin=dict(l=50, r=50, t=50, b=50),
            template='plotly_dark' if dark.value else 'plotly',
        )
        
        frame_slider._props['max'] = n_frames - 1
        frame_slider.value = 0
        frame_slider.update()
        
        frame_plot.figure = fig
        frame_viewer.visible = True
        frame_plot.update()
        frame_index_label.text = f'Frame 1 / {n_frames}'
        
        ui.notify(f'Generated {original_n_frames} frames')

    async def export_frames_callback():
        if current_data is None:
            ui.notify('No data loaded', type='warning')
            return
        
        events = current_data['events']
        width, height = int(current_data['width']), int(current_data['height'])
        delta_t = delta_t_input.value
        mode = get_frame_polarity_mode()
        
        overlay = ui.dialog().props('persistent')
        with overlay, ui.card().classes('items-center p-8'):
            ui.spinner(size='xl')
            ui.label('Exporting frames...').classes('mt-4')
        overlay.open()
        await asyncio.sleep(0.1)
        
        loop = asyncio.get_event_loop()
        frames, _ = await loop.run_in_executor(
            executor, generate_frames, events, width, height, delta_t, mode
        )
        
        overlay.close()
        
        if len(frames) == 0:
            ui.notify('No frames to export', type='warning')
            return
        
        delta_t_us = int(delta_t * 1000)
        output_path = current_file.with_name(f'{current_file.stem}_frames_{delta_t_us}us.tif')
        imageio.mimwrite(str(output_path), frames.astype(np.uint16))
        ui.notify(f'Exported {len(frames)} frames to {output_path.name}')

    frame_slider.on('update:model-value', update_frame_display, throttle=0.1)
    generate_button.on('click', generate_frames_callback)
    export_button.on('click', export_frames_callback)


def shutdown():
    print('Shutting down...')
    executor.shutdown(wait=False)
    os._exit(0)


app.on_startup(lambda: app.native.main_window.maximize())
app.on_shutdown(shutdown)
ui.run(native=True, reconnect_timeout=120)