from nicegui import ui, app
from pathlib import Path
from utils import raw_to_npz, compute_event_histogram
import asyncio
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=1)

@ui.page('/')
def main_page():
    dark = ui.dark_mode(True)
    current_file = None
    current_data = None

    def toggle_dark():
        dark.toggle()
        icon.set_name('light_mode' if dark.value else 'dark_mode')

        if histogram_card.visible:
            fig = histogram_plot.figure
            fig.update_layout(template='plotly_dark' if dark.value else 'plotly')
            histogram_plot.update()

            fig = timetrace_plot.figure
            fig.update_layout(template='plotly_dark' if dark.value else 'plotly')
            timetrace_plot.update()

    def update_histogram():
        if current_data is None:
            return
        
        events = current_data['events']
        width, height = int(current_data['width']), int(current_data['height'])
        
        polarity = polarity_select.value
        if polarity == 'CD ON (polarity=1)':
            mode = 'on'
        elif polarity == 'CD OFF (polarity=0)':
            mode = 'off'
        else:
            mode = 'all'
        
        histogram = compute_event_histogram(events, width, height, mode)
        fig = go.Figure(go.Heatmap(
            z=histogram, 
            colorscale='Viridis',
            colorbar=dict(title='Count')
        ))
        fig.update_layout(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            yaxis=dict(scaleanchor='x', scaleratio=1),
            margin=dict(l=50, r=50, t=50, b=50),
            template='plotly_dark' if dark.value else 'plotly',
            dragmode='drawrect',
            newshape=dict(line=dict(color='cyan', width=2), fillcolor='rgba(0,255,255,0.2)'),
            modebar_add=['drawrect', 'eraseshape']
        )
        histogram_card.visible = True
        histogram_plot.figure = fig
        histogram_plot.update()

    def on_shape_drawn(e):
        if current_data is None or e.args is None:
            return
        
        args = e.args

        if 'shapes' not in args:
            return
        
        shapes = args['shapes']
        if not shapes or len(shapes) == 0:
            return
        
        shape = shapes[-1]
        
        x_min = int(shape['x0'])
        x_max = int(shape['x1'])
        y_min = int(shape['y0'])
        y_max = int(shape['y1'])

        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        plot_timetrace(x_min, x_max, y_min, y_max)

    def plot_timetrace(x_min, x_max, y_min, y_max):
        events = current_data['events']
        
        polarity = polarity_select.value
        if polarity == 'CD ON (polarity=1)':
            events = events[events['p'] == 1]
        elif polarity == 'CD OFF (polarity=0)':
            events = events[events['p'] == 0]
        
        mask = (
            (events['x'] >= x_min) & (events['x'] <= x_max) &
            (events['y'] >= y_min) & (events['y'] <= y_max)
        )
        selected_events = events[mask]
        
        if len(selected_events) == 0:
            ui.notify('No events in selection', type='warning')
            return
        
        max_points = 50000
        if len(selected_events) > max_points:
            indices = np.random.choice(len(selected_events), max_points, replace=False)
            indices.sort()
            selected_events = selected_events[indices]
            ui.notify(f'Downsampled to {max_points:,} points', type='info')
        
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
            xaxis_range=[times.min()-0.01*duration, times.max()+0.01*duration],
            yaxis=dict(visible=False, range=[-0.6, 0.6]),
            margin=dict(l=0, r=0, t=50, b=50),
            template='plotly_dark' if dark.value else 'plotly',
        )
        timetrace_plot.visible = True
        timetrace_plot.figure = fig
        timetrace_plot.update()

    async def pick_file():
        result = await app.native.main_window.create_file_dialog(
            allow_multiple=False
        )
        if result and len(result) > 0:
            path = Path(result[0])
            if path.suffix.lower() in ('.raw', '.npz'):
                await process_file(path)
            else:
                ui.notify('Please select a .raw or .npz file', type='negative')

    async def process_file(path):
        nonlocal current_file, current_data
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
        
        current_data = np.load(current_file)
        events = current_data['events']
        width, height = int(current_data['width']), int(current_data['height'])
        
        bias_names = ['bias_diff', 'bias_diff_off', 'bias_diff_on', 'bias_fo', 'bias_hpf', 'bias_refr']
        bias_columns = []
        bias_row = {}
        for name in bias_names:
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
        stats_table.visible = True

        update_histogram()

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

    with ui.row().classes('w-full items-center gap-4'):
        stats_table = ui.table(columns=[], rows=[], column_defaults={'align': 'center', 'headerClasses': 'uppercase text-primary'})
        stats_table.visible = False

        bias_table = ui.table(columns=[], rows=[], column_defaults={'align': 'center', 'headerClasses': 'uppercase text-primary'})
        bias_table.visible = False
    
    with ui.row().classes('w-full'):
        with ui.card().classes('flex-grow p-0') as histogram_card:
            with ui.row().classes('w-full items-center'):
                polarity_select = ui.select(
                    options=['BOTH', 'CD ON (polarity=1)', 'CD OFF (polarity=0)'],
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

            cd_on_badge.bind_visibility_from(timetrace_plot, 'visible')
            cd_off_badge.bind_visibility_from(timetrace_plot, 'visible')
        histogram_card.visible = False

    ui.separator().bind_visibility_from(histogram_card, 'visible')

app.on_startup(lambda: app.native.main_window.maximize())
ui.run(native=True)