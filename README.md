# EVK4 Event Camera Dashboard

> A modern, modular NiceGUI application for visualizing and analyzing event-based camera data from Prophesee EVK4 sensors.

▶️ **[Watch the demo on YouTube](https://youtu.be/XaYHxhMus8U)**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![NiceGUI](https://img.shields.io/badge/NiceGUI-1.0+-green.svg)
![Plotly](https://img.shields.io/badge/Plotly-interactive-3F4F75.svg)
![Pixi](https://img.shields.io/badge/Pixi-managed-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🌟 Features

- **Interactive Event Visualization** - Real-time 2D histograms with multiple polarity modes
- **ROI Analysis** - Draw regions of interest directly on visualizations
- **Temporal Analysis** - Inter-event interval and power spectrum analysis
- **Frame Generation** - Convert event streams to frame sequences with adjustable time windows
- **Export Capabilities** - Save generated frames as multi-page TIFF files
- **Dark/Light Themes** - Toggle between display modes for comfortable viewing

---

## 🚀 Quick Start

### Installation

This project uses [Pixi](https://pixi.sh/) for dependency management.

#### Prerequisites

**1. Install Pixi** (if you haven't already):
```powershell
# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex

# macOS/Linux
curl -fsSL https://pixi.sh/install.sh | bash
```

**2. Install Prophesee Metavision SDK:**
The Metavision SDK must be installed system-wide before using this dashboard.
- **Windows**: [Metavision SDK Installation Guide](https://docs.prophesee.ai/stable/installation/windows.html)
- **Linux**: [Metavision SDK Installation Guide](https://docs.prophesee.ai/stable/installation/linux.html)

---

#### 📥 Setup (The Easy Way - Windows)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Omnistic/evk4_dashboard.git
   cd evk4_dashboard
   ```
2. **Configure Path:** Open `pixi.toml` and ensure the `PYTHONPATH` points to your Prophesee installation.
   *Windows default: `C:\\Program Files\\Prophesee\\lib\\python3\\site-packages`*
3. **Launch:** Double-click **`launch_dashboard.bat`**. 
   *This script automatically verifies your Pixi installation, sets up the environment, and starts the dashboard.*

---

#### 🛠 Manual Setup (All Platforms)

If you prefer the command line or are on Linux:

```bash
# Clone the repository
git clone https://github.com/Omnistic/evk4_dashboard.git
cd evk4_dashboard

# Install dependencies and set up environment
pixi install

# Run the dashboard
pixi run start
```

### 🏃 Running the Application

| Method | Command / Action | Recommended For |
| :--- | :--- | :--- |
| **One-Click (Win)** | Double-click `launch_dashboard.bat` | Everyday use on Windows |
| **Pixi CLI** | `pixi run start` | Development and Linux/macOS |
| **Python CLI** | `python app.py` | Manual setups without Pixi |

The dashboard will open in a native window, maximized and ready to use.

---

## 📦 Sample Datasets

New to event cameras? Prophesee provides sample datasets to get you started:

**Download Sample Data:**
- [Prophesee Sample Datasets](https://docs.prophesee.ai/stable/datasets.html#chapter-datasets)
- Look for `.raw` files (native event camera format)
- The dashboard will automatically convert `.raw` to `.npz` on first load

**Recommended Dataset:**

We recommend starting with **`monitoring_40_50hz`** - a great dataset for exploring the dashboard's capabilities:

- **Description**: Object vibration at 40Hz and 50Hz frequencies
- **Camera**: Gen3.0
- **Duration**: 6 seconds
- **Size**: RAW EVT2 (157 MB)
- **Perfect for**: Viewing vibration frequencies in the power spectrum analysis

This dataset showcases:
- ✅ Clear temporal patterns in the time trace
- ✅ Distinct frequency peaks (40Hz and 50Hz) in the power spectrum
- ✅ Dynamic event activity for histogram visualization
- ✅ Inter-event interval analysis

> **⚠️ Important:** When analyzing frequencies, use **SIGNED (ON - OFF)** mode in the polarity selector! 
> 
> In BOTH mode, you might see peaks at **80Hz and 100Hz** (double the actual frequencies) because both ON and OFF events are triggered at each vibration cycle. SIGNED mode subtracts OFF from ON events, revealing the true **40Hz and 50Hz** frequencies.

**Usage:**
1. Download the `.raw` file from [Prophesee datasets](https://docs.prophesee.ai/stable/datasets.html#chapter-datasets)
2. Click "Open File" in the dashboard
3. Select the downloaded `.raw` file
4. Dashboard automatically converts to `.npz` and loads the data
5. **Set polarity mode to "SIGNED (ON - OFF)"**
6. Check the **Power Spectrum** plot to see the true 40Hz and 50Hz peaks!

> **Tip:** The first time you load a `.raw` file, conversion to `.npz` may take a moment. Subsequent loads of the same file will be much faster using the `.npz` version.

---

## 📁 Project Structure

```
evk4_dashboard/
├── 📄 launch_dashboard.bat      # Windows one-click launcher
├── 📄 app.py                    # Application entry point
├── 🔧 utils.py                  # Event data processing utilities
├── 📄 pixi.toml                 # Dependency & environment configuration
│
├── 📦 core/                     # Core application logic
│   ├── __init__.py
│   ├── config.py                # User preference persistence
│   ├── state.py                 # State management & configuration
│   ├── constants.py             # Application constants
│   └── validation.py            # Input validation functions
│
├── 💼 services/                 # Business logic services
│   ├── __init__.py
│   └── file_service.py          # File I/O operations
│
└── 🎨 ui/                       # User interface
    ├── __init__.py
    ├── layout.py                # UI component layout
    ├── callbacks.py             # Event handlers
    └── plots.py                 # Plot update functions
```

### Module Overview

| Module | Purpose | Lines |
|--------|---------|-------|
| **app.py** | Application entry point | 106 |
| **core/** | State, constants, validation | 404 |
| **services/** | File handling & data loading | 185 |
| **ui/** | Layout, callbacks, plots | 1,217 |
| **utils.py** | Event data utilities | 512 |

---

## 📖 Usage Guide

### Loading Data

1. Click **"Open File"** and select a `.raw` or `.npz` event data file.
2. `.raw` files are automatically converted to `.npz` on first load (toggle **OVERWRITE** to re-convert).
3. On load, the dashboard displays a **statistics table** (event count, ON/OFF split, duration, event rate, resolution) and a **bias settings table** if a bias file is present in the same folder.

### Event Visualization

The 2D histogram shows spatial event density across the sensor.
**Polarity modes** (selector above the histogram):
- **BOTH** — all events combined
- **CD ON (polarity=1)** — positive-change events only
- **CD OFF (polarity=0)** — negative-change events only
- **SIGNED (ON - OFF)** — signed difference, highlighting net event activity

**Colorscale controls** — Set **Color min / Color max** and press **APPLY** to clamp the histogram colorscale. Leave blank for auto-scaling. Useful when bright hotspots wash out low-activity pixels.

**Region of Interest (ROI):**
1. Click the rectangle draw tool on the histogram toolbar (active by default).
2. Draw a region — all analysis plots update automatically for the last drawn area.
3. Click the border of an ROI and press **Erase active shape** on the histogram toolbar to remove the selected ROI.

### Time Range Filtering

Use the **Time Range** slider (or the **From / To** number inputs) to restrict analysis to a portion of the recording. Click **Apply** to update all plots; **Reset** to return to the full duration.

### Analysis Plots

- **Time Trace** — individual events plotted over time; ON events in orange, OFF events in blue. Downsampled to 10,000 points if the dataset is large.
- **IEI Histogram** — distribution of inter-event intervals on a log-count axis, with a frequency axis (Hz) on top.
- **Power Spectrum** — FFT-based power spectrum of the event rate, useful for identifying periodic activity.

### Generating Frames

1. Set **ΔT (ms)** — temporal bin size per frame. Adjusting this also updates the **Frames** counter automatically (and vice versa).
2. Select **MODE** — polarity mode for frame accumulation.
3. Set **Color min / Color max** (optional) to clamp frame colorscale.
4. Click **"Generate Frames"** to preview frames in the viewer. Use the slider to scrub through frames.
5. Click **"Export TIFF"** to save all frames as a multi-page TIFF alongside the source file.
> Frames are downsampled to 1,000 for display if the total exceeds that limit — the TIFF export always saves all frames at full resolution.

---

## 🏗️ Architecture

### Design Principles

✅ **Separation of Concerns** - Each module has a single, well-defined responsibility  
✅ **Testability** - Components can be tested independently  
✅ **Maintainability** - Clear structure makes updates straightforward  
✅ **Scalability** - Easy to add new features without touching existing code

### Key Patterns

**State Management**
```python
from core import AppState
state = AppState()  # Centralized application state
```

**Component Communication**
```python
# UI components returned as a NamedTuple
components = build_main_layout(dark, on_polarity_change)
components.histogram_plot.update()
```

**Callback Factories**
```python
# Callbacks created with closures over state
toggle_dark = create_toggle_dark_callback(state, dark, components)
```

---

## 🔧 Extending the Dashboard

### Adding a New Plot

**1. Create the plot update function** (`ui/plots.py`)
```python
def update_my_plot(state, dark_mode, polarity_mode, plot_component):
    """Update my custom plot."""
    # Your plotting logic here
    fig = go.Figure(...)
    plot_component.figure = fig
    plot_component.update()
```

**2. Add UI component** (`ui/layout.py`)
```python
my_plot = ui.plotly({})
# Add to UIComponents return value
```

**3. Wire up the callback** (`app.py`)
```python
# Call in polarity change callback
update_my_plot(state, dark.value, polarity_mode, components.my_plot)
```

### Adding a New Constant

**Add to** `core/constants.py`:
```python
MY_NEW_PARAMETER: int = 42
```

**Import where needed:**
```python
from core import MY_NEW_PARAMETER
```

### Adding Validation

**Add to** `core/validation.py`:
```python
def validate_my_input(value: float, name: str) -> bool:
    """Validate my custom input."""
    if value < 0:
        ui.notify(f'{name} must be positive', type='negative')
        return False
    return True
```

---

## 🧪 Testing

### Unit Testing Example

```python
# Test validation logic
from core.validation import validate_roi_bounds

def test_roi_validation():
    # Valid ROI
    assert validate_roi_bounds((0, 100, 0, 100), 640, 480) == True
    
    # ROI outside sensor bounds
    assert validate_roi_bounds((0, 700, 0, 100), 640, 480) == False
    
    # Negative area ROI
    assert validate_roi_bounds((100, 50, 0, 100), 640, 480) == False
```

### Integration Testing

```python
# Test file loading
from services.file_service import load_npz_data
from pathlib import Path

def test_file_loading():
    data = load_npz_data(Path("test_data.npz"))
    assert data is not None
    assert 'events' in data
    assert 'width' in data
    assert 'height' in data
```

---

## 🐛 Troubleshooting

### Common Issues

**File conversion fails**
- Check that Metavision SDK is properly installed
- Verify the `.raw` file is not corrupted

**Plots not displaying**
- Check console for error messages
- Ensure data file contains valid event arrays

**Frame export fails**
- Verify write permissions for output directory
- Check available disk space

---

## 📊 Performance Notes

- **Large datasets** (>10M events) are automatically downsampled for visualization
- **Frame generation** runs in a background thread to keep UI responsive
- **ROI updates** happen synchronously for immediate feedback

**Limits:**
- Time trace: Max 10,000 points (downsampled if exceeded)
- IEI histogram: Max 10,000 intervals
- Frame viewer: Max 1,000 frames displayed (full export unaffected)

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Guidelines

- Follow existing code style and structure
- Add docstrings to new functions
- Update README if adding user-facing features
- Test your changes before submitting

---

## 📚 Dependencies

This project uses **Pixi** for reproducible dependency management. All dependencies are automatically handled by Pixi.

### Core Libraries
- **NiceGUI** - Web-based user interface
- **Plotly** - Interactive plotting
- **NumPy** - Numerical operations
- **imageio** - Image/video I/O
- **tqdm** - Progress bars

### Event Camera Specific
- **Metavision SDK** - Prophesee event camera support
  - Installation varies by platform
  - See [Windows](https://docs.prophesee.ai/stable/installation/windows.html) or [Linux](https://docs.prophesee.ai/stable/installation/linux.html) installation guides
  - Includes SDK driver and Python bindings

See `pixi.toml` for the complete list of dependencies and their versions.

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- Built with [NiceGUI](https://nicegui.io/)
- Interactive visualizations powered by [Plotly](https://plotly.com/)
- Event camera support via [Prophesee Metavision SDK](https://docs.prophesee.ai/)
- Developed with assistance from [Claude](https://claude.ai/) (Anthropic)
- Inspired by the work of Cabriel et al. on event-based super-resolution microscopy — [Evb-SMLM](https://github.com/Clement-Cabriel/Evb-SMLM) ([Nature Photonics, 2023](https://doi.org/10.1038/s41566-023-01308-8)) — which motivated our adoption of event-based cameras

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Omnistic/evk4_dashboard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Omnistic/evk4_dashboard/discussions)
- **Documentation**: [Wiki](https://github.com/Omnistic/evk4_dashboard/wiki)

---

<div align="center">

**Made with ❤️ for the event-based vision community**

[⭐ Star this repo](https://github.com/Omnistic/evk4_dashboard) • [🐛 Report Bug](https://github.com/Omnistic/evk4_dashboard/issues) • [✨ Request Feature](https://github.com/Omnistic/evk4_dashboard/issues)

</div>