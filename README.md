# Terminal Animation Art

A collection of terminal-based animations and visualizations written in Python.

## Gallery

### Neon City

Cyberpunk cityscape with holographic ads, rain particles, and parallax scrolling.

![Neon City](neon_city/preview.gif)

[View Details](neon_city/) | `python -m neon_city`

---

### Ocean Waves

Terminal water simulation using sine wave interference patterns.

![Ocean Waves](ocean_waves/preview.gif)

[View Details](ocean_waves/) | `python ocean_waves/ocean_waves.py`

---

### Pier View

Advanced ocean with wave sets, rogue waves, and atmospheric perspective.

![Pier View](pier_view/preview.gif)

[View Details](pier_view/) | `python pier_view/pier_view.py`

---

### North Sea

Rough ocean view from a boat with 6DOF vessel motion simulation.

![North Sea](north_sea/preview.gif)

[View Details](north_sea/) | `python north_sea/north_sea.py`

---

### Cube

Rotating 3D cube with sub-pixel rendering.

![Cube](cube/preview.gif)

[View Details](cube/) | `python cube/cube.py`

---

### Hyper Cube

4D tesseract projected into 3D space.

![Hyper Cube](hyper_cube/preview.gif)

[View Details](hyper_cube/) | `python hyper_cube/hyper_cube.py`

---

### Matrix Ocean

Ocean waves with Matrix-style code characters.

![Matrix Ocean](matrix_ocean/preview.gif)

[View Details](matrix_ocean/) | `python matrix_ocean/matrix_ocean.py`

---

### Matrix Cube

Rotating 3D cube with Matrix code rain aesthetic.

![Matrix Cube](matrix_cube/preview.gif)

[View Details](matrix_cube/) | `python matrix_cube/matrix_cube.py`

---

### Terminal Cube

Retro terminal-styled cube with hexadecimal glyphs.

![Terminal Cube](terminal_cube/preview.gif)

[View Details](terminal_cube/) | `python terminal_cube/terminal_cube.py`

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/animations.git
cd animations

# Run the graphical animation (requires raylib)
python -m neon_city

# Run any terminal animation
python ocean_waves/ocean_waves.py
python cube/cube.py

# Or use uv for automatic dependency management
uv run ocean_waves/ocean_waves.py
```

## Requirements

- Python 3.12+ (3.10+ for neon_city)
- numpy (all terminal animations)
- raylib, cffi, pillow (neon_city only)

## Installation

```bash
# Using pip
pip install numpy raylib pillow

# Or using uv (recommended)
uv sync
```

## License

MIT
