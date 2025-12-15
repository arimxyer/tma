# Neon City

A cyberpunk cityscape animation with procedural buildings, holographic ads, and rain particles.

![Preview](preview.gif)

## Run

```bash
python -m neon_city
```

## Controls

| Key | Action |
|-----|--------|
| `R` | Regenerate city, vehicles, and holograms |
| `ESC` | Exit |

## Features

- Procedurally generated cyberpunk buildings
- Holographic billboard advertisements with GIF support
- Rain and smoke particle systems
- Flying vehicles with light trails
- Power lines and city cables
- Character silhouette with street lamp
- Post-processing shader (scanlines, vignette, color boost)
- Parallax scrolling city layers

## Technical Details

- 1920x1080 display window
- 480x270 virtual resolution (pixel art upscaling)
- 60 FPS target
- GLSL fragment shader post-processing

## Requirements

- Python 3.10+
- raylib
- cffi
- numpy
- pillow (for GIF loading)
