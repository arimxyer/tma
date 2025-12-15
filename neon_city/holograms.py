"""
Hologram system for neon city - Blade Runner 2049 style floating advertisements.
"""

import pyray as rl
import random
import math
import os
from PIL import Image

from .config import (
    VIRTUAL_WIDTH,
    VIRTUAL_HEIGHT,
    PALETTE_HOLOGRAM,
    HOLOGRAM_SLOTS,
    HOLOGRAM_COUNT_MIN,
    HOLOGRAM_COUNT_MAX,
    HOLOGRAM_HEIGHT_MIN,
    HOLOGRAM_HEIGHT_MAX,
    HOLOGRAM_ASSETS,
    GLITCH_CHANCE,
    GLITCH_DURATION_MIN,
    GLITCH_DURATION_MAX,
)


def pil_to_raylib_texture(pil_image):
    """Convert a PIL Image to a Raylib Texture2D."""
    # Ensure RGBA format
    if pil_image.mode != "RGBA":
        pil_image = pil_image.convert("RGBA")

    width, height = pil_image.size
    pixels = pil_image.tobytes()

    # Create raylib Image from raw pixel data
    rl_image = rl.Image()
    rl_image.width = width
    rl_image.height = height
    rl_image.mipmaps = 1
    rl_image.format = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8

    # Allocate and copy pixel data
    data_ptr = rl.ffi.new(f"unsigned char[{len(pixels)}]", pixels)
    rl_image.data = rl.ffi.cast("void *", data_ptr)

    # Convert to texture
    texture = rl.load_texture_from_image(rl_image)

    return texture, data_ptr  # Return data_ptr to keep it alive


def extract_gif_frames(path):
    """Extract frames from an animated GIF using PIL."""
    gif = Image.open(path)
    frames = []
    durations = []

    try:
        while True:
            # Convert frame to RGBA
            frame = gif.convert("RGBA")
            frames.append(frame.copy())
            # Get frame duration in ms, default to 100ms
            durations.append(
                gif.info.get("duration", 100) / 1000.0
            )  # Convert to seconds
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    return frames, durations


class Hologram:
    """Single hologram instance with rendering and effects."""

    def __init__(
        self,
        x,
        base_y,
        textures,
        data_ptrs,
        color,
        height,
        is_animated,
        frame_durations,
    ):
        self.x = x  # Screen x position (center)
        self.base_y = base_y  # Bottom of hologram (projector location)
        self.textures = textures  # List of raylib textures
        self.data_ptrs = data_ptrs  # Keep data pointers alive
        self.color = color  # Tint color
        self.height = height  # Target height in pixels
        self.is_animated = is_animated
        self.frame_durations = frame_durations if frame_durations else [0.1]

        # Calculate width based on aspect ratio of first texture
        if textures:
            aspect = textures[0].width / textures[0].height
            self.width = int(height * aspect)
        else:
            self.width = int(height * 0.6)

        # Animation state
        self.current_frame = 0
        self.frame_timer = 0.0
        self.glitch_timer = 0.0
        self.glitch_active = False
        self.glitch_type = None
        self.glitch_intensity = 0.0

        # Breathing/float animation
        self.float_phase = random.random() * math.pi * 2
        self.float_speed = 0.5 + random.random() * 0.3

        # Projector position (on a building rooftop)
        # Randomize height to simulate different building heights
        self.projector_y = random.randint(VIRTUAL_HEIGHT - 70, VIRTUAL_HEIGHT - 40)

    def update(self, dt):
        """Update animation state."""
        # Update floating animation phase
        self.float_phase += dt * self.float_speed

        # Update GIF animation
        if self.is_animated and len(self.textures) > 1:
            self.frame_timer += dt
            if self.frame_timer >= self.frame_durations[self.current_frame]:
                self.frame_timer = 0.0
                self.current_frame = (self.current_frame + 1) % len(self.textures)

        # Update glitch state
        if self.glitch_active:
            self.glitch_timer -= dt
            if self.glitch_timer <= 0:
                self.glitch_active = False
                self.glitch_type = None
        else:
            # Random chance to trigger glitch
            if random.random() < GLITCH_CHANCE:
                self.glitch_active = True
                self.glitch_timer = random.uniform(
                    GLITCH_DURATION_MIN, GLITCH_DURATION_MAX
                )
                self.glitch_type = random.choice(
                    ["chromatic", "tear", "static", "flutter"]
                )
                self.glitch_intensity = random.uniform(0.5, 1.0)

    def draw(self):
        """Draw the hologram with all effects."""
        if not self.textures:
            return

        # Calculate floating offset
        float_offset = math.sin(self.float_phase) * 3

        # Hologram position (centered on x, floating above base_y)
        holo_x = int(self.x - self.width // 2)
        holo_y = int(self.base_y - self.height + float_offset)

        # Draw projector beam first (behind hologram)
        self._draw_projector_beam(holo_x, holo_y)

        # Get current texture
        texture = self.textures[self.current_frame]


        # Source rectangle (full texture)
        source = rl.Rectangle(0, 0, texture.width, texture.height)

        # Destination rectangle
        dest = rl.Rectangle(holo_x, holo_y, self.width, self.height)

        # Apply glitch effects or normal render
        if self.glitch_active:
            self._draw_with_glitch(texture, source, dest)
        else:
            self._draw_normal(texture, source, dest)

        # Draw scanlines overlay
        self._draw_scanlines(holo_x, holo_y, self.width, self.height)

    def _draw_normal(self, texture, source, dest):
        """Draw hologram with normal transparency and tint."""
        # Draw with color tint and transparency
        tint = rl.Color(self.color.r, self.color.g, self.color.b, 180)
        rl.draw_texture_pro(texture, source, dest, rl.Vector2(0, 0), 0.0, tint)

        # Add a subtle glow layer
        glow_tint = rl.Color(self.color.r, self.color.g, self.color.b, 60)
        glow_dest = rl.Rectangle(
            dest.x - 2, dest.y - 2, dest.width + 4, dest.height + 4
        )
        rl.draw_texture_pro(
            texture, source, glow_dest, rl.Vector2(0, 0), 0.0, glow_tint
        )

    def _draw_with_glitch(self, texture, source, dest):
        """Draw hologram with active glitch effect."""
        if self.glitch_type == "chromatic":
            self._draw_glitch_chromatic(texture, source, dest)
        elif self.glitch_type == "tear":
            self._draw_glitch_tear(texture, source, dest)
        elif self.glitch_type == "static":
            self._draw_normal(texture, source, dest)
            self._draw_glitch_static(
                int(dest.x), int(dest.y), int(dest.width), int(dest.height)
            )
        elif self.glitch_type == "flutter":
            self._draw_glitch_flutter(texture, source, dest)
        else:
            self._draw_normal(texture, source, dest)

    def _draw_glitch_chromatic(self, texture, source, dest):
        """Chromatic aberration - RGB channel separation."""
        offset = int(2 + self.glitch_intensity * 3)

        # Red channel (offset left)
        red_dest = rl.Rectangle(dest.x - offset, dest.y, dest.width, dest.height)
        red_tint = rl.Color(255, 0, 0, 120)
        rl.draw_texture_pro(texture, source, red_dest, rl.Vector2(0, 0), 0.0, red_tint)

        # Green channel (center)
        green_tint = rl.Color(0, 255, 0, 120)
        rl.draw_texture_pro(texture, source, dest, rl.Vector2(0, 0), 0.0, green_tint)

        # Blue channel (offset right)
        blue_dest = rl.Rectangle(dest.x + offset, dest.y, dest.width, dest.height)
        blue_tint = rl.Color(0, 0, 255, 120)
        rl.draw_texture_pro(
            texture, source, blue_dest, rl.Vector2(0, 0), 0.0, blue_tint
        )

    def _draw_glitch_tear(self, texture, source, dest):
        """Horizontal tear lines - slices displaced horizontally."""
        num_slices = 8
        slice_height = dest.height / num_slices

        for i in range(num_slices):
            # Random horizontal offset for some slices
            offset_x = 0
            if random.random() < 0.4:
                offset_x = random.randint(-8, 8) * self.glitch_intensity

            # Source slice
            src_slice = rl.Rectangle(
                source.x,
                source.y + (source.height / num_slices) * i,
                source.width,
                source.height / num_slices,
            )

            # Destination slice
            dst_slice = rl.Rectangle(
                dest.x + offset_x,
                dest.y + slice_height * i,
                dest.width,
                slice_height + 1,  # +1 to avoid gaps
            )

            tint = rl.Color(self.color.r, self.color.g, self.color.b, 180)
            rl.draw_texture_pro(
                texture, src_slice, dst_slice, rl.Vector2(0, 0), 0.0, tint
            )

    def _draw_glitch_static(self, x, y, w, h):
        """Draw random static noise rectangles."""
        num_rects = int(10 * self.glitch_intensity)
        for _ in range(num_rects):
            rx = x + random.randint(0, w - 5)
            ry = y + random.randint(0, h - 3)
            rw = random.randint(3, 15)
            rh = random.randint(1, 3)

            # Random color from hologram palette or white noise
            if random.random() < 0.5:
                col = rl.Color(255, 255, 255, random.randint(100, 200))
            else:
                base = random.choice(PALETTE_HOLOGRAM)
                col = rl.Color(base.r, base.g, base.b, random.randint(100, 200))

            rl.draw_rectangle(rx, ry, rw, rh, col)

    def _draw_glitch_flutter(self, texture, source, dest):
        """Rapid alpha oscillation."""
        flutter_alpha = int(80 + random.random() * 175)
        tint = rl.Color(self.color.r, self.color.g, self.color.b, flutter_alpha)
        rl.draw_texture_pro(texture, source, dest, rl.Vector2(0, 0), 0.0, tint)

    def _draw_scanlines(self, x, y, w, h):
        """Draw scanline overlay effect."""
        # Draw every other line slightly darker
        scanline_color = rl.Color(0, 0, 0, 40)
        for row in range(0, int(h), 2):
            rl.draw_line(x, y + row, x + w, y + row, scanline_color)

    def _draw_projector_beam(self, holo_x, holo_y):
        """Draw the light beam from building rooftop to hologram."""
        # Beam originates from building rooftop
        beam_base_x = self.x
        beam_base_y = self.projector_y  # From building rooftop

        # Beam top connects to bottom of hologram
        beam_top_left = holo_x - 2
        beam_top_right = holo_x + self.width + 2
        beam_top_y = holo_y + self.height

        # Only draw if beam has positive height (hologram above projector)
        beam_height = beam_base_y - beam_top_y
        if beam_height <= 0:
            return

        # Draw filled beam using horizontal lines (creates gradient fill)
        for i in range(0, int(beam_height), 1):
            y = beam_top_y + i
            # Interpolate width from top (full width) to bottom (point)
            t = i / beam_height
            current_width = (1 - t) * (beam_top_right - beam_top_left)
            x_left = int(beam_base_x - current_width / 2)
            x_right = int(beam_base_x + current_width / 2)

            # Fade alpha towards the bottom
            alpha = int(25 * (1 - t * 0.3))
            line_color = rl.Color(self.color.r, self.color.g, self.color.b, alpha)
            rl.draw_line(x_left, int(y), x_right, int(y), line_color)

        # Add brighter edge lines for definition
        edge_color = rl.Color(self.color.r, self.color.g, self.color.b, 60)
        rl.draw_line_ex(
            rl.Vector2(beam_base_x, beam_base_y),
            rl.Vector2(beam_top_left, beam_top_y),
            1,
            edge_color,
        )
        rl.draw_line_ex(
            rl.Vector2(beam_base_x, beam_base_y),
            rl.Vector2(beam_top_right, beam_top_y),
            1,
            edge_color,
        )

        # Add a bright spot at the projector source
        rl.draw_circle(
            beam_base_x,
            beam_base_y,
            3,
            rl.Color(self.color.r, self.color.g, self.color.b, 100),
        )


class RotatingHologram:
    """A hologram that displays a texture rotating in 3D space."""

    def __init__(self, x, base_y, textures, data_ptrs, color, height):
        self.x = x
        self.base_y = base_y
        self.textures = textures
        self.data_ptrs = data_ptrs
        self.color = color
        self.height = height

        # Calculate width based on aspect ratio
        if textures:
            aspect = textures[0].width / textures[0].height
            self.width = int(height * aspect)
        else:
            self.width = int(height * 0.6)

        # Rotation state
        self.rotation_angle = random.random() * math.pi * 2
        self.rotation_speed = 0.8 + random.random() * 0.4  # Vary speed slightly

        # Floating animation
        self.float_phase = random.random() * math.pi * 2
        self.float_speed = 0.5 + random.random() * 0.3

        # Glitch state (shared with regular Hologram)
        self.glitch_timer = 0.0
        self.glitch_active = False
        self.glitch_type = None
        self.glitch_intensity = 0.0

        # Projector position (on a building rooftop)
        self.projector_y = random.randint(VIRTUAL_HEIGHT - 70, VIRTUAL_HEIGHT - 40)

    def update(self, dt):
        """Update rotation and animation state."""
        self.rotation_angle += dt * self.rotation_speed
        self.float_phase += dt * self.float_speed

        # Update glitch state
        if self.glitch_active:
            self.glitch_timer -= dt
            if self.glitch_timer <= 0:
                self.glitch_active = False
                self.glitch_type = None
        else:
            if random.random() < GLITCH_CHANCE:
                self.glitch_active = True
                self.glitch_timer = random.uniform(
                    GLITCH_DURATION_MIN, GLITCH_DURATION_MAX
                )
                self.glitch_type = random.choice(
                    ["chromatic", "tear", "static", "flutter"]
                )
                self.glitch_intensity = random.uniform(0.5, 1.0)

    def draw(self):
        """Draw the rotating hologram with 3D perspective effect."""
        if not self.textures:
            return

        texture = self.textures[0]

        # Calculate floating offset
        float_offset = math.sin(self.float_phase) * 3

        # 3D rotation creates a width scaling effect (simulate viewing angle)
        # When viewing from front (0 or pi), full width
        # When viewing from side (pi/2 or 3pi/2), narrow (edge-on)
        scale_x = abs(math.cos(self.rotation_angle))
        scale_x = max(0.1, scale_x)  # Never fully disappear

        # Calculate displayed width based on rotation
        displayed_width = int(self.width * scale_x)

        # Hologram position
        holo_x = int(self.x - displayed_width // 2)
        holo_y = int(self.base_y - self.height + float_offset)

        # Draw projector beam
        self._draw_projector_beam(holo_x, holo_y, displayed_width)

        # Draw a subtle glowing background panel for visibility
        self._draw_background_panel(holo_x, holo_y, displayed_width)

        # Source rectangle (full texture)
        source = rl.Rectangle(0, 0, texture.width, texture.height)

        # Determine if we're showing front or back of the "card"
        showing_back = math.cos(self.rotation_angle) < 0

        # If showing back, flip the source horizontally
        if showing_back:
            source.width = -texture.width

        # Destination rectangle with perspective scaling
        dest = rl.Rectangle(holo_x, holo_y, displayed_width, self.height)

        # Apply glitch or normal render
        if self.glitch_active:
            self._draw_with_glitch(texture, source, dest)
        else:
            self._draw_normal(texture, source, dest)

        # Draw scanlines
        self._draw_scanlines(holo_x, holo_y, displayed_width, self.height)

        # Add edge glow when rotating (simulate 3D edge lighting)
        if scale_x < 0.7:
            edge_alpha = int((1 - scale_x) * 150)
            edge_color = rl.Color(self.color.r, self.color.g, self.color.b, edge_alpha)
            # Draw edge highlight on the leading edge
            edge_x = (
                holo_x
                if math.sin(self.rotation_angle) > 0
                else holo_x + displayed_width
            )
            rl.draw_line(edge_x, holo_y, edge_x, holo_y + self.height, edge_color)

    def _draw_background_panel(self, holo_x, holo_y, displayed_width):
        """Draw a background panel matching the projector beam fill style."""
        # Draw filled panel using horizontal lines (same style as beam)
        panel_x = holo_x - 3
        panel_y = holo_y - 3
        panel_w = displayed_width + 6
        panel_h = self.height + 6

        # Fill with horizontal lines at consistent alpha (matching beam)
        fill_color = rl.Color(self.color.r, self.color.g, self.color.b, 25)
        for y in range(panel_y, panel_y + panel_h):
            rl.draw_line(panel_x, y, panel_x + panel_w, y, fill_color)

        # Border glow
        border_color = rl.Color(self.color.r, self.color.g, self.color.b, 60)
        rl.draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, border_color)

        # Inner darker panel
        panel_color = rl.Color(
            self.color.r // 8, self.color.g // 8, self.color.b // 8, 120
        )
        rl.draw_rectangle(
            holo_x - 2, holo_y - 2, displayed_width + 4, self.height + 4, panel_color
        )

    def _draw_normal(self, texture, source, dest):
        """Draw with normal transparency and tint."""
        tint = rl.Color(self.color.r, self.color.g, self.color.b, 180)
        rl.draw_texture_pro(texture, source, dest, rl.Vector2(0, 0), 0.0, tint)

        # Glow layer
        glow_tint = rl.Color(self.color.r, self.color.g, self.color.b, 60)
        glow_dest = rl.Rectangle(
            dest.x - 2, dest.y - 2, dest.width + 4, dest.height + 4
        )
        rl.draw_texture_pro(
            texture, source, glow_dest, rl.Vector2(0, 0), 0.0, glow_tint
        )

    def _draw_with_glitch(self, texture, source, dest):
        """Draw with glitch effect."""
        if self.glitch_type == "chromatic":
            offset = int(2 + self.glitch_intensity * 3)
            red_dest = rl.Rectangle(dest.x - offset, dest.y, dest.width, dest.height)
            rl.draw_texture_pro(
                texture,
                source,
                red_dest,
                rl.Vector2(0, 0),
                0.0,
                rl.Color(255, 0, 0, 120),
            )
            rl.draw_texture_pro(
                texture, source, dest, rl.Vector2(0, 0), 0.0, rl.Color(0, 255, 0, 120)
            )
            blue_dest = rl.Rectangle(dest.x + offset, dest.y, dest.width, dest.height)
            rl.draw_texture_pro(
                texture,
                source,
                blue_dest,
                rl.Vector2(0, 0),
                0.0,
                rl.Color(0, 0, 255, 120),
            )
        elif self.glitch_type == "flutter":
            flutter_alpha = int(80 + random.random() * 175)
            tint = rl.Color(self.color.r, self.color.g, self.color.b, flutter_alpha)
            rl.draw_texture_pro(texture, source, dest, rl.Vector2(0, 0), 0.0, tint)
        else:
            self._draw_normal(texture, source, dest)

    def _draw_scanlines(self, x, y, w, h):
        """Draw scanline overlay."""
        scanline_color = rl.Color(0, 0, 0, 40)
        for row in range(0, int(h), 2):
            rl.draw_line(x, y + row, x + w, y + row, scanline_color)

    def _draw_projector_beam(self, holo_x, holo_y, displayed_width):
        """Draw the projector beam from building rooftop."""
        beam_base_x = self.x
        beam_base_y = self.projector_y  # From building rooftop

        beam_top_left = holo_x - 2
        beam_top_right = holo_x + displayed_width + 2
        beam_top_y = holo_y + self.height

        # Only draw if beam has positive height
        beam_height = beam_base_y - beam_top_y
        if beam_height <= 0:
            return

        # Draw filled beam
        for i in range(0, int(beam_height), 1):
            y = beam_top_y + i
            t = i / beam_height
            current_width = (1 - t) * (beam_top_right - beam_top_left)
            x_left = int(beam_base_x - current_width / 2)
            x_right = int(beam_base_x + current_width / 2)
            alpha = int(25 * (1 - t * 0.3))
            line_color = rl.Color(self.color.r, self.color.g, self.color.b, alpha)
            rl.draw_line(x_left, int(y), x_right, int(y), line_color)

        # Edge lines
        edge_color = rl.Color(self.color.r, self.color.g, self.color.b, 60)
        rl.draw_line_ex(
            rl.Vector2(beam_base_x, beam_base_y),
            rl.Vector2(beam_top_left, beam_top_y),
            1,
            edge_color,
        )
        rl.draw_line_ex(
            rl.Vector2(beam_base_x, beam_base_y),
            rl.Vector2(beam_top_right, beam_top_y),
            1,
            edge_color,
        )

        # Projector spot
        rl.draw_circle(
            beam_base_x,
            beam_base_y,
            3,
            rl.Color(self.color.r, self.color.g, self.color.b, 100),
        )


class HologramManager:
    """Manages loading, placement, and rendering of all holograms."""

    def __init__(self):
        self.holograms = []
        self.loaded_assets = {}  # Cache: path -> (textures, data_ptrs, is_animated, durations)
        self._load_assets()
        self._create_holograms()

    def _load_assets(self):
        """Load all hologram assets from config."""
        for asset in HOLOGRAM_ASSETS:
            path = asset["path"]
            asset_type = asset["type"]

            if not os.path.exists(path):
                print(f"Warning: Hologram asset not found: {path}")
                continue

            if asset_type == "animated":
                # Extract GIF frames
                pil_frames, durations = extract_gif_frames(path)
                textures = []
                data_ptrs = []

                for frame in pil_frames:
                    tex, ptr = pil_to_raylib_texture(frame)
                    textures.append(tex)
                    data_ptrs.append(ptr)

                self.loaded_assets[path] = (textures, data_ptrs, "animated", durations)
            elif asset_type == "rotating":
                # Load static image for rotating display
                pil_img = Image.open(path).convert("RGBA")
                tex, ptr = pil_to_raylib_texture(pil_img)
                self.loaded_assets[path] = ([tex], [ptr], "rotating", [0.1])
            else:
                # Load static image
                pil_img = Image.open(path).convert("RGBA")
                tex, ptr = pil_to_raylib_texture(pil_img)
                self.loaded_assets[path] = ([tex], [ptr], "static", [0.1])

    def _create_holograms(self):
        """Create hologram instances at random slots."""
        if not self.loaded_assets:
            print("Warning: No hologram assets loaded")
            return

        # Decide how many holograms (1 or 2)
        count = random.randint(HOLOGRAM_COUNT_MIN, HOLOGRAM_COUNT_MAX)

        # Select random slots
        available_slots = HOLOGRAM_SLOTS.copy()
        selected_slots = random.sample(
            available_slots, min(count, len(available_slots))
        )

        # Select random assets
        asset_paths = list(self.loaded_assets.keys())
        selected_assets = random.sample(asset_paths, min(count, len(asset_paths)))

        # Create holograms
        for i, (slot, asset_path) in enumerate(zip(selected_slots, selected_assets)):
            textures, data_ptrs, holo_type, durations = self.loaded_assets[asset_path]

            # Calculate position
            x = int(VIRTUAL_WIDTH * slot)

            # Base Y is somewhere in the upper portion of screen (above buildings)
            # Holograms float above the cityscape
            base_y = random.randint(VIRTUAL_HEIGHT - 120, VIRTUAL_HEIGHT - 80)

            # Random height within range
            height = random.randint(HOLOGRAM_HEIGHT_MIN, HOLOGRAM_HEIGHT_MAX)

            # Random color from hologram palette
            color = random.choice(PALETTE_HOLOGRAM)

            if holo_type == "rotating":
                # Create rotating 3D hologram
                hologram = RotatingHologram(
                    x=x,
                    base_y=base_y,
                    textures=textures,
                    data_ptrs=data_ptrs,
                    color=color,
                    height=height,
                )
            else:
                # Create standard hologram (static or animated)
                hologram = Hologram(
                    x=x,
                    base_y=base_y,
                    textures=textures,
                    data_ptrs=data_ptrs,
                    color=color,
                    height=height,
                    is_animated=(holo_type == "animated"),
                    frame_durations=durations,
                )

            self.holograms.append(hologram)

    def update(self, dt):
        """Update all holograms."""
        for hologram in self.holograms:
            hologram.update(dt)

    def draw(self):
        """Draw all holograms."""
        for hologram in self.holograms:
            hologram.draw()

    def unload(self):
        """Unload all textures."""
        for path, (textures, data_ptrs, _, _) in self.loaded_assets.items():
            for tex in textures:
                rl.unload_texture(tex)
        self.loaded_assets.clear()
        self.holograms.clear()
