"""
Building and City classes for the neon cityscape.
"""

import pyray as rl
import random
import math
import time

from .config import (
    VIRTUAL_WIDTH,
    VIRTUAL_HEIGHT,
    PALETTE_NEON,
    C_BUILDING_BACK,
    C_BUILDING_BACK_TOP,
    C_BUILDING_BACK_BOT,
    C_BUILDING_MID,
    C_BUILDING_MID_TOP,
    C_BUILDING_MID_BOT,
    C_BUILDING_FRONT,
    C_BUILDING_FRONT_TOP,
    C_BUILDING_FRONT_BOT,
    C_HAZE_BACK,
    C_HAZE_MID,
    BILLBOARD_AD_IMAGES,
    BILLBOARD_IMAGE_CHANCE,
)


class Building:
    def __init__(self, x, w, h, layer, billboard_textures=None):
        self.x = x
        self.w = w
        self.h = h
        self.layer = layer
        self.billboard_textures = billboard_textures or []

        # Gradient colors based on layer depth
        if layer == 0:
            self.color = C_BUILDING_BACK
            self.color_top = C_BUILDING_BACK_TOP
            self.color_bot = C_BUILDING_BACK_BOT
        elif layer == 1:
            self.color = C_BUILDING_MID
            self.color_top = C_BUILDING_MID_TOP
            self.color_bot = C_BUILDING_MID_BOT
        else:
            self.color = C_BUILDING_FRONT
            self.color_top = C_BUILDING_FRONT_TOP
            self.color_bot = C_BUILDING_FRONT_BOT

        self.windows = []
        self.billboards = []
        self.rooftop_features = []
        self.neon_signs = []

        # Generate rooftop features
        self._generate_rooftop(w, h, layer)

        # Generate Windows - enhanced patterns
        self._generate_windows(w, h, layer)

        # Billboards (Mid/Front layers only)
        if layer > 0 and random.random() < 0.5:
            self._generate_billboard(w, h)

        # Neon signs (small text signs near rooftops)
        sign_chance = 0.12 if layer == 0 else (0.18 if layer == 1 else 0.25)
        if random.random() < sign_chance:
            self._generate_neon_sign(w, h)

    def _generate_rooftop(self, w, h, layer):
        # Antennas - thin vertical lines
        if w > 12:
            num_antennas = random.randint(0, 3) if layer < 2 else random.randint(0, 2)
            for _ in range(num_antennas):
                ax = random.randint(5, max(6, w - 5))
                ah = random.randint(10, 30)
                self.rooftop_features.append(("antenna", ax, ah))

        # Spires/towers on top
        if random.random() < 0.3 and w > 35:
            sw = random.randint(8, min(20, w // 3))
            sh = random.randint(15, 40)
            sx = random.randint(5, max(6, w - sw - 5))
            self.rooftop_features.append(("spire", sx, sw, sh))

        # AC units (small boxes) - front layer mainly
        if layer >= 1 and random.random() < 0.5 and w > 15:
            num_ac = random.randint(1, 3)
            for _ in range(num_ac):
                acx = random.randint(3, max(4, w - 10))
                acw = random.randint(5, 10)
                ach = random.randint(3, 6)
                self.rooftop_features.append(("ac", acx, acw, ach))

        # Setback (stepped top) - creates more interesting silhouette
        if random.random() < 0.4 and w > 45:
            setback_w = random.randint(w // 3, w // 2)
            setback_h = random.randint(10, 25)
            setback_x = random.randint(5, max(6, w - setback_w - 5))
            self.rooftop_features.append(("setback", setback_x, setback_w, setback_h))

    def _generate_windows(self, w, h, layer):
        base_color = random.choice(PALETTE_NEON)
        style = random.choice(
            ["strips", "grid", "dense_strips", "dense_grid", "scattered"]
        )

        if style == "strips":
            cols = w // 5
            for c in range(cols):
                if random.random() > 0.5:
                    continue
                wx = c * 5 + 2
                h_start = random.randint(20, max(21, h - 10))
                h_len = random.randint(15, max(16, h_start - 5))
                for r in range(h_len // 3):
                    wy = h_start - r * 3
                    self.windows.append((wx, wy, 2, 2, base_color))

        elif style == "dense_strips":
            num_strips = random.randint(3, 6)
            for _ in range(num_strips):
                strip_x = random.randint(2, max(3, w - 5))
                strip_h = random.randint(max(10, h // 3), max(15, h - 8))
                strip_start = random.randint(4, max(5, h - strip_h - 2))
                strip_color = (
                    base_color if random.random() > 0.3 else random.choice(PALETTE_NEON)
                )
                for y in range(strip_start, min(strip_start + strip_h, h - 2), 2):
                    self.windows.append((strip_x, y, 2, 1, strip_color))

        elif style == "dense_grid":
            rows = h // 4
            cols = w // 4
            dark_rows = (
                set(random.sample(range(rows), min(rows // 5, 2)))
                if rows > 5
                else set()
            )
            for r in range(rows):
                if r in dark_rows:
                    continue
                for c in range(cols):
                    if random.random() > 0.4:
                        wx = c * 4 + 1
                        wy = r * 4 + 2
                        col = (
                            base_color
                            if random.random() > 0.15
                            else random.choice(PALETTE_NEON)
                        )
                        self.windows.append((wx, wy, 2, 3, col))

        elif style == "grid":
            rows = h // 5
            cols = w // 4
            dark_rows = (
                set(random.sample(range(rows), min(rows // 4, 3)))
                if rows > 4
                else set()
            )
            for r in range(rows):
                if r in dark_rows:
                    continue
                for c in range(cols):
                    if random.random() > 0.5:
                        wx = c * 4 + 1
                        wy = r * 5 + 3
                        col = (
                            base_color
                            if random.random() > 0.1
                            else random.choice(PALETTE_NEON)
                        )
                        self.windows.append((wx, wy, 2, 3, col))

        else:  # scattered
            num_windows = random.randint(20, 50)
            for _ in range(num_windows):
                wx = random.randint(2, w - 4)
                wy = random.randint(4, h - 4)
                ww = random.randint(1, 3)
                wh = random.randint(2, 4)
                self.windows.append((wx, wy, ww, wh, base_color))

    def _generate_billboard(self, w, h):
        bw = random.randint(15, 30)
        bh = random.randint(20, 40)

        if bw < w - 4 and bh < (h - 15):
            bx = (w - bw) // 2
            y_min = 8
            y_max = max(9, h - bh - 8)
            by = random.randint(y_min, y_max)

            b_col = random.choice(PALETTE_NEON)

            # Decide if this billboard uses an image or text
            use_image = (
                self.billboard_textures and random.random() < BILLBOARD_IMAGE_CHANCE
            )

            if use_image:
                # Image billboard
                texture = random.choice(self.billboard_textures)
                self.billboards.append(
                    {
                        "x": bx,
                        "y": by,
                        "w": bw,
                        "h": bh,
                        "color": b_col,
                        "type": random.choice(
                            ["image_pulse", "image_flicker", "image_glitch"]
                        ),
                        "texture": texture,
                        "secondary_color": random.choice(PALETTE_NEON),
                        "glitch_offset": random.random() * 100,
                    }
                )
            else:
                # Text billboard (original behavior)
                text_pattern = self._generate_futuristic_text(bw, bh)

                # Generate multiple text patterns for "video" type
                extra_patterns = [
                    self._generate_futuristic_text(bw, bh),
                    self._generate_futuristic_text(bw, bh),
                ]

                self.billboards.append(
                    {
                        "x": bx,
                        "y": by,
                        "w": bw,
                        "h": bh,
                        "color": b_col,
                        "type": random.choice(
                            [
                                "pulse",
                                "scroll",
                                "solid",
                                "flicker",
                                "glitch",
                                "rainbow",
                                "video",
                            ]
                        ),
                        "text_pattern": text_pattern,
                        "extra_patterns": extra_patterns,
                        "scroll_offset": random.random() * 100,
                        "secondary_color": random.choice(PALETTE_NEON),
                        "glitch_offset": random.random() * 100,
                        "video_frame": 0,
                        "video_timer": 0,
                    }
                )

    def _generate_futuristic_text(self, bw, bh):
        """Generate pixel patterns resembling futuristic mixed-language text."""
        char_patterns = [
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (0, 3),
                (0, 4),
                (1, 4),
                (2, 4),
            ],
            [
                (1, 0),
                (0, 1),
                (2, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (0, 3),
                (2, 3),
                (0, 4),
                (2, 4),
            ],
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (0, 1),
                (0, 2),
                (1, 2),
                (0, 3),
                (0, 4),
                (1, 4),
                (2, 4),
            ],
            [(1, 0), (0, 1), (2, 1), (0, 2), (2, 2), (0, 3), (2, 3), (1, 4)],
            [
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (1, 2),
                (0, 3),
                (1, 3),
                (2, 3),
                (0, 4),
                (2, 4),
            ],
            [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2), (0, 3), (1, 3), (2, 3), (1, 4)],
            [
                (0, 0),
                (2, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (1, 2),
                (0, 3),
                (1, 3),
                (2, 3),
                (1, 4),
            ],
            [(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2), (0, 4), (1, 4), (2, 4)],
            [(1, 0), (0, 1), (2, 1), (1, 2), (0, 3), (2, 3), (1, 4)],
            [(1, 0), (1, 1), (0, 2), (1, 2), (2, 2), (1, 3), (1, 4)],
            [(0, 0), (2, 0), (1, 1), (1, 2), (1, 3), (0, 4), (2, 4)],
            [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (2, 4)],
            [(1, 0), (0, 1), (2, 1), (2, 2), (1, 3), (0, 4), (1, 4), (2, 4)],
            [(0, 0), (1, 0), (2, 0), (2, 1), (1, 2), (2, 3), (0, 4), (1, 4), (2, 4)],
        ]

        pattern = []
        num_chars = max(1, (bw - 4) // 5)
        num_rows = max(1, (bh - 4) // 8)

        for row in range(num_rows):
            row_y = 2 + row * 7
            for char_idx in range(num_chars):
                char_x = 2 + char_idx * 5
                char = random.choice(char_patterns)
                for px, py in char:
                    if char_x + px < bw - 1 and row_y + py < bh - 1:
                        pattern.append((char_x + px, row_y + py))

        return pattern

    def _generate_neon_sign(self, w, h):
        """Generate small neon text signs (horizontal or vertical)."""
        # Simple 3x5 pixel font for basic characters
        char_font = {
            "B": [
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (0, 2),
                (1, 2),
                (0, 3),
                (1, 3),
                (2, 3),
                (0, 4),
                (1, 4),
            ],
            "A": [
                (1, 0),
                (0, 1),
                (2, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (0, 3),
                (2, 3),
                (0, 4),
                (2, 4),
            ],
            "R": [
                (0, 0),
                (1, 0),
                (0, 1),
                (2, 1),
                (0, 2),
                (1, 2),
                (0, 3),
                (2, 3),
                (0, 4),
                (2, 4),
            ],
            "H": [
                (0, 0),
                (2, 0),
                (0, 1),
                (2, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (0, 3),
                (2, 3),
                (0, 4),
                (2, 4),
            ],
            "O": [(1, 0), (0, 1), (2, 1), (0, 2), (2, 2), (0, 3), (2, 3), (1, 4)],
            "T": [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
            "E": [
                (0, 0),
                (1, 0),
                (2, 0),
                (0, 1),
                (0, 2),
                (1, 2),
                (0, 3),
                (0, 4),
                (1, 4),
                (2, 4),
            ],
            "L": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4)],
            "N": [
                (0, 0),
                (2, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (0, 2),
                (2, 2),
                (0, 3),
                (2, 3),
                (0, 4),
                (2, 4),
            ],
            "C": [(1, 0), (2, 0), (0, 1), (0, 2), (0, 3), (1, 4), (2, 4)],
            "U": [
                (0, 0),
                (2, 0),
                (0, 1),
                (2, 1),
                (0, 2),
                (2, 2),
                (0, 3),
                (2, 3),
                (1, 4),
            ],
            "S": [(1, 0), (2, 0), (0, 1), (1, 2), (2, 3), (0, 4), (1, 4)],
            "I": [
                (0, 0),
                (1, 0),
                (2, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (0, 4),
                (1, 4),
                (2, 4),
            ],
            "D": [
                (0, 0),
                (1, 0),
                (0, 1),
                (2, 1),
                (0, 2),
                (2, 2),
                (0, 3),
                (2, 3),
                (0, 4),
                (1, 4),
            ],
            "K": [
                (0, 0),
                (2, 0),
                (0, 1),
                (1, 1),
                (0, 2),
                (1, 2),
                (0, 3),
                (1, 3),
                (0, 4),
                (2, 4),
            ],
            "Y": [(0, 0), (2, 0), (0, 1), (2, 1), (1, 2), (1, 3), (1, 4)],
            "P": [(0, 0), (1, 0), (0, 1), (2, 1), (0, 2), (1, 2), (0, 3), (0, 4)],
            "M": [
                (0, 0),
                (2, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (0, 2),
                (2, 2),
                (0, 3),
                (2, 3),
                (0, 4),
                (2, 4),
            ],
            "G": [
                (1, 0),
                (2, 0),
                (0, 1),
                (0, 2),
                (2, 2),
                (0, 3),
                (2, 3),
                (1, 4),
                (2, 4),
            ],
            "F": [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (1, 2), (0, 3), (0, 4)],
            "X": [
                (0, 0),
                (2, 0),
                (0, 1),
                (2, 1),
                (1, 2),
                (0, 3),
                (2, 3),
                (0, 4),
                (2, 4),
            ],
            # Japanese-style simple kanji approximations (abstract)
            "酒": [
                (0, 0),
                (2, 0),
                (1, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (0, 3),
                (2, 3),
                (0, 4),
                (1, 4),
                (2, 4),
            ],  # sake
            "夜": [
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (0, 3),
                (2, 3),
                (1, 4),
            ],  # night
        }

        # Sign text options - mix of English and pseudo-Japanese
        horizontal_signs = [
            "BAR",
            "CLUB",
            "HOTEL",
            "NEON",
            "TECH",
            "SKY",
            "CUBE",
            "DUSK",
            "PULSE",
            "HEX",
            "GRID",
            "SYNC",
            "BYTE",
        ]
        vertical_signs = [
            "BAR",
            "PUB",
            "INN",
            "SPA",
            "GYM",
            "LAB",
            "HUB",
        ]

        # Choose orientation based on building shape
        is_vertical = random.random() < 0.4  # 40% vertical, 60% horizontal

        if is_vertical:
            text = random.choice(vertical_signs)
            # Calculate sign dimensions
            sign_h = len(text) * 6 + 2  # 5px per char + 1px gap + padding
            sign_w = 5  # 3px char + padding

            if sign_h > h - 10 or sign_w > w - 4:
                return  # Building too small

            sign_x = random.randint(2, max(3, w - sign_w - 2))
            # Position near the top of building (within top 25% or first 30px)
            max_y = min(30, h // 4)
            sign_y = random.randint(3, max(4, max_y))

            # Build pattern for vertical text
            pattern = []
            for i, char in enumerate(text):
                if char in char_font:
                    char_y_offset = i * 6
                    for px, py in char_font[char]:
                        pattern.append((px + 1, py + char_y_offset + 1))
        else:
            text = random.choice(horizontal_signs)
            # Calculate sign dimensions
            sign_w = len(text) * 4 + 2  # 3px per char + 1px gap + padding
            sign_h = 7  # 5px char + padding

            if sign_w > w - 4 or sign_h > h - 10:
                return  # Building too small

            sign_x = random.randint(2, max(3, w - sign_w - 2))
            # Position near the top of building (within top 20% or first 25px)
            max_y = min(25, h // 5)
            sign_y = random.randint(3, max(4, max_y))

            # Build pattern for horizontal text
            pattern = []
            for i, char in enumerate(text):
                if char in char_font:
                    char_x_offset = i * 4
                    for px, py in char_font[char]:
                        pattern.append((px + char_x_offset + 1, py + 1))

        sign_color = random.choice(PALETTE_NEON)

        self.neon_signs.append(
            {
                "x": sign_x,
                "y": sign_y,
                "w": sign_w,
                "h": sign_h,
                "pattern": pattern,
                "color": sign_color,
                "text": text,
                "vertical": is_vertical,
            }
        )

    def draw(self, offset_x):
        bx = int(self.x - offset_x)
        by = VIRTUAL_HEIGHT - self.h
        t = time.time()

        # Draw rooftop features first (behind main body for setbacks)
        for feature in self.rooftop_features:
            if feature[0] == "setback":
                _, sx, sw, sh = feature
                rl.draw_rectangle_gradient_v(
                    bx + sx, by - sh, sw, sh, self.color_top, self.color_bot
                )

        # Draw Body with gradient
        rl.draw_rectangle_gradient_v(
            bx, by, self.w, self.h, self.color_top, self.color_bot
        )

        # Draw rooftop features (on top)
        for feature in self.rooftop_features:
            if feature[0] == "antenna":
                _, ax, ah = feature
                rl.draw_line(bx + ax, by, bx + ax, by - ah, rl.Color(60, 60, 70, 255))
                if int(t * 0.5) % 2 == 0:
                    rl.draw_circle(bx + ax, by - ah, 1, rl.Color(255, 50, 50, 255))
            elif feature[0] == "spire":
                _, sx, sw, sh = feature
                rl.draw_rectangle(bx + sx, by - sh, sw, sh, self.color)
                rl.draw_rectangle(
                    bx + sx + sw // 2 - 1,
                    by - sh - 2,
                    2,
                    2,
                    rl.Color(255, 200, 100, 200),
                )
            elif feature[0] == "ac":
                _, acx, acw, ach = feature
                rl.draw_rectangle(
                    bx + acx, by - ach, acw, ach, rl.Color(40, 40, 50, 255)
                )

        # Draw Windows (slow, subtle flicker)
        for wx, wy, ww, wh, col in self.windows:
            win_screen_x = bx + wx
            win_screen_y = by + wy

            window_id = (wx * 7 + wy * 13) % 100
            if window_id < 2:
                if int(t * 0.5 + window_id) % 4 == 0:
                    continue

            rl.draw_rectangle(win_screen_x, win_screen_y, ww, wh, col)

        # Draw Neon Signs (small text signs)
        for sign in self.neon_signs:
            sx = bx + sign["x"]
            sy = by + sign["y"]
            col = sign["color"]
            sign_w = sign.get("w", 20)
            sign_h = sign.get("h", 7)

            pulse = (math.sin(t * 2.0) + 1) * 0.5
            glow_alpha = int(80 + pulse * 80)

            # Draw sign background (dark with slight transparency)
            rl.draw_rectangle(sx, sy, sign_w, sign_h, rl.Color(5, 2, 10, 200))

            # Draw subtle glow behind text
            glow_col = rl.Color(col.r, col.g, col.b, glow_alpha // 2)
            rl.draw_rectangle(sx - 1, sy - 1, sign_w + 2, sign_h + 2, glow_col)

            # Draw the text pixels
            text_alpha = int(200 + pulse * 55)
            text_col = rl.Color(col.r, col.g, col.b, text_alpha)
            for px, py in sign["pattern"]:
                rl.draw_rectangle(sx + px, sy + py, 1, 1, text_col)

        # Draw Billboards with futuristic text or images
        for b in self.billboards:
            bbx = bx + b["x"]
            bby = by + b["y"]
            col = b["color"]
            sec_col = b.get("secondary_color", col)

            alpha = 255
            text_alpha = 255
            scroll_x = 0

            # Glitch state tracking
            is_glitching = False
            glitch_offset_x = 0
            chromatic_offset = 0

            # Check if this is an image billboard
            is_image_billboard = b["type"].startswith("image_")

            if b["type"] == "pulse":
                factor = (math.sin(t * 1.0) + 1) * 0.5
                alpha = int(120 + factor * 135)
                text_alpha = int(180 + factor * 75)
            elif b["type"] == "scroll":
                scroll_x = int((t * 5 + b.get("scroll_offset", 0)) % (b["w"] + 10)) - 5
                alpha = 200
            elif b["type"] == "flicker":
                if random.random() > 0.995:
                    alpha = random.randint(50, 150)
                else:
                    alpha = 220
                text_alpha = alpha
            elif b["type"] == "glitch":
                alpha = 220
                text_alpha = 255
                # Rare glitch triggers (very occasional)
                glitch_seed = t * 0.3 + b.get("glitch_offset", 0)
                if math.sin(glitch_seed * 1.5) > 0.96 or random.random() > 0.998:
                    is_glitching = True
                    chromatic_offset = random.randint(1, 2)
                    glitch_offset_x = random.randint(-2, 2)
            elif b["type"] == "rainbow":
                alpha = 220
                text_alpha = 255
                # Cycle hue over time
                hue_shift = (t * 0.5 + b.get("scroll_offset", 0) * 0.01) % 1.0
                # Convert hue to RGB (simplified HSV to RGB)
                h = hue_shift * 6
                x_val = 1 - abs(h % 2 - 1)
                if h < 1:
                    col = rl.Color(255, int(x_val * 255), 0, 255)
                elif h < 2:
                    col = rl.Color(int(x_val * 255), 255, 0, 255)
                elif h < 3:
                    col = rl.Color(0, 255, int(x_val * 255), 255)
                elif h < 4:
                    col = rl.Color(0, int(x_val * 255), 255, 255)
                elif h < 5:
                    col = rl.Color(int(x_val * 255), 0, 255, 255)
                else:
                    col = rl.Color(255, 0, int(x_val * 255), 255)
            elif b["type"] == "video":
                alpha = 230
                text_alpha = 255
            elif b["type"] == "image_pulse":
                factor = (math.sin(t * 1.0) + 1) * 0.5
                alpha = int(150 + factor * 105)
            elif b["type"] == "image_flicker":
                if random.random() > 0.995:
                    alpha = random.randint(80, 180)
                else:
                    alpha = 240
            elif b["type"] == "image_glitch":
                alpha = 240
                # Rare glitch triggers
                glitch_seed = t * 0.3 + b.get("glitch_offset", 0)
                if math.sin(glitch_seed * 1.5) > 0.96 or random.random() > 0.998:
                    is_glitching = True
                    chromatic_offset = random.randint(1, 2)
                    glitch_offset_x = random.randint(-2, 2)

            # Draw billboard background and glow
            glow_col = rl.Color(col.r, col.g, col.b, alpha // 3)
            rl.draw_rectangle(bbx - 3, bby - 3, b["w"] + 6, b["h"] + 6, glow_col)
            rl.draw_rectangle(bbx, bby, b["w"], b["h"], rl.Color(10, 5, 15, 240))

            if is_image_billboard:
                # Draw image billboard
                texture = b.get("texture")
                if texture and texture.id > 0:
                    # Calculate scale to fit billboard while maintaining aspect ratio
                    tex_w = texture.width
                    tex_h = texture.height
                    scale_w = (b["w"] - 2) / tex_w
                    scale_h = (b["h"] - 2) / tex_h
                    scale = min(scale_w, scale_h)

                    draw_w = int(tex_w * scale)
                    draw_h = int(tex_h * scale)
                    # Center the image in the billboard
                    draw_x = bbx + (b["w"] - draw_w) // 2
                    draw_y = bby + (b["h"] - draw_h) // 2

                    # Source and destination rectangles
                    src_rect = rl.Rectangle(0, 0, tex_w, tex_h)
                    dst_rect = rl.Rectangle(draw_x, draw_y, draw_w, draw_h)

                    if is_glitching and chromatic_offset > 0:
                        # Chromatic aberration for image - draw tinted copies offset
                        # Red channel (offset left)
                        rl.draw_texture_pro(
                            texture,
                            src_rect,
                            rl.Rectangle(
                                draw_x - chromatic_offset, draw_y, draw_w, draw_h
                            ),
                            rl.Vector2(0, 0),
                            0,
                            rl.Color(255, 0, 0, 100),
                        )
                        # Blue channel (offset right)
                        rl.draw_texture_pro(
                            texture,
                            src_rect,
                            rl.Rectangle(
                                draw_x + chromatic_offset, draw_y, draw_w, draw_h
                            ),
                            rl.Vector2(0, 0),
                            0,
                            rl.Color(0, 0, 255, 100),
                        )
                        # Main image with glitch offset
                        rl.draw_texture_pro(
                            texture,
                            src_rect,
                            rl.Rectangle(
                                draw_x + glitch_offset_x, draw_y, draw_w, draw_h
                            ),
                            rl.Vector2(0, 0),
                            0,
                            rl.Color(255, 255, 255, alpha),
                        )
                        # Draw tear lines
                        for _ in range(random.randint(1, 3)):
                            tear_y = bby + random.randint(0, b["h"])
                            tear_offset = random.randint(-4, 4)
                            rl.draw_rectangle(
                                bbx + tear_offset,
                                tear_y,
                                b["w"],
                                1,
                                rl.Color(col.r, col.g, col.b, 120),
                            )
                    else:
                        # Normal image rendering with neon tint
                        tint = rl.Color(
                            min(255, 180 + col.r // 4),
                            min(255, 180 + col.g // 4),
                            min(255, 180 + col.b // 4),
                            alpha,
                        )
                        rl.draw_texture_pro(
                            texture, src_rect, dst_rect, rl.Vector2(0, 0), 0, tint
                        )
            else:
                # Text billboard rendering
                # Get the text pattern to draw
                if b["type"] == "video":
                    # Video type switches between patterns
                    video_period = 2.0
                    frame_idx = int(t / video_period) % 3
                    patterns = [b.get("text_pattern", [])] + b.get("extra_patterns", [])
                    text_pattern = (
                        patterns[frame_idx]
                        if frame_idx < len(patterns)
                        else b.get("text_pattern", [])
                    )
                else:
                    text_pattern = b.get("text_pattern", [])

                # Draw text with glitch effects if active
                if is_glitching and chromatic_offset > 0:
                    # Chromatic aberration - draw RGB channels offset
                    for px, py in text_pattern:
                        draw_x = bbx + px + scroll_x + glitch_offset_x
                        if bbx <= draw_x < bbx + b["w"] - 1:
                            # Red channel (offset left)
                            rl.draw_rectangle(
                                draw_x - chromatic_offset,
                                bby + py,
                                1,
                                1,
                                rl.Color(255, 0, 0, 150),
                            )
                            # Green channel (center)
                            rl.draw_rectangle(
                                draw_x, bby + py, 1, 1, rl.Color(0, 255, 0, 150)
                            )
                            # Blue channel (offset right)
                            rl.draw_rectangle(
                                draw_x + chromatic_offset,
                                bby + py,
                                1,
                                1,
                                rl.Color(0, 0, 255, 150),
                            )

                    # Draw some horizontal tear lines
                    for _ in range(random.randint(1, 3)):
                        tear_y = bby + random.randint(0, b["h"])
                        tear_offset = random.randint(-6, 6)
                        rl.draw_rectangle(
                            bbx + tear_offset,
                            tear_y,
                            b["w"],
                            1,
                            rl.Color(col.r, col.g, col.b, 100),
                        )
                else:
                    # Normal text rendering
                    text_col = rl.Color(col.r, col.g, col.b, text_alpha)
                    for px, py in text_pattern:
                        draw_x = bbx + px + scroll_x
                        if bbx <= draw_x < bbx + b["w"] - 1:
                            rl.draw_rectangle(draw_x, bby + py, 1, 1, text_col)

            # Draw border lines
            rl.draw_rectangle(
                bbx, bby, b["w"], 1, rl.Color(sec_col.r, sec_col.g, sec_col.b, alpha)
            )
            rl.draw_rectangle(
                bbx,
                bby + b["h"] - 1,
                b["w"],
                1,
                rl.Color(sec_col.r, sec_col.g, sec_col.b, alpha // 2),
            )
            rl.draw_rectangle_lines(bbx, bby, b["w"], b["h"], rl.Color(40, 30, 50, 200))


class City:
    def __init__(self):
        self.layers = [[], [], []]  # Back, Mid, Front
        self.billboard_textures = []  # Shared textures for image billboards

        # Load billboard ad textures
        self._load_billboard_textures()

        for layer in range(3):
            self.generate_layer(layer)

    def _load_billboard_textures(self):
        """Load all billboard ad images as textures."""
        import os

        for path in BILLBOARD_AD_IMAGES:
            if os.path.exists(path):
                texture = rl.load_texture(path)
                if texture.id > 0:
                    self.billboard_textures.append(texture)

    def unload_textures(self):
        """Unload all billboard textures - call before closing."""
        for texture in self.billboard_textures:
            rl.unload_texture(texture)

    def generate_layer(self, layer_idx):
        cx = -20
        while cx < VIRTUAL_WIDTH + 20:
            w = random.randint(30, 80)

            if layer_idx == 0:
                h = random.randint(140, 220)
            elif layer_idx == 1:
                h = random.randint(90, 170)
            else:
                h = random.randint(50, 110)
                w = random.randint(20, 50)

            self.layers[layer_idx].append(
                Building(cx, w, h, layer_idx, self.billboard_textures)
            )

            overlap = random.randint(-15, -2)
            cx += w + overlap

    def update(self):
        pass  # Static city

    def draw(self):
        for b in self.layers[0]:
            b.draw(0)

        rl.draw_rectangle(0, 0, VIRTUAL_WIDTH, VIRTUAL_HEIGHT, C_HAZE_BACK)

        for b in self.layers[1]:
            b.draw(0)

        rl.draw_rectangle(0, 0, VIRTUAL_WIDTH, VIRTUAL_HEIGHT, C_HAZE_MID)

        for b in self.layers[2]:
            b.draw(0)
