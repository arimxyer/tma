# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
# ]
# ///

import sys
import time
import math
import os
import shutil
import ctypes
import numpy as np
import random


# --- Windows ANSI Support ---
def enable_windows_ansi():
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        hStdOut = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(hStdOut, ctypes.byref(mode))
        mode.value |= 4
        kernel32.SetConsoleMode(hStdOut, mode)


# --- Configuration ---
FPS = 30  # Lower FPS fits the "Terminal" retro feel better
SPEED = 0.8
CUBE_SIZE = 1.8
CAMERA_DIST = 55.0  # Zoomed out (was 35.0)
FOV = 22.0

# Colors
RESET = "\033[0m"

# The "Data" Palette - Hex and Tech Glyphs
GLYPHS = list("0123456789ABCDEF<>[]+-=*:.")
N_GLYPHS = len(GLYPHS)


def get_terminal_size():
    cols, rows = shutil.get_terminal_size()
    return cols, rows


def hide_cursor():
    sys.stdout.write("\033[?25l")


def show_cursor():
    sys.stdout.write("\033[?25h")


def move_cursor(x, y):
    sys.stdout.write(f"\033[{y};{x}H")


def clear_screen():
    sys.stdout.write("\033[2J")


# --- Geometry ---
VERTICES = (
    np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    * CUBE_SIZE
)

FACES = [
    [0, 1, 2, 3],  # Back
    [4, 5, 6, 7],  # Front
    [0, 4, 7, 3],  # Left
    [1, 5, 6, 2],  # Right
    [3, 7, 6, 2],  # Top
    [0, 4, 5, 1],  # Bottom
]

# Base HUD Color (Orange)
BASE_COLOR = np.array([255, 140, 0])


def rotate_y(verts, angle):
    c, s = np.cos(angle), np.sin(angle)
    v_new = verts.copy()
    v_new[:, 0] = verts[:, 0] * c + verts[:, 2] * s
    v_new[:, 2] = -verts[:, 0] * s + verts[:, 2] * c
    return v_new


def rotate_x(verts, angle):
    c, s = np.cos(angle), np.sin(angle)
    v_new = verts.copy()
    v_new[:, 1] = verts[:, 1] * c - verts[:, 2] * s
    v_new[:, 2] = verts[:, 1] * s + verts[:, 2] * c
    return v_new


def project(verts, width, height):
    v_cam = verts.copy()
    v_cam[:, 2] += CAMERA_DIST

    factor = FOV / np.maximum(v_cam[:, 2], 0.1)

    x_proj = v_cam[:, 0] * factor
    y_proj = v_cam[:, 1] * factor

    # Standard terminal mapping (approx 2:1 char aspect ratio correction)
    # We multiply Y by 0.5 to squat it down, or X by 2.0?
    # Actually, characters are tall. To make a square look square, we need more X pixels than Y pixels.
    # The previous code handled this by using sub-pixels (perfect square).
    # Here, we use single chars.
    # To fix aspect ratio: Multiply Screen X by 2.0 relative to Y.

    screen_x = x_proj * (height * 1.0) + width * 0.5  # Scale X more?
    screen_y = -y_proj * (height * 0.5) + height * 0.5

    return np.stack([screen_x, screen_y, v_cam[:, 2]], axis=1)


def rasterize_hud_quad(buffer_char, buffer_fg, p, light_intensity, t):
    # p is 4x2 array
    min_x = max(0, int(np.min(p[:, 0])))
    max_x = min(buffer_char.shape[1] - 1, int(np.max(p[:, 0])))
    min_y = max(0, int(np.min(p[:, 1])))
    max_y = min(buffer_char.shape[0] - 1, int(np.max(p[:, 1])))

    if min_x > max_x or min_y > max_y:
        return

    tris = [[0, 1, 2], [0, 2, 3]]

    for tri_idx in tris:
        v0, v1, v2 = p[tri_idx[0]], p[tri_idx[1]], p[tri_idx[2]]

        Y, X = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]

        def edge(a, b, px, py):
            return (px - a[0]) * (b[1] - a[1]) - (py - a[1]) * (b[0] - a[0])

        w0 = edge(v1, v2, X, Y)
        w1 = edge(v2, v0, X, Y)
        w2 = edge(v0, v1, X, Y)

        mask = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))

        if not np.any(mask):
            continue

        # --- HUD TEXTURE GENERATION ---

        # We want the text to look like "Scrolling Data".
        # We can map the character index to (X + Y + Time).

        # Data Stream Effect:
        # stream_idx = (X + Y*width + int(t * speed)) % N_GLYPHS
        # But we need vectorized lookup.

        # Make a random-looking field based on coords
        # (X * 3 + Y * 7 + int(t*10)) % N_GLYPHS

        glyph_indices = (X * 3 + Y * 5 + int(t * 15)) % N_GLYPHS

        # Apply mask
        region_char = buffer_char[min_y : max_y + 1, min_x : max_x + 1]
        region_fg = buffer_fg[min_y : max_y + 1, min_x : max_x + 1]

        # Pick Characters
        # We need to map indices to actual characters.
        # Numpy char array is tricky with arbitrary mapping.
        # But we can just use a lookup string later?
        # No, simpler to store indices or chars directly.
        # Let's store indices in buffer_char (int array) and decode at end.

        region_char[mask] = glyph_indices[mask]

        # Color Calculation
        # Brightness = light_intensity
        # Add some digital noise/variation to brightness
        noise = ((X + Y * 3) % 5) * 0.05
        local_intensity = light_intensity + noise

        # Calculate RGB
        # local_intensity is (H, W), BASE_COLOR is (3,)
        # We need (H, W, 1) * (3,) -> (H, W, 3)
        c = (BASE_COLOR * local_intensity[..., None]).astype(np.uint8)
        c = np.clip(c, 0, 255)

        # Set colors
        # Ensure mask is broadcastable if needed, but usually it matches H,W
        # c is full size (H,W,3). region_fg is (H',W',3). mask is (H',W').
        # region_fg[mask] expects shape (N, 3) where N is number of True in mask.
        # c[mask] extracts exactly that.
        region_fg[mask] = c[mask]

        buffer_char[min_y : max_y + 1, min_x : max_x + 1] = region_char
        buffer_fg[min_y : max_y + 1, min_x : max_x + 1] = region_fg


def main():
    enable_windows_ansi()
    clear_screen()
    hide_cursor()

    cols, rows = get_terminal_size()
    rows -= 1

    t_start = time.time()

    # Pre-compute Glyph lookup for fast rendering
    GLYPH_ARR = np.array(GLYPHS)

    try:
        while True:
            new_cols, new_rows = get_terminal_size()
            new_rows -= 1
            if new_cols != cols or new_rows != rows:
                cols, rows = new_cols, new_rows
                clear_screen()

            # Standard Resolution (No sub-pixels)
            buf_h = rows
            buf_w = cols

            # Buffer for Character Indices (initialized to -1 or space index)
            # Use ' ' (space) as background
            space_idx = len(GLYPHS)  # Special index for empty
            buffer_char_idx = np.full((buf_h, buf_w), space_idx, dtype=int)
            buffer_fg = np.zeros((buf_h, buf_w, 3), dtype=np.uint8)

            t = (time.time() - t_start) * SPEED

            # Rotate
            curr_verts = VERTICES
            curr_verts = rotate_x(curr_verts, t * 0.5)
            curr_verts = rotate_y(curr_verts, t * 0.7)

            # Project
            proj_verts = project(curr_verts, buf_w, buf_h)

            # Sort Faces
            face_depths = []
            for i, face in enumerate(FACES):
                z_avg = np.mean(proj_verts[face, 2])
                face_depths.append((z_avg, i))
            face_depths.sort(key=lambda x: x[0], reverse=True)

            # Render
            for z, idx in face_depths:
                pts = proj_verts[FACES[idx], :2]

                # Lighting
                v0 = curr_verts[FACES[idx][0]]
                v1 = curr_verts[FACES[idx][1]]
                v2 = curr_verts[FACES[idx][2]]
                normal = np.cross(v1 - v0, v2 - v0)
                normal /= np.linalg.norm(normal) + 1e-6

                light = np.array([0.5, 0.5, -1.0])
                light /= np.linalg.norm(light)

                intensity = np.dot(normal, -light)
                intensity = max(0.2, (intensity + 0.5))

                rasterize_hud_quad(buffer_char_idx, buffer_fg, pts, intensity, t)

            # Decode Characters
            # Create a full character map including Space
            full_glyphs = np.concatenate([GLYPH_ARR, [" "]])
            final_chars = full_glyphs[buffer_char_idx]

            # ANSI Colors
            def ansi_fg(rgb):
                return np.char.add(
                    "\033[38;2;",
                    np.char.add(
                        rgb[..., 0].astype(str),
                        np.char.add(
                            ";",
                            np.char.add(
                                rgb[..., 1].astype(str),
                                np.char.add(
                                    ";", np.char.add(rgb[..., 2].astype(str), "m")
                                ),
                            ),
                        ),
                    ),
                )

            fgs = ansi_fg(buffer_fg)

            # Combine: Color + Char
            # Note: We don't use BG color (keep it transparent/black)
            output_grid = np.char.add(fgs, final_chars)

            lines = ["".join(row) for row in output_grid]
            output = (RESET + "\n").join(lines) + RESET

            move_cursor(1, 1)
            sys.stdout.buffer.write(output.encode("utf-8"))
            sys.stdout.flush()

            time.sleep(1 / FPS)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(RESET)
        clear_screen()
        show_cursor()
        print("Terminal Cube ended.")


if __name__ == "__main__":
    main()
