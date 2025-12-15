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
FPS = 60
SPEED = 1.0
CUBE_SIZE = 1.5
CAMERA_DIST = 30.0  # Far away (Telephoto)
FOV = 18.0  # Zoomed in

# Colors
RESET = "\033[0m"
PIXEL_CHAR = "▀"  # Half-block for 2x vertical resolution


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
# 8 Vertices of a Cube
VERTICES = (
    np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],  # Back Z=-1
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],  # Front Z=1
        ],
        dtype=float,
    )
    * CUBE_SIZE
)

# 6 Faces (Quad indices)
FACES = [
    [0, 1, 2, 3],  # Back
    [4, 5, 6, 7],  # Front
    [0, 4, 7, 3],  # Left
    [1, 5, 6, 2],  # Right
    [3, 7, 6, 2],  # Top
    [0, 4, 5, 1],  # Bottom
]

# Colors (R, G, B)
# ALL ORANGE - Varied slightly for distinction if needed,
# but rely on lighting for main differentiation.
# Deep Orange to Bright Orange
BASE_ORANGE = [255, 120, 0]

# Generate slight variations so faces aren't identical if lighting is flat
FACE_COLORS = [
    [255, 100, 0],
    [255, 120, 0],
    [255, 140, 0],
    [255, 110, 0],
    [255, 130, 0],
    [255, 150, 0],
]


def rotate_y(verts, angle):
    c, s = np.cos(angle), np.sin(angle)
    # Rotate around Y axis
    v_new = verts.copy()
    v_new[:, 0] = verts[:, 0] * c + verts[:, 2] * s
    v_new[:, 2] = -verts[:, 0] * s + verts[:, 2] * c
    return v_new


def rotate_x(verts, angle):
    c, s = np.cos(angle), np.sin(angle)
    # Rotate around X axis
    v_new = verts.copy()
    v_new[:, 1] = verts[:, 1] * c - verts[:, 2] * s
    v_new[:, 2] = verts[:, 1] * s + verts[:, 2] * c
    return v_new


def project(verts, width, height):
    # Camera Transform
    v_cam = verts.copy()
    v_cam[:, 2] += CAMERA_DIST

    # Perspective
    factor = FOV / np.maximum(v_cam[:, 2], 0.1)

    x_proj = v_cam[:, 0] * factor
    y_proj = v_cam[:, 1] * factor

    # Map to Screen
    screen_x = x_proj * (height * 0.5) + width * 0.5
    screen_y = -y_proj * (height * 0.5) + height * 0.5

    return np.stack([screen_x, screen_y, v_cam[:, 2]], axis=1)


def rasterize_quad_matrix(buffer_color, p, color, t):
    # p is 4x2 array of screen points
    min_x = max(0, int(np.min(p[:, 0])))
    max_x = min(buffer_color.shape[1] - 1, int(np.max(p[:, 0])))
    min_y = max(0, int(np.min(p[:, 1])))
    max_y = min(buffer_color.shape[0] - 1, int(np.max(p[:, 1])))

    if min_x > max_x or min_y > max_y:
        return

    # Split quad into two triangles
    tris = [[0, 1, 2], [0, 2, 3]]

    for tri_idx in tris:
        v0, v1, v2 = p[tri_idx[0]], p[tri_idx[1]], p[tri_idx[2]]

        # Grid for Barycentric check
        Y, X = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]

        def edge(a, b, px, py):
            return (px - a[0]) * (b[1] - a[1]) - (py - a[1]) * (b[0] - a[0])

        w0 = edge(v1, v2, X, Y)
        w1 = edge(v2, v0, X, Y)
        w2 = edge(v0, v1, X, Y)

        mask = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))

        if not np.any(mask):
            continue

        # Draw with "Matrix" texture
        # We want to modulate the color based on a scrolling code pattern or grid
        # Simple scanline / matrix grid effect:
        # Check (X + Y) or (X - Y) pattern

        # Scrolling vertical lines:
        # (X + int(t * 10)) % 4 == 0

        # Or simple grid lines
        grid_mask = ((X + Y) % 3 == 0) | ((X - Y + int(t * 20)) % 6 == 0)

        # Where grid_mask is True, we use bright color.
        # Where False, we use dim color (or black).

        c_bright = np.array(color, dtype=np.uint8)
        c_dim = (c_bright * 0.2).astype(np.uint8)  # Faded background

        region = buffer_color[min_y : max_y + 1, min_x : max_x + 1]

        # Apply texture
        # Initialize region canvas for this poly
        poly_region = np.zeros_like(region)

        # Logic:
        # If mask is True:
        #   If grid_mask is True -> Bright
        #   Else -> Dim

        # We need to broadcast grid_mask to 3 channels? No, just assign.

        # Update only masked pixels
        # Create a textured color block
        texture = np.where(grid_mask[..., None], c_bright, c_dim)

        region[mask] = texture[mask]
        buffer_color[min_y : max_y + 1, min_x : max_x + 1] = region


def main():
    enable_windows_ansi()
    clear_screen()
    hide_cursor()

    cols, rows = get_terminal_size()
    rows -= 1

    t_start = time.time()

    try:
        while True:
            new_cols, new_rows = get_terminal_size()
            new_rows -= 1
            if new_cols != cols or new_rows != rows:
                cols, rows = new_cols, new_rows
                clear_screen()

            buf_h = rows * 2
            buf_w = cols
            # Black background
            buffer_color = np.zeros((buf_h, buf_w, 3), dtype=np.uint8)

            t = (time.time() - t_start) * SPEED

            # Rotate
            curr_verts = VERTICES
            curr_verts = rotate_x(curr_verts, t * 0.5)
            curr_verts = rotate_y(curr_verts, t * 0.7)

            # Project
            proj_verts = project(curr_verts, buf_w, buf_h)

            # Sort Faces (Painter's Algo)
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
                intensity = max(0.4, (intensity + 0.5))  # Minimum brightness for glow

                base_c = np.array(FACE_COLORS[idx])
                final_c = (base_c * intensity).astype(int)
                final_c = np.clip(final_c, 0, 255)

                # Pass 't' for animation
                rasterize_quad_matrix(buffer_color, pts, final_c, t)

            # Convert to ANSI
            top = buffer_color[0::2]
            bot = buffer_color[1::2]

            min_h = min(len(top), len(bot))
            top = top[:min_h]
            bot = bot[:min_h]

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

            def ansi_bg(rgb):
                return np.char.add(
                    "\033[48;2;",
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

            fgs = ansi_fg(top)
            bgs = ansi_bg(bot)

            chars = np.char.add(fgs, np.char.add(bgs, PIXEL_CHAR))

            lines = ["".join(row) for row in chars]
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
        print("Matrix Cube ended.")


if __name__ == "__main__":
    main()
