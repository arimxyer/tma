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
FACE_COLORS = [
    [200, 50, 50],  # Red
    [50, 200, 50],  # Green
    [50, 50, 200],  # Blue
    [200, 200, 50],  # Yellow
    [200, 50, 200],  # Magenta
    [50, 200, 200],  # Cyan
]


def rotate_y(verts, angle):
    c, s = np.cos(angle), np.sin(angle)
    # Rotate around Y axis
    # x' = x*c + z*s
    # z' = -x*s + z*c
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


def rotate_z(verts, angle):
    c, s = np.cos(angle), np.sin(angle)
    # Rotate around Z axis
    v_new = verts.copy()
    v_new[:, 0] = verts[:, 0] * c - verts[:, 1] * s
    v_new[:, 1] = verts[:, 0] * s + verts[:, 1] * c
    return v_new


def project(verts, width, height):
    # Camera Transform: Move world away from camera
    # Camera is at (0,0,0), Object moved to (0,0,CAMERA_DIST)
    # We want standard look-down-Z
    v_cam = verts.copy()
    v_cam[:, 2] += CAMERA_DIST

    # Perspective Division
    # x' = x / z
    factor = FOV / np.maximum(v_cam[:, 2], 0.1)

    x_proj = v_cam[:, 0] * factor
    y_proj = v_cam[:, 1] * factor

    # Map to Screen
    # Invert Y because screen Y grows downwards
    screen_x = x_proj * (height * 0.5) + width * 0.5
    screen_y = -y_proj * (height * 0.5) + height * 0.5

    return np.stack([screen_x, screen_y, v_cam[:, 2]], axis=1)


def rasterize_quad(buffer_color, buffer_depth, p, color, z_depth):
    # p is 4x2 array of screen points
    # Simple bounding box rasterization
    min_x = max(0, int(np.min(p[:, 0])))
    max_x = min(buffer_color.shape[1] - 1, int(np.max(p[:, 0])))
    min_y = max(0, int(np.min(p[:, 1])))
    max_y = min(buffer_color.shape[0] - 1, int(np.max(p[:, 1])))

    if min_x > max_x or min_y > max_y:
        return

    # Check against Z-buffer for the whole quad (flat shading assumption for speed)
    # Using the average Z depth of the face
    # We will just overwrite if z_depth < current_z
    # But for robustness, let's use per-pixel check?
    # Actually, standard Painter's algo (draw back to front) doesn't need Z-buffer check if sorted correctly.
    # BUT, intersecting geometries need it. A cube doesn't intersect itself.
    # So we can just blast the pixels.

    # Point-in-polygon test (Crossing Number or Edge functions)
    # Since it's convex (quad), simpler logic applies.
    # Let's use matplotlib path logic or edge functions.
    # Edge functions are fast vectorized.

    # Split quad into two triangles: 0-1-2 and 0-2-3
    tris = [[0, 1, 2], [0, 2, 3]]

    for tri_idx in tris:
        v0, v1, v2 = p[tri_idx[0]], p[tri_idx[1]], p[tri_idx[2]]

        # Grid
        Y, X = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]

        # Edge function: (px - v0x)*(v1y - v0y) - (py - v0y)*(v1x - v0x)
        def edge(a, b, px, py):
            return (px - a[0]) * (b[1] - a[1]) - (py - a[1]) * (b[0] - a[0])

        w0 = edge(v1, v2, X, Y)
        w1 = edge(v2, v0, X, Y)
        w2 = edge(v0, v1, X, Y)

        # If all >= 0 or all <= 0 (handle winding)
        mask = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))

        if not np.any(mask):
            continue

        # Draw
        # Apply color
        # Expand color to mask shape
        c = np.array(color, dtype=np.uint8)

        # Only update buffer where mask is True
        region = buffer_color[min_y : max_y + 1, min_x : max_x + 1]
        region[mask] = c
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
            # Resize
            new_cols, new_rows = get_terminal_size()
            new_rows -= 1
            if new_cols != cols or new_rows != rows:
                cols, rows = new_cols, new_rows
                clear_screen()

            # Sub-pixel buffer (2x height)
            buf_h = rows * 2
            buf_w = cols
            buffer_color = np.full((buf_h, buf_w, 3), 15, dtype=np.uint8)  # Dark BG
            # buffer_depth not strictly needed for Painter's algo on a single convex hull

            t = (time.time() - t_start) * SPEED

            # Rotation
            # Combine rotations
            curr_verts = VERTICES
            curr_verts = rotate_x(curr_verts, t * 0.5)
            curr_verts = rotate_y(curr_verts, t * 0.7)

            # Project
            proj_verts = project(curr_verts, buf_w, buf_h)

            # Sort Faces by Depth (Painter's Algorithm)
            # Calculate average Z of each face
            face_depths = []
            for i, face in enumerate(FACES):
                z_avg = np.mean(proj_verts[face, 2])
                face_depths.append((z_avg, i))

            # Sort: Furthest first (High Z -> Low Z)
            # Since Z increases away from camera in our system?
            # Verts at -1..1. Camera shift +30. Z is ~29..31.
            # Larger Z is further away.
            face_depths.sort(key=lambda x: x[0], reverse=True)

            # Render
            for z, idx in face_depths:
                # Get projected 2D points
                pts = proj_verts[FACES[idx], :2]

                # Lighting (Simple flat shading based on normal)
                # Recompute Normal in World Space
                v0 = curr_verts[FACES[idx][0]]
                v1 = curr_verts[FACES[idx][1]]
                v2 = curr_verts[FACES[idx][2]]
                normal = np.cross(v1 - v0, v2 - v0)
                normal /= np.linalg.norm(normal) + 1e-6

                # Light Source (Static or moving)
                light = np.array([0.5, 0.5, -1.0])
                light /= np.linalg.norm(light)

                # Dot
                intensity = np.dot(normal, -light)
                # Remap -1..1 to 0.2..1.0
                intensity = max(0.2, (intensity + 0.5))

                base_c = np.array(FACE_COLORS[idx])
                final_c = (base_c * intensity).astype(int)
                final_c = np.clip(final_c, 0, 255)

                rasterize_quad(buffer_color, None, pts, final_c, z)

            # Convert Buffer to ANSI
            # Split Top/Bot
            top = buffer_color[0::2]
            bot = buffer_color[1::2]

            # Trim if needed
            min_h = min(len(top), len(bot))
            top = top[:min_h]
            bot = bot[:min_h]

            # Construct strings
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
        print("Done.")


if __name__ == "__main__":
    main()
