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
        mode.value |= 4  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        kernel32.SetConsoleMode(hStdOut, mode)


# --- Configuration ---
FPS = 60
CUBE_SIZE = 1.0  # Back to standard size
SPEED = 1.0

# Rotation speeds
ROT_SPEED_X = 0.4
ROT_SPEED_Y = 0.6
ROT_SPEED_Z = 0.2

# Camera
# Moving camera far back reduces perspective distortion (fisheye effect)
# making the cube look more "solid" and less like it's stretching.
CAMERA_DIST = 20.0
FOV = 10.0

# --- Constants ---
# Upper Half Block
# Used for sub-pixel rendering:
# Top sub-pixel is Foreground color
# Bottom sub-pixel is Background color
PIXEL_CHAR = "▀"
RESET = "\033[0m"


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


# --- 3D Geometry ---

# Cube vertices (x, y, z)
VERTICES = (
    np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],  # Back face
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],  # Front face
        ],
        dtype=float,
    )
    * CUBE_SIZE
)

# Triangles (v0, v1, v2) - Normalized to Standard CCW Winding for "Outside" faces
TRIANGLES = np.array(
    [
        # Front (+Z): Normal (0,0,-1)
        [4, 6, 5],
        [4, 7, 6],
        # Back (-Z): Normal (0,0,1)
        [1, 0, 3],
        [1, 3, 2],
        # Top (+Y): Normal (0,1,0)
        [7, 6, 2],
        [7, 2, 3],
        # Bottom (-Y): Normal (0,-1,0)
        [0, 1, 5],
        [0, 5, 4],
        # Right (+X): Normal (1,0,0)
        [5, 6, 2],
        [5, 2, 1],
        # Left (-X): Normal (-1,0,0)
        [4, 0, 3],
        [4, 3, 7],
    ]
)

# Colors for each face (one color per triangle pair)
# Orange, Red, Green, Blue, Yellow, Magenta
FACE_COLORS = np.array(
    [
        [255, 140, 0],  # Front (Orange)
        [255, 140, 0],
        [200, 0, 0],  # Back (Red)
        [200, 0, 0],
        [0, 200, 0],  # Top (Green)
        [0, 200, 0],
        [0, 0, 200],  # Bottom (Blue)
        [0, 0, 200],
        [255, 255, 0],  # Right (Yellow)
        [255, 255, 0],
        [255, 0, 255],  # Left (Magenta)
        [255, 0, 255],
    ]
)


def rotate(vertices, t):
    # Combined rotation matrix
    a_x = t * ROT_SPEED_X
    a_y = t * ROT_SPEED_Y
    a_z = t * ROT_SPEED_Z

    # X
    cx, sx = np.cos(a_x), np.sin(a_x)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

    # Y
    cy, sy = np.cos(a_y), np.sin(a_y)
    Ry = np.array([[cy, 0, sx], [0, 1, 0], [-sy, 0, cy]])

    # Z
    cz, sz = np.cos(a_z), np.sin(a_z)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    # R = Rz @ Ry @ Rx
    R = Rz @ (Ry @ Rx)

    return vertices @ R.T


def project(vertices, width, height):
    # Perspective projection
    # Camera is at (0, 0, -CAMERA_DIST) looking at (0,0,0)
    # We move vertices relative to camera
    v_cam = vertices.copy()
    v_cam[:, 2] += CAMERA_DIST

    # Avoid div by zero
    v_cam[:, 2] = np.maximum(v_cam[:, 2], 0.1)

    factor = FOV / v_cam[:, 2]

    # Project to 2D
    x_proj = v_cam[:, 0] * factor
    y_proj = v_cam[:, 1] * factor

    # Screen coordinates
    # Note: height is the SUB-PIXEL height (rows * 2)
    # Aspect ratio correction: Terminal chars are ~2:1, but we use sub-pixels
    # which are ~1:1. So we map directly to square pixels.

    screen_x = (x_proj * (height * 0.5) + width * 0.5).astype(int)
    screen_y = (-y_proj * (height * 0.5) + height * 0.5).astype(int)

    return np.stack([screen_x, screen_y, v_cam[:, 2]], axis=1)


def rasterize_triangle(buffer_color, buffer_depth, v0, v1, v2, color, light_intensity):
    # v = [x, y, z]
    # Bounding box
    min_x = max(0, int(min(v0[0], v1[0], v2[0])))
    max_x = min(buffer_color.shape[1] - 1, int(max(v0[0], v1[0], v2[0])))
    min_y = max(0, int(min(v0[1], v1[1], v2[1])))
    max_y = min(buffer_color.shape[0] - 1, int(max(v0[1], v1[1], v2[1])))

    if min_x > max_x or min_y > max_y:
        return

    # Create grid
    # We use numpy broadcasting to compute barycentric coords for the whole box at once
    # Edge functions
    # Area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x)

    # Points P(px, py)
    py, px = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]

    # Edge 1: v0 -> v1
    edge0 = (v1[0] - v0[0]) * (py - v0[1]) - (v1[1] - v0[1]) * (px - v0[0])
    # Edge 2: v1 -> v2
    edge1 = (v2[0] - v1[0]) * (py - v1[1]) - (v2[1] - v1[1]) * (px - v1[0])
    # Edge 3: v2 -> v0
    edge2 = (v0[0] - v2[0]) * (py - v2[1]) - (v0[1] - v2[1]) * (px - v2[0])

    # Check if point is inside triangle (all edges >= 0 or all <= 0 depending on winding)
    # We assume consistent winding. Let's handle both for safety or just >= 0.
    # Actually, cross product logic:
    # If winding is counter-clockwise: edges >= 0
    # Our winding might vary, let's just check if all signs match.

    # Ideally we backface cull before this function, so we know winding is correct.
    # Let's assume edges >= 0 (or close to it for integer precision issues)
    mask = (edge0 >= 0) & (edge1 >= 0) & (edge2 >= 0)

    if not np.any(mask):
        return

    # Interpolate Z (Barycentric)
    # Area of full triangle
    area = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
    if abs(area) < 1.0:
        return

    # Weights
    w0 = edge1 / area
    w1 = edge2 / area
    w2 = edge0 / area

    # Depth Z at each pixel
    z_interp = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]

    # Z-Buffer Test
    # Only update pixels where mask is True AND z < current_z
    current_z = buffer_depth[min_y : max_y + 1, min_x : max_x + 1]
    update_mask = mask & (z_interp < current_z)

    if not np.any(update_mask):
        return

    # Apply update
    # Update Depth
    buffer_depth[min_y : max_y + 1, min_x : max_x + 1] = np.where(
        update_mask, z_interp, current_z
    )

    # Update Color
    # Calculate lit color
    r, g, b = color
    # Simple ambient + diffuse
    lit_factor = 0.2 + 0.8 * light_intensity
    final_color = np.array(
        [r * lit_factor, g * lit_factor, b * lit_factor], dtype=np.uint8
    )

    # Broadcast color to shape of update region
    region_color = buffer_color[min_y : max_y + 1, min_x : max_x + 1]
    # We need to set RGB values where update_mask is true
    # Numpy advanced indexing
    region_color[update_mask] = final_color
    buffer_color[min_y : max_y + 1, min_x : max_x + 1] = region_color


def main():
    enable_windows_ansi()
    clear_screen()
    hide_cursor()

    cols, rows = get_terminal_size()
    rows -= 1

    # Resolution:
    # Width = cols
    # Height = rows * 2 (Sub-pixels)
    render_w = cols
    render_h = rows * 2

    t_start = time.time()

    try:
        while True:
            # Handle Resize
            new_cols, new_rows = get_terminal_size()
            new_rows -= 1
            if new_cols != cols or new_rows != rows:
                cols, rows = new_cols, new_rows
                render_w = cols
                render_h = rows * 2
                clear_screen()

            t = (time.time() - t_start) * SPEED

            # 1. Transform Vertices
            curr_verts = rotate(VERTICES, t)
            proj_verts = project(curr_verts, render_w, render_h)

            # 2. Lighting
            # Rotating light source
            light_x = np.sin(t * 1.5)
            light_y = np.cos(t * 0.5)
            light_z = -1.0
            light_dir = np.array([light_x, light_y, light_z])
            light_dir /= np.linalg.norm(light_dir)

            # 3. Rasterization Buffers
            # Color buffer: (H, W, 3) uint8
            # Depth buffer: (H, W) float
            # Initialize with background color (dark blue-ish)
            bg_color = np.array([10, 10, 20], dtype=np.uint8)
            color_buf = np.full((render_h, render_w, 3), bg_color, dtype=np.uint8)
            depth_buf = np.full((render_h, render_w), np.inf, dtype=float)

            # 4. Render Triangles
            for i in range(len(TRIANGLES)):
                tri = TRIANGLES[i]

                # Projected Screen Points (2D)
                # p = [screen_x, screen_y, depth]
                p0 = proj_verts[tri[0]]
                p1 = proj_verts[tri[1]]
                p2 = proj_verts[tri[2]]

                # 2D Backface Culling (Signed Area)
                # Area = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
                # Y is flipped in screen space (down is +), so typical CCW logic reverses.
                # Winding is normalized to CCW in 3D.
                # Let's test sign.
                area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (
                    p2[0] - p0[0]
                )

                # If area is negative (or positive?), cull.
                # Trial: Cull if <= 0 (Clockwise on screen)
                if area <= 0:
                    continue

                # Lighting Normal (3D)
                v0 = curr_verts[tri[0]]
                v1 = curr_verts[tri[1]]
                v2 = curr_verts[tri[2]]
                edge_a = v1 - v0
                edge_b = v2 - v0
                normal = np.cross(edge_a, edge_b)
                normal /= np.linalg.norm(normal) + 1e-6

                intensity = max(0.0, np.dot(normal, -light_dir))

                rasterize_triangle(
                    color_buf, depth_buf, p0, p1, p2, FACE_COLORS[i], intensity
                )

            # 5. Output Processing (Sub-pixel rendering)
            # We need to collapse render_h (2*rows) -> rows
            # Top pixel -> FG color
            # Bottom pixel -> BG color
            # Char -> Upper Half Block

            # Split buffer into top and bottom rows
            # rows 0, 2, 4... are top
            # rows 1, 3, 5... are bottom
            top_rows = color_buf[0::2]
            bot_rows = color_buf[1::2]

            # Ensure shapes match (handle odd height if resize glitched)
            min_h = min(top_rows.shape[0], bot_rows.shape[0])
            top_rows = top_rows[:min_h]
            bot_rows = bot_rows[:min_h]

            # Vectorized String Construction
            # We need to build ANSI strings:
            # \033[38;2;r;g;bm (FG) + \033[48;2;r;g;bm (BG) + ▀

            # It's faster to do this in list comprehension or numpy string ops
            # Let's use numpy string ops for speed

            def make_ansi_fg(rgb):
                return np.char.add(
                    np.char.add("\033[38;2;", rgb[:, :, 0].astype(str)),
                    np.char.add(
                        ";",
                        np.char.add(
                            rgb[:, :, 1].astype(str),
                            np.char.add(
                                ";", np.char.add(rgb[:, :, 2].astype(str), "m")
                            ),
                        ),
                    ),
                )

            def make_ansi_bg(rgb):
                return np.char.add(
                    np.char.add("\033[48;2;", rgb[:, :, 0].astype(str)),
                    np.char.add(
                        ";",
                        np.char.add(
                            rgb[:, :, 1].astype(str),
                            np.char.add(
                                ";", np.char.add(rgb[:, :, 2].astype(str), "m")
                            ),
                        ),
                    ),
                )

            fgs = make_ansi_fg(top_rows)
            bgs = make_ansi_bg(bot_rows)

            # Combine
            # Line = FG + BG + Char + RESET
            # We can skip reset per char if we just overwrite,
            # but safer to carry colors.
            # Optimization: Join FG+BG+Char
            pixels = np.char.add(fgs, bgs)
            pixels = np.char.add(pixels, PIXEL_CHAR)

            # Add newlines at end of rows
            # Join pixels in each row
            # We can't use np.apply_along_axis easily for string join.
            # Convert to python list of strings is usually fast enough for 60 rows

            frame_lines = []
            for row in pixels:
                frame_lines.append("".join(row))

            frame_str = (RESET + "\n").join(frame_lines) + RESET

            move_cursor(1, 1)
            sys.stdout.buffer.write(frame_str.encode("utf-8"))
            sys.stdout.flush()

            # FPS Lock
            elapsed = time.time() - (t / SPEED + t_start)
            if elapsed < 1.0 / FPS:
                time.sleep(1.0 / FPS - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(RESET)
        clear_screen()
        show_cursor()
        print("Hyper-Cube ended.")


if __name__ == "__main__":
    main()
