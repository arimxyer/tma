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
FPS = 30
SCALE_X = 0.15
SCALE_Y = 0.3  # Vertical compression for perspective
SPEED = 2.0

# Character gradient (from low/deep to high/foam)
CHARS = np.array(list(" .:;~=+*#%@"))
N_CHARS = len(CHARS)


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


# --- Color Functions ---
def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def ansi_color_bg(r, g, b):
    return f"\033[48;2;{r};{g};{b}m"


def ansi_color_fg(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


RESET = "\033[0m"


# Generate color palette (Gradient: Dark Blue -> Teal -> Cyan -> White)
def generate_palette(steps):
    # Keyframes: (position 0.0-1.0, (r, g, b))
    stops = [
        (0.0, (0, 5, 20)),  # Abyss
        (0.4, (0, 40, 100)),  # Deep Blue
        (0.7, (0, 150, 180)),  # Teal
        (0.9, (100, 255, 255)),  # Cyan Foam
        (1.0, (230, 255, 255)),  # Whitecap
    ]

    palette_fg = []
    palette_bg = []

    for i in range(steps):
        t = i / (steps - 1)
        # Find blending interval
        for j in range(len(stops) - 1):
            if stops[j][0] <= t <= stops[j + 1][0]:
                start_t, start_c = stops[j]
                end_t, end_c = stops[j + 1]

                # Interpolate
                local_t = (t - start_t) / (end_t - start_t)
                r = int(start_c[0] + (end_c[0] - start_c[0]) * local_t)
                g = int(start_c[1] + (end_c[1] - start_c[1]) * local_t)
                b = int(start_c[2] + (end_c[2] - start_c[2]) * local_t)

                # Formatting
                # For waves, we color the Foreground (character)
                # and keep Background dark or slightly tinted for depth
                palette_fg.append(ansi_color_fg(r, g, b))
                # Slight blue tint for background to reduce contrast harshness
                bg_r, bg_g, bg_b = r // 10, g // 10, b // 5
                palette_bg.append(ansi_color_bg(bg_r, bg_g, bg_b))
                break

    return np.array(palette_fg), np.array(palette_bg)


def main():
    enable_windows_ansi()
    clear_screen()
    hide_cursor()

    cols, rows = get_terminal_size()
    # Reduce rows slightly to avoid scroll jitter at bottom
    rows -= 1

    # Precompute grid
    x = np.arange(cols) * SCALE_X
    y = np.arange(rows) * SCALE_Y
    X, Y = np.meshgrid(x, y)

    # Precompute palette
    PALETTE_STEPS = 60
    fg_palette, bg_palette = generate_palette(PALETTE_STEPS)

    t_start = time.time()

    try:
        while True:
            # Handle resize dynamically
            new_cols, new_rows = get_terminal_size()
            new_rows -= 1
            if new_cols != cols or new_rows != rows:
                cols, rows = new_cols, new_rows
                x = np.arange(cols) * SCALE_X
                y = np.arange(rows) * SCALE_Y
                X, Y = np.meshgrid(x, y)
                clear_screen()

            t = (time.time() - t_start) * SPEED

            # --- WAVE SIMULATION ---
            # 1. Main Swell (Diagonal)
            Z = np.sin(X * 0.5 + Y * 0.5 + t * 0.5)
            # 2. Cross Chop (Opposing)
            Z += 0.5 * np.sin(X * 0.8 - Y * 0.2 + t * 0.8)
            # 3. Detail Ripples (Faster)
            Z += 0.2 * np.sin(X * 2.0 + t * 2.0)

            # Normalize Z to 0..1 range
            # Theoretical min/max of sum of sines:
            # 1 + 0.5 + 0.2 = 1.7 (max), -1.7 (min)
            Z_norm = (Z + 1.7) / 3.4

            # Clamp for safety
            Z_norm = np.clip(Z_norm, 0, 0.99)

            # --- RENDER ---

            # 1. Map height to indices
            char_indices = (Z_norm * N_CHARS).astype(int)
            color_indices = (Z_norm * PALETTE_STEPS).astype(int)

            # 2. Look up attributes
            # We construct the frame buffer using numpy vectorization for speed

            # Get the strings
            fgs = fg_palette[color_indices]
            bgs = bg_palette[color_indices]
            chars = CHARS[char_indices]

            # Combine: BG + FG + Char
            # This is the heavy string concatenation.
            # Numpy char arrays are fixed width, which is tricky.
            # Instead, we flatten and join in python, or use object arrays.

            # Optimized approach:
            # Flatten everything to 1D
            buffer_data = np.char.add(fgs.flatten(), bgs.flatten())
            buffer_data = np.char.add(buffer_data, chars.flatten())

            # Join all characters
            frame_str = "".join(buffer_data)

            # Reset cursor and print
            move_cursor(1, 1)
            sys.stdout.buffer.write(frame_str.encode("utf-8"))
            sys.stdout.buffer.write(RESET.encode("utf-8"))
            sys.stdout.flush()

            # FPS Cap
            time.sleep(1 / FPS)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(RESET)
        clear_screen()
        show_cursor()
        print("Ocean simulation ended.")


if __name__ == "__main__":
    main()
