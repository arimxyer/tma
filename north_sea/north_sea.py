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
SPEED = 1.2  # Slightly faster, more urgent feel

# --- Boat Motion Parameters ---
ROLL_AMPLITUDE = 0.08  # Side-to-side tilt
ROLL_PERIOD = 4.5  # Seconds per roll cycle
PITCH_AMPLITUDE = 0.06  # Front-to-back tilt
PITCH_PERIOD = 5.2  # Seconds per pitch cycle
HEAVE_AMPLITUDE = 0.15  # Up-down motion (affects horizon position)
HEAVE_PERIOD = 3.8  # Seconds per heave cycle
YAW_AMPLITUDE = 0.03  # Slight rotation
YAW_PERIOD = 12.0  # Slow yaw drift

# --- Wave Parameters (North Sea - rough, steep, chaotic) ---
WAVE_STEEPNESS = 2.8  # Higher = sharper peaks
BASE_WAVE_HEIGHT = 1.4  # Larger waves
CHOP_INTENSITY = 0.6  # Surface chop amount
CROSS_SWELL_ANGLE = 0.7  # How much crossing swell

# --- Visual Parameters ---
HORIZON_BASE = 0.45  # Horizon closer to middle (you're low in the water)
SPRAY_THRESHOLD = 0.75  # When to show spray/foam
FOV = 2.0  # Field of view

# Character sets for different wave faces
# Rising face (toward viewer), falling face (away), and foam
CHARS_RISE = np.array(list(" .,:;/|[{#@@"))   # Ascending shapes
CHARS_FALL = np.array(list(" .,:;\\|]}#@@"))  # Descending shapes
CHARS_FLAT = np.array(list(" .,~-=~-.,  "))   # Horizontal shapes
CHARS_FOAM = np.array(list("*%#@@@@@@@@@@"))  # Foam/crest
N_CHARS = len(CHARS_RISE)

# Lighting direction (normalized) - light coming from upper-left-behind
LIGHT_DIR_X = -0.3
LIGHT_DIR_Z = -0.8  # Toward viewer
LIGHT_DIR_Y = 0.5   # From above


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
def ansi_color_bg(r, g, b):
    return f"\033[48;2;{r};{g};{b}m"


def ansi_color_fg(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


RESET = "\033[0m"


def generate_north_sea_palette(steps):
    """Cold, gray-green North Sea colors with stormy highlights."""
    stops = [
        (0.0, (5, 10, 15)),       # Deep black-green abyss
        (0.2, (10, 25, 35)),      # Dark cold depth
        (0.4, (20, 45, 55)),      # Mid-depth gray-green
        (0.55, (35, 65, 75)),     # Cold gray-blue
        (0.7, (55, 90, 100)),     # Surface gray
        (0.82, (90, 120, 125)),   # Pale gray-green
        (0.9, (160, 180, 180)),   # Foam gray
        (0.95, (200, 215, 215)),  # Spray
        (1.0, (240, 245, 245)),   # White foam
    ]

    palette_fg = []
    palette_bg = []

    for i in range(steps):
        t = i / (steps - 1)
        for j in range(len(stops) - 1):
            if stops[j][0] <= t <= stops[j + 1][0]:
                start_t, start_c = stops[j]
                end_t, end_c = stops[j + 1]
                local_t = (t - start_t) / (end_t - start_t)
                r = int(start_c[0] + (end_c[0] - start_c[0]) * local_t)
                g = int(start_c[1] + (end_c[1] - start_c[1]) * local_t)
                b = int(start_c[2] + (end_c[2] - start_c[2]) * local_t)

                palette_fg.append(ansi_color_fg(r, g, b))
                # Darker, more saturated background
                bg_r = max(0, r // 8)
                bg_g = max(0, g // 7)
                bg_b = max(0, b // 6)
                palette_bg.append(ansi_color_bg(bg_r, bg_g, bg_b))
                break

    return np.array(palette_fg), np.array(palette_bg)


def generate_stormy_sky_lut(max_rows):
    """Gray, overcast North Sea sky."""
    sky_fgs = []
    sky_bgs = []

    for y in range(max_rows + 10):
        t = y / max(max_rows, 1)
        # Dark gray at top, slightly lighter toward horizon
        r = int(25 + t * 35)
        g = int(30 + t * 40)
        b = int(40 + t * 45)
        sky_bgs.append(ansi_color_bg(r, g, b))
        sky_fgs.append(ansi_color_fg(r, g, b))

    return np.array(sky_fgs), np.array(sky_bgs)


class BoatMotion:
    """Simulates 6DOF boat motion on rough seas."""

    def __init__(self):
        # Use irrational ratios to avoid obvious loops
        self.roll_freq = 2 * math.pi / ROLL_PERIOD
        self.pitch_freq = 2 * math.pi / PITCH_PERIOD
        self.heave_freq = 2 * math.pi / HEAVE_PERIOD
        self.yaw_freq = 2 * math.pi / YAW_PERIOD

        # Secondary frequencies for complexity
        self.roll_freq2 = self.roll_freq * 1.73
        self.pitch_freq2 = self.pitch_freq * 1.41
        self.heave_freq2 = self.heave_freq * 2.17

    def get_motion(self, t):
        """Returns (roll, pitch, heave, yaw) at time t."""

        # Roll - side to side
        roll = (
            math.sin(t * self.roll_freq) * ROLL_AMPLITUDE +
            math.sin(t * self.roll_freq2) * ROLL_AMPLITUDE * 0.3 +
            math.sin(t * self.roll_freq * 0.37) * ROLL_AMPLITUDE * 0.2
        )

        # Pitch - front to back
        pitch = (
            math.sin(t * self.pitch_freq) * PITCH_AMPLITUDE +
            math.sin(t * self.pitch_freq2) * PITCH_AMPLITUDE * 0.25 +
            math.sin(t * self.pitch_freq * 0.53) * PITCH_AMPLITUDE * 0.15
        )

        # Heave - up and down
        heave = (
            math.sin(t * self.heave_freq) * HEAVE_AMPLITUDE +
            math.sin(t * self.heave_freq2) * HEAVE_AMPLITUDE * 0.4 +
            math.sin(t * self.heave_freq * 0.61) * HEAVE_AMPLITUDE * 0.2
        )

        # Yaw - rotation
        yaw = (
            math.sin(t * self.yaw_freq) * YAW_AMPLITUDE +
            math.sin(t * self.yaw_freq * 2.3) * YAW_AMPLITUDE * 0.3
        )

        return roll, pitch, heave, yaw


def compute_north_sea_waves(world_x, world_z, t, normalized_y):
    """
    Compute wave height AND surface normals for North Sea conditions.
    Returns height H and slope components (dH_dx, dH_dz) for lighting.
    """
    H = np.zeros_like(world_x)
    dH_dx = np.zeros_like(world_x)  # Slope in X direction
    dH_dz = np.zeros_like(world_x)  # Slope in Z direction

    # --- Primary Swell (from the northwest) ---
    k1 = 0.25
    dir1_x, dir1_z = 0.3, 0.95
    phase1 = (world_x * dir1_x + world_z * dir1_z) * k1 + t * 0.7
    raw1 = np.sin(phase1)
    cos1 = np.cos(phase1)
    # Steep, aggressive peaks
    h1 = np.sign(raw1) * np.abs(raw1) ** (1.0 / WAVE_STEEPNESS)
    h1 = (h1 + 1) * 0.5
    amp1 = BASE_WAVE_HEIGHT
    H += h1 * amp1
    # Derivative: d/dx of sin(phase) = cos(phase) * d(phase)/dx
    dH_dx += cos1 * k1 * dir1_x * amp1
    dH_dz += cos1 * k1 * dir1_z * amp1

    # --- Secondary Swell (crossing from different angle) ---
    k2 = 0.35
    dir2_x, dir2_z = -0.5 * CROSS_SWELL_ANGLE, 0.85
    phase2 = (world_x * dir2_x + world_z * dir2_z) * k2 + t * 0.55
    raw2 = np.sin(phase2)
    cos2 = np.cos(phase2)
    h2 = np.sign(raw2) * np.abs(raw2) ** (1.0 / (WAVE_STEEPNESS * 0.9))
    h2 = (h2 + 1) * 0.5
    amp2 = BASE_WAVE_HEIGHT * 0.6
    H += h2 * amp2
    dH_dx += cos2 * k2 * dir2_x * amp2
    dH_dz += cos2 * k2 * dir2_z * amp2

    # --- Tertiary Swell (long period, from behind) ---
    k3 = 0.12
    dir3_x, dir3_z = 0.0, 1.0
    phase3 = world_z * k3 - t * 0.3
    cos3 = np.cos(phase3)
    h3 = (np.sin(phase3) + 1) * 0.5
    amp3 = BASE_WAVE_HEIGHT * 0.35
    H += h3 * amp3
    dH_dz += cos3 * k3 * amp3

    # --- Wind Chop (rough surface texture) ---
    k4 = 1.5
    dir4_x, dir4_z = 0.8, 0.6
    phase4 = (world_x * dir4_x + world_z * dir4_z) * k4 + t * 2.2
    cos4 = np.cos(phase4)
    h4 = np.sin(phase4) * CHOP_INTENSITY
    chop_scale = 0.3 + H * 0.4
    H += h4 * chop_scale
    dH_dx += cos4 * k4 * dir4_x * CHOP_INTENSITY * chop_scale
    dH_dz += cos4 * k4 * dir4_z * CHOP_INTENSITY * chop_scale

    # --- Secondary Chop (crossing) ---
    k5 = 2.2
    dir5_x, dir5_z = -0.5, 0.9
    phase5 = (world_x * dir5_x + world_z * dir5_z) * k5 + t * 2.8
    cos5 = np.cos(phase5)
    h5 = np.sin(phase5) * CHOP_INTENSITY * 0.5
    H += h5
    dH_dx += cos5 * k5 * dir5_x * CHOP_INTENSITY * 0.5
    dH_dz += cos5 * k5 * dir5_z * CHOP_INTENSITY * 0.5

    # --- Micro Texture ---
    k6 = 5.0
    dir6_x, dir6_z = 1.1, 0.7
    phase6 = (world_x * dir6_x + world_z * dir6_z) * k6 + t * 4.5
    cos6 = np.cos(phase6)
    h6 = np.sin(phase6) * 0.08
    H += h6
    dH_dx += cos6 * k6 * dir6_x * 0.08
    dH_dz += cos6 * k6 * dir6_z * 0.08

    # --- Random-ish peaks (rogue wave seeds) ---
    rogue_mod = np.sin(world_z * 0.05 + t * 0.08) * np.sin(world_x * 0.03 + t * 0.05)
    rogue_mod = np.maximum(rogue_mod, 0) ** 2
    H += rogue_mod * 0.5

    # Shoaling - waves grow as they approach
    shoal = 0.4 + 0.6 * np.clip(normalized_y, 0, 1)
    H *= shoal
    dH_dx *= shoal
    dH_dz *= shoal

    return H, dH_dx, dH_dz


def main():
    enable_windows_ansi()
    clear_screen()
    hide_cursor()

    cols, rows = get_terminal_size()
    rows -= 1

    # Precompute palette
    PALETTE_STEPS = 60
    fg_palette, bg_palette = generate_north_sea_palette(PALETTE_STEPS)

    # Initial setup
    sx = np.linspace(-1, 1, cols)
    sy = np.arange(rows)
    SX, SY = np.meshgrid(sx, sy)

    # Boat motion simulator
    boat = BoatMotion()

    t_start = time.time()

    try:
        while True:
            # Resize Check
            new_cols, new_rows = get_terminal_size()
            new_rows -= 1
            if new_cols != cols or new_rows != rows:
                cols, rows = new_cols, new_rows
                sx = np.linspace(-1, 1, cols)
                sy = np.arange(rows)
                SX, SY = np.meshgrid(sx, sy)
                clear_screen()

            t = (time.time() - t_start) * SPEED

            # --- BOAT MOTION ---
            roll, pitch, heave, yaw = boat.get_motion(t)

            # Apply roll (tilts horizon left/right)
            # Apply pitch (tilts horizon up/down)
            # Apply heave (moves horizon up/down)

            horizon_base = rows * HORIZON_BASE

            # Heave moves the whole horizon
            horizon_offset = heave * rows * 0.3

            # Roll tilts the horizon based on X position
            roll_offset = SX * roll * rows * 0.4

            # Pitch affects the whole view
            pitch_offset = pitch * rows * 0.2

            # Combined horizon position per pixel
            HORIZON_Y = horizon_base + horizon_offset + roll_offset + pitch_offset

            # Apply slight yaw distortion to X coordinates
            SX_warped = SX + yaw * SY / rows * 0.5

            # Calculate Normalized Y for Water Projection
            water_depth = rows - HORIZON_Y
            normalized_y = (SY - HORIZON_Y) / (water_depth + 0.1)
            is_water = normalized_y >= 0

            # --- WATER PHYSICS ---

            # Projection - lower camera height (you're on a boat, not a pier)
            valid_y = np.maximum(normalized_y, 0.008)
            world_z = 12.0 / valid_y  # Closer view than pier
            world_x = SX_warped * world_z * FOV

            # Add boat motion to world coordinates (you're moving with the waves)
            world_x += math.sin(t * 0.3) * 2
            world_z += t * 0.5  # Slight forward drift

            # Compute waves with slope information
            H, dH_dx, dH_dz = compute_north_sea_waves(world_x, world_z, t, normalized_y)

            # --- LIGHTING CALCULATION ---
            # Surface normal from slopes: N = normalize(-dH_dx, 1, -dH_dz)
            # For efficiency, we compute the dot product with light direction directly

            # The normal Y component is always positive (pointing up)
            # Steeper slopes = more horizontal normal = catches different light

            # Normalize slopes for lighting (approximate)
            slope_magnitude = np.sqrt(dH_dx**2 + dH_dz**2 + 1)
            nx = -dH_dx / slope_magnitude
            ny = 1.0 / slope_magnitude
            nz = -dH_dz / slope_magnitude

            # Dot product with light direction
            # Light from upper-left-behind the viewer
            lighting = nx * LIGHT_DIR_X + ny * LIGHT_DIR_Y + nz * LIGHT_DIR_Z

            # Remap lighting to useful range
            lighting = (lighting + 1) * 0.5  # 0 to 1
            lighting = np.clip(lighting, 0.1, 1.0)

            # --- TROUGH SHADOWS ---
            # Darken areas where height is low (deep troughs)
            H_normalized = (H - H.min()) / (H.max() - H.min() + 0.01)
            trough_shadow = 0.4 + 0.6 * H_normalized

            # Combine lighting with height and trough shadows
            Z_final = H_normalized * lighting * trough_shadow

            # Add extra brightness on steep forward-facing slopes (wave faces catching light)
            forward_facing = np.maximum(-dH_dz, 0)  # Positive when facing viewer
            forward_facing = np.clip(forward_facing / 0.8, 0, 1)
            Z_final += forward_facing * 0.15

            # Normalization
            Z_norm = Z_final
            Z_norm = np.clip(Z_norm, 0, 1)

            # Contrast curve - more aggressive for stormy look
            Z_norm = np.power(Z_norm, 1.1)

            # --- FOAM ON CRESTS ---
            # Foam appears on high points AND on forward-facing steep slopes
            foam_height = H_normalized > 0.7
            foam_steep = forward_facing > 0.5
            is_foam = foam_height | (foam_steep & (H_normalized > 0.5))

            # Spray effect - random bright spots on high waves
            spray_mask = is_foam & (np.random.random(Z_norm.shape) > 0.5)
            Z_norm = np.where(spray_mask, np.minimum(Z_norm + 0.25, 1.0), Z_norm)

            # Distance haze
            distance_factor = np.clip(normalized_y, 0, 1)
            haze = 0.3 + 0.7 * distance_factor
            Z_norm = 0.25 + (Z_norm - 0.25) * haze

            Z_norm = np.clip(Z_norm, 0, 1)

            # --- SLOPE-AWARE CHARACTER SELECTION ---
            # Choose character based on wave slope direction

            char_idx = (Z_norm * N_CHARS).astype(int)
            char_idx = np.clip(char_idx, 0, N_CHARS - 1)

            # Determine which character set to use based on Z slope
            # Rising toward viewer (negative dH_dz) = front face
            # Falling away (positive dH_dz) = back face
            slope_threshold = 0.15

            is_rising = dH_dz < -slope_threshold  # Front face of wave
            is_falling = dH_dz > slope_threshold   # Back face of wave
            is_flat = ~is_rising & ~is_falling

            # Select characters from appropriate set
            chars_rise = CHARS_RISE[char_idx]
            chars_fall = CHARS_FALL[char_idx]
            chars_flat = CHARS_FLAT[char_idx]
            chars_foam = CHARS_FOAM[char_idx]

            # Combine: foam overrides, then slope-based
            w_chars = np.where(is_foam, chars_foam,
                      np.where(is_rising, chars_rise,
                      np.where(is_falling, chars_fall, chars_flat)))

            # --- RENDERING ---

            # Water color indices
            w_col_idx = (Z_norm * PALETTE_STEPS).astype(int)
            w_col_idx = np.clip(w_col_idx, 0, PALETTE_STEPS - 1)

            w_fgs = fg_palette[w_col_idx]
            w_bgs = bg_palette[w_col_idx]

            # Sky
            sky_fgs_lut, sky_bgs_lut = generate_stormy_sky_lut(rows)
            sy_int = SY.astype(int)
            sy_int = np.clip(sy_int, 0, len(sky_bgs_lut) - 1)
            s_fgs = sky_fgs_lut[sy_int]
            s_bgs = sky_bgs_lut[sy_int]
            s_chars = np.full(SY.shape, " ")

            # Merge
            final_fgs = np.where(is_water, w_fgs, s_fgs)
            final_bgs = np.where(is_water, w_bgs, s_bgs)
            final_chars = np.where(is_water, w_chars, s_chars)

            # Build frame
            buffer_data = np.char.add(final_fgs.flatten(), final_bgs.flatten())
            buffer_data = np.char.add(buffer_data, final_chars.flatten())
            frame_str = "".join(buffer_data)

            move_cursor(1, 1)
            sys.stdout.buffer.write(frame_str.encode("utf-8"))
            sys.stdout.buffer.write(RESET.encode("utf-8"))
            sys.stdout.flush()

            time.sleep(1 / FPS)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(RESET)
        clear_screen()
        show_cursor()
        print("North Sea voyage ended.")


if __name__ == "__main__":
    main()
