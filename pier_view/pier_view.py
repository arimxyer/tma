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
FPS = 60  # Smoother motion for high detail
SPEED = 1.0  # Slower for "bigger" feel
FOV = 2.5  # Zoom out significantly (was 1.5)

# --- Wave Size Characteristics ---
# Wave set parameters (groups of larger waves)
WAVE_SET_PERIOD = 45.0  # Seconds between wave set peaks
WAVE_SET_SIZE = 7  # Number of waves in a set
WAVE_SET_INTENSITY = 0.4  # How much larger set waves are (0-1)

# Rogue wave parameters
ROGUE_WAVE_CHANCE = 0.0008  # Per-frame chance of rogue wave trigger
ROGUE_WAVE_MULTIPLIER = 1.8  # How much larger than normal
ROGUE_WAVE_DURATION = 8.0  # Seconds for rogue wave to pass

# Breathing/swell envelope (slow overall intensity changes)
SWELL_ENVELOPE_PERIOD = 120.0  # Very slow breathing cycle
SWELL_ENVELOPE_DEPTH = 0.3  # Min intensity multiplier during calm

# Atmospheric perspective
HORIZON_HAZE = 0.6  # How much to fade waves at horizon (0-1)
DEPTH_FOAM_THRESHOLD = 0.7  # How close waves need to be to show foam

# Character gradient (from low/deep to high/foam)
# Extended gradient for more nuanced height representation
CHARS = np.array(list("  .,-~:;=+*x%#@"))
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
def ansi_color_bg(r, g, b):
    return f"\033[48;2;{r};{g};{b}m"


def ansi_color_fg(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


RESET = "\033[0m"


# Generate color palette (High Contrast: Dark Troughs, Bright Peaks)
def generate_palette(steps):
    stops = [
        (0.0, (0, 0, 5)),  # Pitch Black Abyss
        (0.3, (0, 15, 35)),  # Deep Shadow
        (0.55, (0, 50, 90)),  # Mid Ocean
        (0.75, (0, 100, 130)),  # Upper Wave
        (0.9, (50, 220, 220)),  # Highlight
        (0.96, (200, 255, 255)),  # Foam
        (1.0, (255, 255, 255)),  # Whitecap
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
                # Darker background for contrast
                bg_r, bg_g, bg_b = r // 12, g // 12, b // 6
                palette_bg.append(ansi_color_bg(bg_r, bg_g, bg_b))
                break

    return np.array(palette_fg), np.array(palette_bg)


def generate_sky_palette(height):
    # Gradient from Dark Purple/Black (top) to Horizon Haze (bottom)
    sky_colors = []
    for y in range(height):
        t = y / max(height, 1)
        r = int(10 + t * 40)
        g = int(10 + t * 60)
        b = int(30 + t * 100)
        sky_colors.append(
            (ansi_color_fg(r // 2, g // 2, b // 2), ansi_color_bg(r, g, b))
        )
    return sky_colors


# --- Wave Size Modulation Functions ---
def compute_wave_set_envelope(t, world_z):
    """
    Creates wave sets - groups of larger waves that roll in together.
    Uses multiple incommensurate frequencies to avoid obvious repetition.
    """
    # Multiple overlapping cycles with irrational ratios (never quite repeat)
    cycle1 = math.sin(t * 0.0731) * 0.5 + 0.5  # ~86 sec period
    cycle2 = math.sin(t * 0.0523) * 0.3 + 0.5  # ~120 sec period
    cycle3 = math.sin(t * 0.0892) * 0.2 + 0.5  # ~70 sec period

    # Combine cycles - creates irregular "bunching" of energy
    combined = (cycle1 * cycle2 * cycle3)

    # Spatial variation - different parts of the ocean have different intensity
    spatial = np.sin(world_z * 0.08 + t * 0.12)
    spatial = (spatial + 1) * 0.5

    # Subtle modulation, not dramatic swings
    intensity = 1.0 + combined * spatial * WAVE_SET_INTENSITY * 0.7

    return intensity


def compute_swell_envelope(t):
    """
    Very slow, organic variation in ocean energy.
    Uses multiple incommensurate periods so it never feels like a loop.
    """
    # Multiple slow cycles with irrational relationships
    wave1 = math.sin(t * 0.0089)  # ~11 min period
    wave2 = math.sin(t * 0.0134)  # ~7.8 min period
    wave3 = math.sin(t * 0.0201)  # ~5.2 min period

    # Combine with different weights
    combined = wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.2

    # Normalize to subtle range - don't want dramatic calm/storm swings
    # Just gentle variation in overall intensity
    envelope = 0.85 + combined * 0.15

    return envelope


class RogueWaveState:
    """Tracks rogue wave events - occasional massive waves."""
    def __init__(self):
        self.active = False
        self.start_time = 0
        self.position_z = 0  # Where in world-z the rogue wave is centered

    def update(self, t):
        # Check if current rogue wave has passed
        if self.active:
            if t - self.start_time > ROGUE_WAVE_DURATION:
                self.active = False

        # Maybe trigger new rogue wave
        if not self.active and np.random.random() < ROGUE_WAVE_CHANCE:
            self.active = True
            self.start_time = t
            self.position_z = np.random.uniform(5, 30)  # Start somewhere in view

    def get_multiplier(self, t, world_z):
        if not self.active:
            return np.ones_like(world_z)

        # Progress through the rogue wave event (0 to 1)
        progress = (t - self.start_time) / ROGUE_WAVE_DURATION

        # Rogue wave moves toward viewer
        current_z = self.position_z - progress * 25

        # Gaussian envelope around the rogue wave position
        distance = np.abs(world_z - current_z)
        envelope = np.exp(-distance**2 / 50)

        # Amplitude envelope: builds up, peaks, then subsides
        time_envelope = np.sin(progress * math.pi) ** 0.5

        multiplier = 1.0 + envelope * (ROGUE_WAVE_MULTIPLIER - 1.0) * time_envelope

        return multiplier


def apply_atmospheric_perspective(Z_norm, normalized_y):
    """
    Distant waves (near horizon) should appear more muted/hazy.
    Close waves should have full contrast and detail.
    """
    # Distance factor: 0 at horizon, 1 at viewer
    distance_factor = np.clip(normalized_y, 0, 1)

    # Haze reduces contrast at distance
    haze = 1.0 - (1.0 - distance_factor) * HORIZON_HAZE

    # Apply haze: push values toward middle gray at distance
    middle = 0.35
    Z_hazed = middle + (Z_norm - middle) * haze

    return Z_hazed


def apply_foam_depth_scaling(Z_norm, normalized_y):
    """
    Only show bright foam/whitecaps on waves that are close enough to see.
    Distant waves appear more uniform/muted.
    """
    # Close waves can show full foam
    # Distant waves have their peaks suppressed
    distance_factor = np.clip(normalized_y / DEPTH_FOAM_THRESHOLD, 0, 1)

    # Foam threshold: only peaks above this can be bright white
    foam_threshold = 0.7

    # For distant waves, cap the maximum brightness
    max_brightness = foam_threshold + (1.0 - foam_threshold) * distance_factor

    Z_scaled = np.minimum(Z_norm, max_brightness)

    return Z_scaled


def main():
    enable_windows_ansi()
    clear_screen()
    hide_cursor()

    cols, rows = get_terminal_size()
    # Subtract 1 to prevent scroll
    rows -= 1

    # Precompute palette
    PALETTE_STEPS = 60
    fg_palette, bg_palette = generate_palette(PALETTE_STEPS)

    # Full screen meshgrid (screen coordinates)
    # y = 0 (top) to rows (bottom)
    # x = 0 (left) to cols (right)

    # Initial setup
    sx = np.linspace(-1, 1, cols)
    sy = np.arange(rows)
    SX, SY = np.meshgrid(sx, sy)

    t_start = time.time()

    # Initialize rogue wave tracker
    rogue_wave = RogueWaveState()

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

            # Update rogue wave state
            rogue_wave.update(t)

            # Compute global swell envelope (slow breathing)
            swell_envelope = compute_swell_envelope(t)

            # --- STABLE HORIZON ---
            # Keep horizon fixed - let the waves themselves create depth perception
            # Base horizon at roughly 35% down screen
            base_horizon_y = rows * 0.35

            # Horizon is a simple horizontal line
            # The waves will provide all the visual motion and depth cues
            HORIZON_Y = np.full((rows, cols), base_horizon_y)

            # Calculate Normalized Y for Water Projection
            # If Y < Horizon, it's sky. If Y >= Horizon, it's water.
            # Avoid divide by zero by adding epsilon
            water_depth = rows - HORIZON_Y
            normalized_y = (SY - HORIZON_Y) / (water_depth + 0.1)

            # Mask: Where are we?
            is_water = normalized_y >= 0

            # --- WATER PHYSICS (FRACTAL TROCHOIDAL WAVES) ---

            # 1. Projection
            # Push camera WAY higher (20.0) to see much further into the horizon
            # This packs many more waves into the vertical space
            valid_y = np.maximum(normalized_y, 0.005)
            world_z = 20.0 / valid_y
            world_x = SX * world_z * FOV

            # 2. Fractal Summation (6 Layers)
            # Accumulators
            H = np.zeros_like(world_x)

            # Layer 0: The Groundswell (Subtle deep ocean undulation)
            # Multiple overlapping swells at different angles for organic feel
            k0a = 0.06
            k0b = 0.045
            phase0a = world_z * k0a + world_x * 0.02 + t * 0.12
            phase0b = world_z * k0b - world_x * 0.015 + t * 0.09
            # Combine two crossing swells
            raw0 = np.sin(phase0a) * 0.6 + np.sin(phase0b) * 0.4
            h0 = ((raw0 + 1.0) * 0.5) ** 1.3
            # Gentle contribution, not overwhelming
            H += h0 * 0.35

            # Layer 1: The Leviathan (Base Swell)
            # Increased frequency to pack more waves into the view
            k1 = 0.3
            phase1 = (world_x * 0.1 + world_z * 0.9) * k1 + t * 0.6
            raw1 = np.sin(phase1)
            # Sharpen peaks (Trochoid approx)
            h1 = ((raw1 + 1.0) * 0.5) ** 3.0
            H += h1 * 1.0

            # Layer 2: The Roller (Secondary Swell)
            k2 = 0.6
            phase2 = (world_x * 0.5 - world_z * 0.4) * k2 + t * 0.9
            raw2 = np.sin(phase2)
            h2 = ((raw2 + 1.0) * 0.5) ** 2.5
            H += h2 * 0.5

            # Layer 3: The Chop (Breaking surface)
            k3 = 1.2
            phase3 = (world_x * 0.8 + world_z * 0.3) * k3 + t * 1.4
            raw3 = np.sin(phase3)
            h3 = ((raw3 + 1.0) * 0.5) ** 2.0
            H += h3 * 0.25

            # Layer 4: Detail Noise (Texture)
            k4 = 3.0
            phase4 = (world_x * 1.0 - world_z * 1.0) * k4 + t * 2.5
            h4 = np.sin(phase4)
            # Modulate by underlying height (physics: smooth troughs, rough peaks)
            H += h4 * 0.1 * (H + 0.1)

            # Layer 5: Micro Ripple
            k5 = 8.0
            phase5 = (world_x * 1.2 + world_z * 0.8) * k5 + t * 4.0
            h5 = np.sin(phase5) * 0.05
            H += h5

            # Shoaling: Scale amplitude by distance
            # Waves at horizon (small y) are smaller, waves at front (y=1) are huge
            H *= 0.2 + 0.8 * np.minimum(normalized_y, 1.0)

            # --- WAVE SIZE MODULATIONS ---

            # Apply wave set envelope (groups of larger waves)
            wave_set_mult = compute_wave_set_envelope(t, world_z)
            H *= wave_set_mult

            # Apply global swell envelope (slow breathing)
            H *= swell_envelope

            # Apply rogue wave multiplier (occasional massive waves)
            rogue_mult = rogue_wave.get_multiplier(t, world_z)
            H *= rogue_mult

            # Enhanced size contrast: make peaks more dramatic
            # Amplify the difference between troughs and crests
            H_enhanced = np.where(H > 0.5, H + (H - 0.5) * 0.3, H * 0.9)
            H = H_enhanced

            # Lighting / Slope Calc (Derivative of main layers)
            # Approx slope for Layer 1+2
            slope1 = np.cos(phase1) * k1
            slope2 = np.cos(phase2) * k2
            total_slope = slope1 * 1.0 + slope2 * 0.5

            # Backlight Effect:
            # If slope is facing camera (neg slope?), bright.
            # Actually, standard lighting: Sun is top/back.
            # Forward faces (facing camera) are in shadow?
            # Or Sun is "Pier" light? Let's assume ambient light + top down.
            # Sharp peaks get light.

            slope_factor = total_slope * 0.3

            # Final Composition
            Z_final = H + slope_factor

            # Normalization with high contrast curve
            # Range is roughly 0..1.8
            Z_norm = (Z_final + 0.2) / 2.0

            # Power curve to crush blacks and pop whites
            # This is the "HDR" look
            Z_norm = np.power(np.clip(Z_norm, 0, 1), 1.5)

            # --- DEPTH-BASED VISUAL EFFECTS ---

            # Apply atmospheric perspective (distant waves are hazier)
            Z_norm = apply_atmospheric_perspective(Z_norm, normalized_y)

            # Apply foam depth scaling (only close waves show bright foam)
            Z_norm = apply_foam_depth_scaling(Z_norm, normalized_y)

            # Final clamp
            Z_norm = np.clip(Z_norm, 0, 1)

            # --- RENDERING ---

            # Prepare buffers

            # SKY GENERATION
            # Simple vertical gradient based on Y coordinate
            # Map Y (0..rows) to color index
            sky_t = SY / (rows * 0.5)  # approximate
            sky_r = (10 + sky_t * 40).astype(int)
            sky_g = (10 + sky_t * 60).astype(int)
            sky_b = (30 + sky_t * 100).astype(int)

            # Clamp colors
            sky_r = np.clip(sky_r, 0, 255)
            sky_g = np.clip(sky_g, 0, 255)
            sky_b = np.clip(sky_b, 0, 255)

            # Construct Sky ANSI strings
            # We need to construct this efficiently.
            # Let's pre-format the strings? Too slow per pixel?
            # Actually, let's just use the palette approach for sky too?
            # Or simpler: Just use a static background for sky and spaces.

            # OPTIMIZATION:
            # Let's treat the whole screen as one array of indices into two palettes:
            # 1. Sky Palette
            # 2. Water Palette
            # But they overlap in logic.

            # Let's stick to the string construction which worked well.

            # Water Indices
            w_char_idx = (Z_norm * N_CHARS).astype(int)
            w_col_idx = (Z_norm * PALETTE_STEPS).astype(int)

            # Safety clamp for indices
            w_char_idx = np.clip(w_char_idx, 0, N_CHARS - 1)
            w_col_idx = np.clip(w_col_idx, 0, PALETTE_STEPS - 1)

            w_fgs = fg_palette[w_col_idx]
            w_bgs = bg_palette[w_col_idx]
            w_chars = CHARS[w_char_idx]

            # Sky Arrays (we need string arrays)
            # For speed, let's just make the sky Solid Black/Dark Blue with stars?
            # Or computed gradient. Computed gradient in Python string ops is fast enough.

            # Vectorized ANSI for sky is tricky without a lookup table.
            # Let's use a lookup table for Sky too.
            SKY_STEPS = rows
            # Precompute sky strings
            sky_fgs_lut = []
            sky_bgs_lut = []
            for y in range(SKY_STEPS + 10):  # pad
                t = y / max(SKY_STEPS, 1)
                r, g, b = int(10 + t * 40), int(10 + t * 60), int(30 + t * 100)
                sky_bgs_lut.append(ansi_color_bg(r, g, b))
                # Stars? Randomly add stars
                if np.random.rand() > 0.98:
                    sky_fgs_lut.append(ansi_color_fg(255, 255, 255))
                else:
                    sky_fgs_lut.append(ansi_color_fg(r, g, b))  # hide char

            sky_fgs_lut = np.array(sky_fgs_lut)
            sky_bgs_lut = np.array(sky_bgs_lut)

            # Map SY to indices
            sy_int = SY.astype(int)
            sy_int = np.clip(sy_int, 0, len(sky_bgs_lut) - 1)

            s_fgs = sky_fgs_lut[sy_int]
            s_bgs = sky_bgs_lut[sy_int]
            s_chars = np.full(SY.shape, " ")  # Empty sky

            # MERGE MASKS
            # Where is_water is True, use w_ arrays. Else s_ arrays.

            final_fgs = np.where(is_water, w_fgs, s_fgs)
            final_bgs = np.where(is_water, w_bgs, s_bgs)
            final_chars = np.where(is_water, w_chars, s_chars)

            # Flatten and Join
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
        print("Pier view ended.")


if __name__ == "__main__":
    main()
