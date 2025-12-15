"""
Particle system for rain and smoke effects.
"""

import pyray as rl
import random
from dataclasses import dataclass

from .config import VIRTUAL_WIDTH, VIRTUAL_HEIGHT


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: rl.Color
    size: float


class ParticleSystem:
    # Time for rain to fade in/out (seconds)
    RAIN_FADE_DURATION = 4.0

    def __init__(self):
        self.particles = []
        # Rain state management
        self.is_raining = False
        self.rain_timer = 0.0
        self.rain_duration = 0.0  # How long current state lasts
        self.rain_intensity = 0.0  # 0.0 to 1.0, controls spawn rate and visibility
        self._set_next_rain_state()

    def _set_next_rain_state(self):
        """Set random duration for next rain state."""
        if self.is_raining:
            # Rain for 20-40 seconds
            self.rain_duration = random.uniform(20.0, 40.0)
        else:
            # Dry for 15-30 seconds
            self.rain_duration = random.uniform(15.0, 30.0)
        self.rain_timer = 0.0

    def update_rain_state(self, dt):
        """Update rain on/off cycle with gradual fade. Returns True if any rain visible."""
        self.rain_timer += dt
        if self.rain_timer >= self.rain_duration:
            self.is_raining = not self.is_raining
            self._set_next_rain_state()

        # Gradually adjust rain intensity
        if self.is_raining:
            # Fade in
            self.rain_intensity = min(1.0, self.rain_intensity + dt / self.RAIN_FADE_DURATION)
        else:
            # Fade out
            self.rain_intensity = max(0.0, self.rain_intensity - dt / self.RAIN_FADE_DURATION)

        # Return True if rain is visible (intensity > 0) - used for umbrella trigger
        return self.rain_intensity > 0.1

    def spawn_rain(self, count=3):
        # Spawn rate scales with intensity - fewer particles when fading
        effective_count = max(1, int(count * self.rain_intensity))
        # Alpha scales with intensity for gradual fade effect
        alpha = int(150 * self.rain_intensity)

        for _ in range(effective_count):
            x = random.randint(0, VIRTUAL_WIDTH + 100) - 50
            y = -10
            vx = -2.0  # Wind blowing left
            vy = random.uniform(10.0, 15.0)  # Fast fall
            life = 40.0  # Live long enough to cross screen
            color = rl.Color(150, 200, 255, alpha)  # Fades with intensity
            self.particles.append(Particle(x, y, vx, vy, life, 40.0, color, 1.0))

    def spawn_smoke(self, x, y):
        # Slowed down significantly for gentle drift
        vx = random.uniform(0.03, 0.15)
        vy = random.uniform(-0.15, -0.06)
        life = random.uniform(80, 140)
        c_val = random.randint(140, 190)  # Brighter smoke
        color = rl.Color(c_val, c_val, c_val, 160)  # Higher base alpha
        self.particles.append(
            Particle(x, y, vx, vy, life, life, color, random.uniform(2, 4))
        )

    def update(self, dt):
        alive = []
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.life -= 1

            # Bounds check for rain
            if p.vy > 5.0 and p.y > VIRTUAL_HEIGHT:
                continue

            if p.life > 0:
                alive.append(p)
        self.particles = alive

    def draw(self):
        for p in self.particles:
            if p.vy > 5.0:  # Rain
                # Draw Line
                start_v = rl.Vector2(p.x, p.y)
                end_v = rl.Vector2(p.x - p.vx * 2, p.y - p.vy * 2)  # Trail
                rl.draw_line_v(start_v, end_v, p.color)
            else:  # Smoke
                # Draw soft rect/circle - clearer visibility
                alpha = int((p.life / p.max_life) * 140)  # Higher alpha
                col = rl.Color(p.color.r, p.color.g, p.color.b, alpha)
                rl.draw_rectangle(int(p.x), int(p.y), int(p.size), int(p.size), col)
