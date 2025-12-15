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
    def __init__(self):
        self.particles = []
        # Rain state management
        self.is_raining = False
        self.rain_timer = 0.0
        self.rain_duration = 0.0  # How long current state lasts
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
        """Update rain on/off cycle. Returns True if raining."""
        self.rain_timer += dt
        if self.rain_timer >= self.rain_duration:
            self.is_raining = not self.is_raining
            self._set_next_rain_state()
        return self.is_raining

    def spawn_rain(self, count=3):
        for _ in range(count):
            x = random.randint(0, VIRTUAL_WIDTH + 100) - 50
            y = -10
            vx = -2.0  # Wind blowing left
            vy = random.uniform(10.0, 15.0)  # Fast fall
            life = 40.0  # Live long enough to cross screen
            color = rl.Color(150, 200, 255, 150)  # Transparent blue
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
