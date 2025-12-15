# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "raylib",
#     "cffi",
# ]
# ///
"""
Neon City - A cyberpunk cityscape animation.
"""

import pyray as rl
import random
import time

from .config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    VIRTUAL_WIDTH,
    VIRTUAL_HEIGHT,
    TARGET_FPS,
    C_VOID,
    C_SKY_BOT,
)
from .particles import ParticleSystem
from .buildings import City
from .vehicles import FlyingObject
from .effects import draw_distant_city_lights, draw_power_lines
from .figure import draw_sprite_figure, draw_street_light
from .shaders import FRAGMENT_SHADER
from .holograms import HologramManager


def main():
    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Neon City - Raylib Edition")
    rl.set_target_fps(TARGET_FPS)

    # Load character sprite
    sprite = rl.load_texture("sprites/sprite-3.png")
    sprite_scale = 0.12  # Scale down for 480x270 virtual resolution

    # Load umbrella sprite
    umbrella = rl.load_texture("sprites/umbrella.png")

    # Create Low-Res Render Target
    target = rl.load_render_texture(VIRTUAL_WIDTH, VIRTUAL_HEIGHT)

    # Load Shader
    shader = rl.load_shader_from_memory(rl.ffi.NULL, FRAGMENT_SHADER)

    # Get shader locations
    loc_res = rl.get_shader_location(shader, "resolution")
    loc_time = rl.get_shader_location(shader, "time")

    res_vals = rl.ffi.new("float[2]", [VIRTUAL_WIDTH, VIRTUAL_HEIGHT])
    rl.set_shader_value(
        shader, loc_res, res_vals, rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2
    )

    # Entities
    city = City()
    particles = ParticleSystem()
    flying_objects = [FlyingObject() for _ in range(5)]
    hologram_manager = HologramManager()

    start_time = time.time()

    # Main Loop
    while not rl.window_should_close():
        t = time.time() - start_time

        # Regenerate on R key
        if rl.is_key_pressed(82):  # KEY_R
            city = City()
            flying_objects = [FlyingObject() for _ in range(5)]
            hologram_manager.unload()
            hologram_manager = HologramManager()

        # Update
        city.update()
        dt = 1.0 / 60.0
        is_raining = particles.update_rain_state(dt)
        if is_raining:
            particles.spawn_rain()
        particles.update(dt)
        for fo in flying_objects:
            fo.update()
        hologram_manager.update(1.0 / 60.0)

        # Update Shader uniforms
        t_val = rl.ffi.new("float[1]", [t])
        rl.set_shader_value(
            shader, loc_time, t_val, rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT
        )

        # --- DRAW TO VIRTUAL TEXTURE ---
        rl.begin_texture_mode(target)
        rl.clear_background(C_VOID)

        # 1. Sky Gradient
        rl.draw_rectangle_gradient_v(
            0, 0, VIRTUAL_WIDTH, VIRTUAL_HEIGHT, C_VOID, C_SKY_BOT
        )

        # 1.5 Distant city lights (far background)
        draw_distant_city_lights()

        # 2. City Layers
        city.draw()

        # 2.3 Holograms (floating above buildings)
        hologram_manager.draw()

        # 2.5 Flying objects (in the distance)
        for fo in flying_objects:
            fo.draw(t)

        # 2.6 Power lines
        draw_power_lines(t)

        # 3. Reflection zone
        refl_y = VIRTUAL_HEIGHT - 30

        # Figure sits on RIGHT side
        sprite_w = int(sprite.width * sprite_scale)

        # 4. Figure on building ledge
        figure_x = VIRTUAL_WIDTH - sprite_w - 5  # Pushed to far right
        ledge_y = refl_y

        # Draw minimal building ledge (just under figure)
        rl.draw_rectangle(
            int(figure_x - 10),
            int(ledge_y),
            VIRTUAL_WIDTH - int(figure_x) + 10,
            8,
            rl.Color(25, 18, 30, 255),
        )
        rl.draw_rectangle(
            int(figure_x - 10),
            int(ledge_y),
            VIRTUAL_WIDTH - int(figure_x) + 10,
            2,
            rl.Color(40, 30, 50, 255),
        )

        # Draw overhang light above figure
        draw_street_light(figure_x, ledge_y, t, sprite_w)

        # Draw the figure sprite (with umbrella if raining)
        cig_pos = draw_sprite_figure(sprite, figure_x, ledge_y, t, sprite_scale, umbrella, is_raining)

        # Smoke spawn
        if random.random() < 0.1:
            particles.spawn_smoke(cig_pos[0], cig_pos[1])

        # 7. Particles (Rain/Smoke)
        particles.draw()

        rl.end_texture_mode()

        # --- DRAW TO SCREEN (UPSCALED) ---
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)

        # Draw render texture to screen, scaled up
        source_rec = rl.Rectangle(0, 0, VIRTUAL_WIDTH, -VIRTUAL_HEIGHT)
        dest_rec = rl.Rectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        origin = rl.Vector2(0, 0)

        rl.begin_shader_mode(shader)
        rl.draw_texture_pro(target.texture, source_rec, dest_rec, origin, 0.0, rl.WHITE)
        rl.end_shader_mode()

        rl.draw_fps(10, 10)
        rl.end_drawing()

    hologram_manager.unload()
    rl.unload_texture(umbrella)
    rl.unload_texture(sprite)
    rl.unload_render_texture(target)
    rl.unload_shader(shader)
    rl.close_window()


if __name__ == "__main__":
    main()
