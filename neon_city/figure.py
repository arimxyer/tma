"""
Character/figure sprite drawing.
"""

import pyray as rl
import random


# Flicker state - persists between frames
_flicker_state = {"is_flickering": False, "flicker_end": 0, "flicker_intensity": 1.0}


def draw_street_light(x, ledge_y, t, sprite_w):
    """Draw a street lamp arm coming from right, curving over the character."""
    global _flicker_state

    pole_color = rl.Color(25, 20, 35, 255)

    # Short arm from right edge, natural curve down to lamp
    arm_start_x = int(x + sprite_w + 6)  # Just off right edge
    arm_start_y = int(ledge_y - 62)
    lamp_x = int(x + sprite_w // 2 + 1)  # Above character center
    lamp_y = int(ledge_y - 52)

    # Control point for bezier curve (pulls the curve into natural arc)
    ctrl_x = arm_start_x - 3
    ctrl_y = lamp_y - 5

    # Draw smooth quadratic bezier curve
    steps = 10
    for i in range(steps):
        t1 = i / steps
        t2 = (i + 1) / steps

        # Quadratic bezier formula
        x1 = (
            (1 - t1) * (1 - t1) * arm_start_x
            + 2 * (1 - t1) * t1 * ctrl_x
            + t1 * t1 * lamp_x
        )
        y1 = (
            (1 - t1) * (1 - t1) * arm_start_y
            + 2 * (1 - t1) * t1 * ctrl_y
            + t1 * t1 * lamp_y
        )
        x2 = (
            (1 - t2) * (1 - t2) * arm_start_x
            + 2 * (1 - t2) * t2 * ctrl_x
            + t2 * t2 * lamp_x
        )
        y2 = (
            (1 - t2) * (1 - t2) * arm_start_y
            + 2 * (1 - t2) * t2 * ctrl_y
            + t2 * t2 * lamp_y
        )

        rl.draw_line(int(x1), int(y1), int(x2), int(y2), pole_color)
        rl.draw_line(int(x1), int(y1) + 1, int(x2), int(y2) + 1, pole_color)

    # Lamp fixture (hanging from end of curve)
    fixture_w = 6
    fixture_h = 4
    fixture_x = lamp_x - fixture_w // 2
    fixture_y = lamp_y
    rl.draw_rectangle(
        fixture_x, fixture_y, fixture_w, fixture_h, rl.Color(30, 25, 40, 255)
    )

    # Handle flicker state
    if not _flicker_state["is_flickering"]:
        # ~0.3% per frame = occasional flicker every few seconds
        if random.random() < 0.003:
            _flicker_state["is_flickering"] = True
            _flicker_state["flicker_end"] = t + random.uniform(0.1, 0.3)
    else:
        if t >= _flicker_state["flicker_end"]:
            _flicker_state["is_flickering"] = False
            _flicker_state["flicker_intensity"] = 1.0
        else:
            _flicker_state["flicker_intensity"] = random.uniform(0.2, 1.0)

    intensity = _flicker_state["flicker_intensity"]

    # Light color - warm yellow/orange
    light_r = int(255 * intensity)
    light_g = int(180 * intensity)
    light_b = int(100 * intensity)

    # Light origin (bottom center of fixture)
    light_x = fixture_x + fixture_w // 2
    light_y = fixture_y + fixture_h

    # Light cone shining down onto character - wide enough to cover full body
    cone_height = int(ledge_y - light_y - 2)
    cone_width_bottom = 55  # Wide enough to illuminate full character

    # Soft ambient glow - larger and brighter
    glow_alpha = int(45 * intensity)
    rl.draw_circle(
        light_x,
        light_y + cone_height // 2,
        28,
        rl.Color(light_r, light_g, light_b, glow_alpha),
    )
    # Inner brighter glow
    rl.draw_circle(
        light_x,
        light_y + cone_height // 2,
        18,
        rl.Color(light_r, light_g, light_b, int(glow_alpha * 1.3)),
    )

    # Light cone (triangular gradient expanding downward) - brighter to illuminate silhouette
    for i in range(cone_height):
        progress = i / cone_height
        width = int(cone_width_bottom * progress)
        alpha = int((75 - 50 * progress) * intensity)  # Much brighter
        if alpha > 0 and width > 0:
            rl.draw_rectangle(
                light_x - width // 2,
                light_y + i,
                width,
                1,
                rl.Color(light_r, light_g, light_b, alpha),
            )

    # Bright spot at lamp fixture
    if intensity > 0.5:
        rl.draw_circle(
            light_x, light_y, 3, rl.Color(255, 220, 150, int(100 * intensity))
        )
        rl.draw_circle(
            light_x, light_y, 2, rl.Color(255, 250, 220, int(160 * intensity))
        )


def draw_sprite_figure(sprite, x, ledge_y, t, scale=0.12):
    """Draw the character sprite sitting on ledge with cigarette."""
    sprite_w = int(sprite.width * scale)
    sprite_h = int(sprite.height * scale)

    # Position sprite so butt sits ON the ledge
    sprite_x = x
    sprite_y = ledge_y - sprite_h * 0.905

    # Draw the sprite
    rl.draw_texture_ex(sprite, rl.Vector2(sprite_x, sprite_y), 0, scale, rl.WHITE)

    # Add cigarette in hand (static ember, brighter)
    glow_alpha = 220

    # Cigarette position - tuned values
    finger_x = sprite_x + sprite_w * 0.46
    finger_y = sprite_y + sprite_h * 0.458

    # Draw cigarette stick extending LEFT from fingers
    cig_length = max(2, int(1.5 * scale * 30))
    cig_color = rl.Color(90, 80, 70, 255)
    rl.draw_rectangle(
        int(finger_x - cig_length),
        int(finger_y),
        cig_length,
        max(1, int(scale * 4)),
        cig_color,
    )

    # Ember at the LEFT tip of cigarette
    ember_x = finger_x - cig_length
    ember_y = finger_y

    # Ember glow - brighter and more visible
    rl.draw_circle(
        int(ember_x),
        int(ember_y),
        max(3, int(6 * scale * 5)),
        rl.Color(255, 60, 0, glow_alpha // 4),
    )
    rl.draw_circle(
        int(ember_x),
        int(ember_y),
        max(2, int(4 * scale * 5)),
        rl.Color(255, 100, 20, glow_alpha // 2),
    )
    rl.draw_circle(
        int(ember_x),
        int(ember_y),
        max(1, int(2 * scale * 5)),
        rl.Color(255, 150, 50, glow_alpha),
    )

    return int(ember_x), int(ember_y)
