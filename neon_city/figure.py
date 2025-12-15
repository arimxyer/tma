"""
Character/figure sprite drawing.
"""

import pyray as rl
import random


# Flicker state - persists between frames
_flicker_state = {"is_flickering": False, "flicker_end": 0, "flicker_intensity": 1.0}

# Umbrella animation state
_umbrella_state = {
    "state": "closed",  # closed, opening, open, closing
    "progress": 0.0,    # 0.0 to 1.0 animation progress
}
UMBRELLA_ANIM_DURATION = 0.5  # seconds for open/close animation


def ease_in_out_cubic(t):
    """Smooth easing function for natural motion."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def update_umbrella_state(dt, should_be_open):
    """Update umbrella animation state machine."""
    global _umbrella_state
    state = _umbrella_state["state"]
    progress = _umbrella_state["progress"]

    if state == "closed":
        if should_be_open:
            _umbrella_state["state"] = "opening"
            _umbrella_state["progress"] = 0.0
    elif state == "opening":
        progress += dt / UMBRELLA_ANIM_DURATION
        if progress >= 1.0:
            _umbrella_state["state"] = "open"
            _umbrella_state["progress"] = 1.0
        else:
            _umbrella_state["progress"] = progress
    elif state == "open":
        if not should_be_open:
            _umbrella_state["state"] = "closing"
            _umbrella_state["progress"] = 1.0
    elif state == "closing":
        progress -= dt / UMBRELLA_ANIM_DURATION
        if progress <= 0.0:
            _umbrella_state["state"] = "closed"
            _umbrella_state["progress"] = 0.0
        else:
            _umbrella_state["progress"] = progress

    return _umbrella_state["progress"]


def draw_procedural_umbrella(center_x, center_y, open_amount, t):
    """Draw procedurally animated umbrella with smooth open/close."""
    import math

    if open_amount <= 0.01:
        return  # Don't draw when fully closed

    eased = ease_in_out_cubic(open_amount)

    # Canopy parameters - wide and flat elliptical shape
    closed_angle = 15  # degrees when closed (narrow)
    open_angle = 170   # degrees when fully open (wider arc)
    current_angle = closed_angle + (open_angle - closed_angle) * eased

    # Elliptical radii - wider than tall for flat umbrella look
    base_radius_x = 26  # Wider horizontally
    base_radius_y = 14  # Shorter vertically (flatter)
    radius_x = base_radius_x * (0.5 + 0.5 * eased)
    radius_y = base_radius_y * (0.5 + 0.5 * eased)

    # Handle length - extends as umbrella opens
    handle_length = 6 + 8 * eased

    # Colors - darker black umbrella
    canopy_color = rl.Color(8, 8, 12, int(255 * eased))
    canopy_dark = rl.Color(3, 3, 6, int(255 * eased))  # Even darker inner layer
    handle_color = rl.Color(25, 25, 30, int(255 * eased))
    edge_color = rl.Color(35, 35, 45, int(255 * eased))

    # Draw handle first (behind canopy)
    handle_end_y = center_y + handle_length
    # Thicker handle (draw twice offset)
    rl.draw_line(
        int(center_x), int(center_y + radius_y * 0.5),
        int(center_x), int(handle_end_y),
        handle_color
    )
    rl.draw_line(
        int(center_x + 1), int(center_y + radius_y * 0.5),
        int(center_x + 1), int(handle_end_y),
        handle_color
    )
    # Handle hook
    hook_radius = 3
    rl.draw_circle_lines(
        int(center_x - hook_radius), int(handle_end_y),
        hook_radius, handle_color
    )

    # Draw canopy as filled elliptical arc
    half_angle = current_angle / 2
    start_angle = 270 - half_angle  # Point upward

    # Draw filled sector using triangles - elliptical
    segments = 20  # More segments for smoother curve
    angle_step = current_angle / segments

    # Draw darker inner layer first for thickness effect
    for i in range(segments):
        angle1 = math.radians(start_angle + i * angle_step)
        angle2 = math.radians(start_angle + (i + 1) * angle_step)

        x1 = center_x + radius_x * 0.85 * math.cos(angle1)
        y1 = center_y + radius_y * 0.85 * math.sin(angle1)
        x2 = center_x + radius_x * 0.85 * math.cos(angle2)
        y2 = center_y + radius_y * 0.85 * math.sin(angle2)

        rl.draw_triangle(
            rl.Vector2(center_x, center_y),
            rl.Vector2(x1, y1),
            rl.Vector2(x2, y2),
            canopy_dark
        )

    # Draw main canopy layer
    for i in range(segments):
        angle1 = math.radians(start_angle + i * angle_step)
        angle2 = math.radians(start_angle + (i + 1) * angle_step)

        x1 = center_x + radius_x * math.cos(angle1)
        y1 = center_y + radius_y * math.sin(angle1)
        x2 = center_x + radius_x * math.cos(angle2)
        y2 = center_y + radius_y * math.sin(angle2)

        rl.draw_triangle(
            rl.Vector2(center_x, center_y),
            rl.Vector2(x1, y1),
            rl.Vector2(x2, y2),
            canopy_color
        )

    # Draw thick canopy edge for definition (multiple lines)
    for i in range(segments):
        angle1 = math.radians(start_angle + i * angle_step)
        angle2 = math.radians(start_angle + (i + 1) * angle_step)

        x1 = center_x + radius_x * math.cos(angle1)
        y1 = center_y + radius_y * math.sin(angle1)
        x2 = center_x + radius_x * math.cos(angle2)
        y2 = center_y + radius_y * math.sin(angle2)

        # Draw edge twice for thickness
        rl.draw_line(int(x1), int(y1), int(x2), int(y2), edge_color)
        rl.draw_line(int(x1), int(y1) + 1, int(x2), int(y2) + 1, edge_color)

    # Draw ribs radiating from center
    rib_count = 6
    for i in range(rib_count):
        rib_angle = math.radians(start_angle + (i + 0.5) * (current_angle / rib_count))
        rib_x = center_x + radius_x * 0.92 * math.cos(rib_angle)
        rib_y = center_y + radius_y * 0.92 * math.sin(rib_angle)
        rl.draw_line(
            int(center_x), int(center_y),
            int(rib_x), int(rib_y),
            rl.Color(18, 18, 24, int(200 * eased))
        )

    # Center hub
    rl.draw_circle(int(center_x), int(center_y), 2, handle_color)


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


def draw_umbrella(umbrella_texture, x, ledge_y, sprite_w, sprite_h):
    """Draw umbrella above character when it's raining."""
    if umbrella_texture is None:
        return

    # Scale down from 48x48 source sprite
    umbrella_scale = 0.5
    umbrella_w = int(umbrella_texture.width * umbrella_scale)
    umbrella_h = int(umbrella_texture.height * umbrella_scale)

    # Position umbrella above and slightly in front of character
    umbrella_x = x + sprite_w * 0.3 - umbrella_w // 2
    umbrella_y = ledge_y - sprite_h * 0.9 - umbrella_h * 0.6

    rl.draw_texture_ex(
        umbrella_texture,
        rl.Vector2(umbrella_x, umbrella_y),
        0,
        umbrella_scale,
        rl.WHITE
    )


def draw_sprite_figure(sprite, x, ledge_y, t, scale=0.12, umbrella_texture=None, is_raining=False, dt=1.0/60.0):
    """Draw the character sprite sitting on ledge with cigarette."""
    sprite_w = int(sprite.width * scale)
    sprite_h = int(sprite.height * scale)

    # Position sprite so butt sits ON the ledge
    sprite_x = x
    sprite_y = ledge_y - sprite_h * 0.905

    # Update and draw procedural umbrella with smooth animation
    open_amount = update_umbrella_state(dt, is_raining)
    # Position umbrella to cover upper body
    umbrella_x = sprite_x + sprite_w * 0.5 + 8
    umbrella_y = sprite_y + 6
    draw_procedural_umbrella(umbrella_x, umbrella_y, open_amount, t)

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
