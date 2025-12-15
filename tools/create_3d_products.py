"""
Generate 3D rotating product sprites for holograms.
Creates wireframe perfume bottle and branded soda can.
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --- 3D Math (adapted from terminal_cube.py) ---


def rotate_y(verts, angle):
    """Rotate vertices around Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    v_new = verts.copy()
    v_new[:, 0] = verts[:, 0] * c + verts[:, 2] * s
    v_new[:, 2] = -verts[:, 0] * s + verts[:, 2] * c
    return v_new


def rotate_x(verts, angle):
    """Rotate vertices around X axis."""
    c, s = np.cos(angle), np.sin(angle)
    v_new = verts.copy()
    v_new[:, 1] = verts[:, 1] * c - verts[:, 2] * s
    v_new[:, 2] = verts[:, 1] * s + verts[:, 2] * c
    return v_new


def project(verts, width, height, camera_dist=5.0, fov=2.0):
    """Project 3D vertices to 2D screen coordinates."""
    v_cam = verts.copy()
    v_cam[:, 2] += camera_dist

    factor = fov / np.maximum(v_cam[:, 2], 0.1)

    x_proj = v_cam[:, 0] * factor
    y_proj = v_cam[:, 1] * factor

    screen_x = x_proj * (height * 0.5) + width * 0.5
    screen_y = -y_proj * (height * 0.5) + height * 0.5

    return np.stack([screen_x, screen_y, v_cam[:, 2]], axis=1)


# --- Geometry Generators ---


def create_cylinder_vertices(radius=1.0, height=2.0, segments=16):
    """Create vertices for a cylinder."""
    vertices = []

    # Top and bottom circles
    for y in [-height / 2, height / 2]:
        for i in range(segments):
            angle = (i / segments) * 2 * math.pi
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            vertices.append([x, y, z])

    return np.array(vertices, dtype=float)


def create_cylinder_edges(segments=16):
    """Create edge indices for a cylinder."""
    edges = []

    # Bottom circle edges
    for i in range(segments):
        edges.append((i, (i + 1) % segments))

    # Top circle edges
    for i in range(segments):
        edges.append((segments + i, segments + (i + 1) % segments))

    # Vertical edges
    for i in range(segments):
        edges.append((i, segments + i))

    return edges


def create_bottle_vertices():
    """Create vertices for a perfume bottle shape."""
    vertices = []
    segments = 12

    # Base (wider)
    base_radius = 0.8
    base_height = -1.2
    for i in range(segments):
        angle = (i / segments) * 2 * math.pi
        x = base_radius * math.cos(angle)
        z = base_radius * math.sin(angle)
        vertices.append([x, base_height, z])

    # Body bottom
    body_radius = 0.7
    body_bottom = -1.0
    for i in range(segments):
        angle = (i / segments) * 2 * math.pi
        x = body_radius * math.cos(angle)
        z = body_radius * math.sin(angle)
        vertices.append([x, body_bottom, z])

    # Body top (slightly narrower)
    body_top = 0.3
    body_top_radius = 0.6
    for i in range(segments):
        angle = (i / segments) * 2 * math.pi
        x = body_top_radius * math.cos(angle)
        z = body_top_radius * math.sin(angle)
        vertices.append([x, body_top, z])

    # Shoulder
    shoulder_radius = 0.35
    shoulder_height = 0.5
    for i in range(segments):
        angle = (i / segments) * 2 * math.pi
        x = shoulder_radius * math.cos(angle)
        z = shoulder_radius * math.sin(angle)
        vertices.append([x, shoulder_height, z])

    # Neck bottom
    neck_radius = 0.2
    neck_bottom = 0.6
    for i in range(segments):
        angle = (i / segments) * 2 * math.pi
        x = neck_radius * math.cos(angle)
        z = neck_radius * math.sin(angle)
        vertices.append([x, neck_bottom, z])

    # Neck top
    neck_top = 1.0
    for i in range(segments):
        angle = (i / segments) * 2 * math.pi
        x = neck_radius * math.cos(angle)
        z = neck_radius * math.sin(angle)
        vertices.append([x, neck_top, z])

    # Cap
    cap_radius = 0.25
    cap_top = 1.3
    for i in range(segments):
        angle = (i / segments) * 2 * math.pi
        x = cap_radius * math.cos(angle)
        z = cap_radius * math.sin(angle)
        vertices.append([x, cap_top, z])

    return np.array(vertices, dtype=float)


def create_bottle_edges(segments=12):
    """Create edge indices for perfume bottle."""
    edges = []
    num_rings = 7  # base, body_bottom, body_top, shoulder, neck_bottom, neck_top, cap

    # Horizontal rings
    for ring in range(num_rings):
        offset = ring * segments
        for i in range(segments):
            edges.append((offset + i, offset + (i + 1) % segments))

    # Vertical connections between rings
    for ring in range(num_rings - 1):
        offset = ring * segments
        next_offset = (ring + 1) * segments
        # Connect every other vertex for cleaner look
        for i in range(0, segments, 2):
            edges.append((offset + i, next_offset + i))

    return edges


# --- Rendering ---


def render_wireframe_frame(
    vertices, edges, width, height, color, glow_color, angle_y, angle_x=0.15
):
    """Render a single wireframe frame."""
    # Create image with transparency
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rotate vertices
    rotated = rotate_x(vertices, angle_x)  # Slight tilt
    rotated = rotate_y(rotated, angle_y)

    # Project to 2D
    projected = project(rotated, width, height, camera_dist=4.5, fov=1.8)

    # Draw glow layer (thicker, more transparent)
    for edge in edges:
        p1 = projected[edge[0]]
        p2 = projected[edge[1]]
        draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=glow_color, width=4)

    # Draw main lines
    for edge in edges:
        p1 = projected[edge[0]]
        p2 = projected[edge[1]]
        draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=color, width=1)

    # Draw vertices as small dots for extra glow
    for v in projected:
        x, y = int(v[0]), int(v[1])
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)

    return img


def render_can_frame(vertices, edges, width, height, label_color, angle_y, segments=16):
    """Render a soda can frame with label."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rotate vertices
    rotated = rotate_x(vertices, 0.15)  # Slight tilt
    rotated = rotate_y(rotated, angle_y)

    # Project to 2D
    projected = project(rotated, width, height, camera_dist=4.0, fov=1.6)

    # Sort segments by depth for proper rendering
    segment_depths = []
    for i in range(segments):
        mid_z = (projected[i][2] + projected[segments + i][2]) / 2
        segment_depths.append((mid_z, i))
    segment_depths.sort(reverse=True)  # Back to front

    # Fill the can body with color
    for _, i in segment_depths:
        # Check if this segment is front-facing
        mid_angle = ((i + 0.5) / segments) * 2 * math.pi + angle_y

        # Draw quad for this segment
        bottom_left = projected[i]
        bottom_right = projected[(i + 1) % segments]
        top_left = projected[segments + i]
        top_right = projected[segments + (i + 1) % segments]

        # Shade based on angle (lighting from front)
        shade = (math.cos(mid_angle) + 1) / 2
        shade = 0.3 + 0.7 * shade  # Minimum brightness

        r = int(label_color[0] * shade)
        g = int(label_color[1] * shade)
        b = int(label_color[2] * shade)

        draw.polygon(
            [
                (bottom_left[0], bottom_left[1]),
                (bottom_right[0], bottom_right[1]),
                (top_right[0], top_right[1]),
                (top_left[0], top_left[1]),
            ],
            fill=(r, g, b, 230),
        )

    # Draw top ellipse (lid)
    top_points = [
        (projected[segments + i][0], projected[segments + i][1])
        for i in range(segments)
    ]
    draw.polygon(top_points, fill=(200, 200, 200, 200))

    # Draw highlight edges
    highlight_color = (255, 255, 255, 100)
    for i in range(segments):
        p1 = projected[segments + i]
        p2 = projected[segments + (i + 1) % segments]
        draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=highlight_color, width=1)

    # Add brand text on visible side
    text_angle = -angle_y
    text_visibility = math.cos(text_angle)

    if text_visibility > 0.2:  # Text is somewhat visible
        center_x = width // 2
        center_y = height // 2

        try:
            font_large = ImageFont.truetype("arial.ttf", 28)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font_large = ImageFont.load_default()
            font_small = font_large

        text_alpha = int(255 * min(1, text_visibility * 1.5))
        text_offset_x = int(math.sin(text_angle) * 20)

        # Brand name "NEON"
        text = "NEON"
        bbox = draw.textbbox((0, 0), text, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_x = center_x - text_width // 2 + text_offset_x
        text_y = center_y - 25

        # Cyan glow effect
        glow_color = (0, 255, 255, text_alpha // 2)
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx != 0 or dy != 0:
                    draw.text(
                        (text_x + dx, text_y + dy),
                        text,
                        font=font_large,
                        fill=glow_color,
                    )

        # Main text
        draw.text(
            (text_x, text_y), text, font=font_large, fill=(255, 255, 255, text_alpha)
        )

        # Subtext "COLA"
        subtext = "COLA"
        bbox2 = draw.textbbox((0, 0), subtext, font=font_small)
        subtext_width = bbox2[2] - bbox2[0]
        draw.text(
            (center_x - subtext_width // 2 + text_offset_x, text_y + 30),
            subtext,
            font=font_small,
            fill=(0, 255, 255, text_alpha),
        )

    return img


def create_wireframe_bottle_gif(output_path, num_frames=48, size=(200, 300)):
    """Create animated wireframe perfume bottle GIF."""
    vertices = create_bottle_vertices()
    edges = create_bottle_edges()

    # Neon cyan color for wireframe
    color = (0, 255, 255, 255)
    glow_color = (0, 255, 255, 100)

    frames = []
    for i in range(num_frames):
        angle = (i / num_frames) * 2 * math.pi
        frame = render_wireframe_frame(
            vertices, edges, size[0], size[1], color, glow_color, angle
        )
        frames.append(frame)

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
        disposal=2,
    )
    print(f"Created {output_path}")


def create_soda_can_gif(output_path, num_frames=48, size=(150, 200)):
    """Create animated soda can GIF."""
    segments = 24  # More segments for smoother look
    vertices = create_cylinder_vertices(radius=0.7, height=2.0, segments=segments)
    edges = create_cylinder_edges(segments=segments)

    # Hot pink/magenta for the can label
    label_color = (255, 20, 147)

    frames = []
    for i in range(num_frames):
        angle = (i / num_frames) * 2 * math.pi
        frame = render_can_frame(
            vertices, edges, size[0], size[1], label_color, angle, segments=segments
        )
        frames.append(frame)

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
        disposal=2,
    )
    print(f"Created {output_path}")


if __name__ == "__main__":
    # Create wireframe perfume bottle
    create_wireframe_bottle_gif(
        "neon_city/images/perfume_bottle_3d.gif", num_frames=60, size=(200, 300)
    )

    # Create branded soda can
    create_soda_can_gif(
        "neon_city/images/neon_cola_3d.gif", num_frames=60, size=(180, 280)
    )

    print("\nDone! New hologram assets created.")
