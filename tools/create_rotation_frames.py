"""
Generate rotation animation frames from a static image.
Creates a GIF of the image rotating in 3D space (turntable effect).
"""

import math
import os
from PIL import Image


def create_rotation_frames(
    input_path: str,
    output_path: str,
    num_frames: int = 60,
    duration_ms: int = 50,
    rotation_type: str = "y_axis",  # "y_axis", "card_flip", "wobble"
):
    """
    Generate rotation frames from a static image.

    Args:
        input_path: Path to source image
        output_path: Path for output GIF
        num_frames: Number of frames in the rotation
        duration_ms: Duration per frame in milliseconds
        rotation_type: Type of rotation effect
    """
    # Load the source image
    img = Image.open(input_path).convert("RGBA")
    width, height = img.size

    frames = []

    for i in range(num_frames):
        # Calculate rotation angle (0 to 2*PI)
        angle = (i / num_frames) * 2 * math.pi

        if rotation_type == "y_axis":
            # Y-axis rotation (turntable) - scale X based on cos(angle)
            frame = create_y_rotation_frame(img, angle, width, height)
        elif rotation_type == "card_flip":
            # Card flip with perspective
            frame = create_card_flip_frame(img, angle, width, height)
        elif rotation_type == "wobble":
            # Gentle wobble rotation
            frame = create_wobble_frame(img, angle, width, height)
        else:
            frame = img.copy()

        frames.append(frame)

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,  # Clear frame before drawing next
    )

    print(f"Created {output_path} with {num_frames} frames")
    return output_path


def create_y_rotation_frame(img, angle, width, height):
    """Create a frame showing Y-axis rotation (turntable effect)."""
    # Calculate the apparent width based on rotation angle
    scale_x = abs(math.cos(angle))
    scale_x = max(0.05, scale_x)  # Never fully disappear

    new_width = max(1, int(width * scale_x))

    # Determine if we're seeing the "back" (mirror the image)
    showing_back = math.cos(angle) < 0

    # Resize image to new width
    if showing_back:
        # Flip horizontally for back view
        resized = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        resized = img.copy()

    resized = resized.resize((new_width, height), Image.Resampling.LANCZOS)

    # Create frame with original dimensions, paste centered
    frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    x_offset = (width - new_width) // 2
    frame.paste(resized, (x_offset, 0))

    # Add edge highlight when rotated
    if scale_x < 0.7:
        # Add a subtle edge glow effect
        edge_intensity = int((1 - scale_x) * 100)
        add_edge_highlight(frame, x_offset, new_width, height, edge_intensity, angle)

    return frame


def create_card_flip_frame(img, angle, width, height):
    """Create a frame with perspective card flip effect."""
    # Similar to y_rotation but with perspective distortion
    scale_x = abs(math.cos(angle))
    scale_x = max(0.05, scale_x)

    new_width = max(1, int(width * scale_x))
    showing_back = math.cos(angle) < 0

    # Create the base rotated image
    if showing_back:
        source = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        source = img.copy()

    # Apply perspective transform for more 3D look
    # When edge-on, add slight vertical stretch to simulate perspective
    perspective_stretch = 1.0 + (1 - scale_x) * 0.1
    new_height = min(int(height * perspective_stretch), height + 20)

    resized = source.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create frame
    frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    x_offset = (width - new_width) // 2
    y_offset = (height - new_height) // 2

    # Ensure we don't paste outside bounds
    paste_y = max(0, y_offset)
    frame.paste(resized, (x_offset, paste_y))

    return frame


def create_wobble_frame(img, angle, width, height):
    """Create a gentle wobble/tilt effect."""
    # Wobble is a smaller rotation, oscillating back and forth
    wobble_angle = math.sin(angle) * 0.3  # Max 0.3 radians (~17 degrees)

    scale_x = math.cos(wobble_angle)
    new_width = max(1, int(width * scale_x))

    resized = img.resize((new_width, height), Image.Resampling.LANCZOS)

    frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    x_offset = (width - new_width) // 2
    frame.paste(resized, (x_offset, 0))

    return frame


def add_edge_highlight(frame, x_offset, obj_width, height, intensity, angle):
    """Add a highlight on the leading edge to simulate 3D lighting."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(frame)

    # Determine which edge is "facing the light"
    if math.sin(angle) > 0:
        edge_x = x_offset
    else:
        edge_x = x_offset + obj_width - 1

    # Draw a subtle highlight line
    highlight_color = (255, 255, 255, min(intensity, 150))
    draw.line([(edge_x, 0), (edge_x, height - 1)], fill=highlight_color, width=2)


def process_hologram_assets():
    """Process all hologram assets marked for rotation."""
    assets_to_rotate = [
        (
            "neon_city/images/dior_perfume.png",
            "neon_city/images/dior_perfume_rotating.gif",
        ),
        (
            "neon_city/images/en-card-cherry.png",
            "neon_city/images/en-card-cherry_rotating.gif",
        ),
    ]

    for input_path, output_path in assets_to_rotate:
        if os.path.exists(input_path):
            print(f"Processing {input_path}...")
            create_rotation_frames(
                input_path,
                output_path,
                num_frames=48,  # Smooth rotation
                duration_ms=40,  # ~25 FPS
                rotation_type="y_axis",
            )
        else:
            print(f"Warning: {input_path} not found")


if __name__ == "__main__":
    process_hologram_assets()
    print(
        "\nDone! Update config.py to use the new _rotating.gif files as 'animated' type."
    )
