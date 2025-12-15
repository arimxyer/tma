# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pillow",
# ]
# ///
"""
Convert an image to a pixel art silhouette sprite.

Usage:
    uv run create_sprite.py <input_image> [--height 48] [--output sprite.png]
"""

import argparse
from PIL import Image


def create_silhouette(input_path, output_path="sprite.png", target_height=48, threshold=128, invert=False):
    """
    Convert image to pixel art silhouette.

    Args:
        input_path: Path to source image
        output_path: Where to save the sprite
        target_height: Height of output sprite in pixels
        threshold: Grayscale threshold (0-255) for silhouette cutoff
        invert: If True, light pixels become silhouette instead of dark
    """
    print(f"Loading {input_path}...")
    img = Image.open(input_path)

    # Calculate target width maintaining aspect ratio
    aspect = img.width / img.height
    target_width = int(target_height * aspect)

    print(f"Original: {img.width}x{img.height} -> Target: {target_width}x{target_height}")

    # Resize first (use LANCZOS for quality downscale)
    img = img.resize((target_width, target_height), Image.LANCZOS)

    # Convert to grayscale
    gray = img.convert('L')

    # Create RGBA output with transparent background
    result = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))

    # Threshold to create silhouette
    pixels = gray.load()
    result_pixels = result.load()

    if pixels is None or result_pixels is None:
        raise ValueError("Failed to load pixel data")

    silhouette_color = (5, 2, 8, 255)  # Dark purple to match scene

    for y in range(target_height):
        for x in range(target_width):
            pixel_value = pixels[x, y]
            if isinstance(pixel_value, tuple):
                pixel_value = pixel_value[0]
            is_silhouette = pixel_value < threshold
            if invert:
                is_silhouette = not is_silhouette
            if is_silhouette:
                result_pixels[x, y] = silhouette_color

    result.save(output_path, 'PNG')
    print(f"Saved {target_width}x{target_height} sprite to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image to pixel art silhouette")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--output", "-o", default="sprite.png", help="Output path")
    parser.add_argument("--height", type=int, default=48, help="Target height in pixels")
    parser.add_argument("--threshold", "-t", type=int, default=128, help="Grayscale threshold (0-255)")
    parser.add_argument("--invert", "-i", action="store_true", help="Invert silhouette (light becomes solid)")

    args = parser.parse_args()
    create_silhouette(args.input, args.output, args.height, args.threshold, args.invert)
