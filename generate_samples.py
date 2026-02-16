"""
Generate side-by-side comparison images showing IR, VI, and Fused triplets.
Creates a 'samples' directory with individual triplet images and a combined grid.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import random

# --- Configuration ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IR_DIR = os.path.join(PROJECT_DIR, "data", "test", "ir")
VI_DIR = os.path.join(PROJECT_DIR, "data", "test", "vis")
FUSED_DIR = os.path.join(PROJECT_DIR, "outputs", "inference")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "samples")

NUM_SAMPLES = 12  # Generate 12 triplets (mix of day and night)

# We want a good mix of Day (D) and Night (N) scenes
SELECTED_NAMES = [
    # Night scenes
    "00004N",
    "00020N",
    "00040N",
    "00714N",
    "00766N",
    "00862N",
    # Day scenes
    "00099D",
    "00241D",
    "00383D",
    "00500D",
    "00621D",
    "01449D",
]


def create_triplet_image(ir_path, vi_path, fused_path, name, output_path):
    """Create a single side-by-side image: IR | VI | Fused with labels."""
    ir_img = Image.open(ir_path).convert("RGB")
    vi_img = Image.open(vi_path).convert("RGB")
    fused_img = Image.open(fused_path).convert("RGB")

    # Resize all to same dimensions
    target_w, target_h = 320, 240
    ir_img = ir_img.resize((target_w, target_h), Image.LANCZOS)
    vi_img = vi_img.resize((target_w, target_h), Image.LANCZOS)
    fused_img = fused_img.resize((target_w, target_h), Image.LANCZOS)

    # Create canvas: 3 images side by side with padding and labels
    padding = 10
    label_height = 35
    title_height = 40
    total_w = target_w * 3 + padding * 4
    total_h = target_h + label_height + title_height + padding * 2

    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 30))
    draw = ImageDraw.Draw(canvas)

    # Try to use a nice font, fall back to default
    try:
        label_font = ImageFont.truetype("arial.ttf", 18)
        title_font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
            title_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
        except:
            label_font = ImageFont.load_default()
            title_font = ImageFont.load_default()

    # Title
    scene_type = "Night Scene" if name.endswith("N") else "Day Scene"
    title_text = f"Sample: {name} ({scene_type})"
    bbox = draw.textbbox((0, 0), title_text, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, 8), title_text, fill=(255, 255, 255), font=title_font)

    # Paste images
    y_offset = title_height
    positions = [
        (padding, y_offset),
        (padding * 2 + target_w, y_offset),
        (padding * 3 + target_w * 2, y_offset),
    ]
    labels = ["Infrared (IR)", "Visible (VI)", "Fused Output"]
    colors = [(255, 100, 100), (100, 200, 100), (100, 180, 255)]

    for (x, y), img, label, color in zip(positions, [ir_img, vi_img, fused_img], labels, colors):
        # Draw border
        draw.rectangle([x - 2, y - 2, x + target_w + 1, y + target_h + 1], outline=color, width=2)
        canvas.paste(img, (x, y))
        # Draw label below
        bbox = draw.textbbox((0, 0), label, font=label_font)
        lw = bbox[2] - bbox[0]
        label_x = x + (target_w - lw) // 2
        label_y = y + target_h + 5
        draw.text((label_x, label_y), label, fill=color, font=label_font)

    canvas.save(output_path, quality=95)
    return canvas


def create_grid_image(triplet_images, output_path):
    """Create a large grid showing all triplets together."""
    if not triplet_images:
        return

    # Layout: 3 columns, N rows
    cols = 3
    rows = (len(triplet_images) + cols - 1) // cols

    sample_w, sample_h = triplet_images[0].size
    grid_padding = 15
    header_height = 50

    grid_w = cols * sample_w + (cols + 1) * grid_padding
    grid_h = rows * sample_h + (rows + 1) * grid_padding + header_height

    grid = Image.new("RGB", (grid_w, grid_h), (15, 15, 25))
    draw = ImageDraw.Draw(grid)

    # Header
    try:
        header_font = ImageFont.truetype("arial.ttf", 28)
    except:
        try:
            header_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
        except:
            header_font = ImageFont.load_default()

    header_text = "FusionNet - IR/VI Image Fusion Results"
    bbox = draw.textbbox((0, 0), header_text, font=header_font)
    hw = bbox[2] - bbox[0]
    draw.text(((grid_w - hw) // 2, 10), header_text, fill=(200, 220, 255), font=header_font)

    for i, img in enumerate(triplet_images):
        row = i // cols
        col = i % cols
        x = grid_padding + col * (sample_w + grid_padding)
        y = header_height + grid_padding + row * (sample_h + grid_padding)
        grid.paste(img, (x, y))

    grid.save(output_path, quality=95)
    print(f"Grid image saved to: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    triplet_images = []
    generated = 0

    for name in SELECTED_NAMES:
        ir_path = os.path.join(IR_DIR, f"{name}.png")
        vi_path = os.path.join(VI_DIR, f"{name}.png")
        fused_path = os.path.join(FUSED_DIR, f"fused_{name}.png")

        # Check all three exist
        if not os.path.exists(ir_path):
            print(f"  [SKIP] IR not found: {ir_path}")
            continue
        if not os.path.exists(vi_path):
            print(f"  [SKIP] VI not found: {vi_path}")
            continue
        if not os.path.exists(fused_path):
            print(f"  [SKIP] Fused not found: {fused_path}")
            continue

        output_path = os.path.join(OUTPUT_DIR, f"triplet_{name}.png")
        print(f"  Generating triplet for {name}...")
        img = create_triplet_image(ir_path, vi_path, fused_path, name, output_path)
        triplet_images.append(img)
        generated += 1

    print(f"\nGenerated {generated} individual triplet images in: {OUTPUT_DIR}")

    if triplet_images:
        grid_path = os.path.join(OUTPUT_DIR, "all_samples_grid.png")
        create_grid_image(triplet_images, grid_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
