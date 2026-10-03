"""
linkedin_kronecker_spongebob.py

Create a LinkedIn-ready square MP4 of SpongeBob rotating over the Krusty Krab
using an explicit chunked Kronecker-product rotation.

Required files in the same folder:
    spongebob.png
    krusty_krab.png

Install:
    python -m pip install numpy pillow imageio imageio-ffmpeg

Run:
    python linkedin_kronecker_spongebob.py

Output:
    spongebob_kronecker_linkedin.mp4
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

SPRITE_FILE = "spongebob.png"
BACKGROUND_FILE = "krusty_krab.png"
OUTPUT_FILE = "spongebob_kronecker_linkedin.mp4"

CANVAS_SIZE = 1080
FPS = 12
DURATION_SECONDS = 6
TOTAL_FRAMES = FPS * DURATION_SECONDS
SPRITE_HEIGHT = 330
CHUNK_SIZE = 700

TITLE = "Rotating SpongeBob with a Kronecker Product"
SUBTITLE = "Linear algebra -> pixel coordinates -> animation"


def fit_background(img, size):
    img = img.convert("RGB")
    w, h = img.size
    scale = max(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return img.crop((left, top, left + size, top + size)).convert("RGBA")


def prepare_sprite(img, target_height):
    img = img.convert("RGBA")
    scale = target_height / img.height
    target_width = int(img.width * scale)
    return img.resize((target_width, target_height), Image.Resampling.LANCZOS)


def kronecker_rotate_sprite(sprite, angle_degrees):
    src = np.asarray(sprite, dtype=np.uint8)
    h, w, _ = src.shape

    yy, xx = np.indices((h, w))
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    rgba = src.reshape(-1, 4)
    x = xx.ravel().astype(np.float64) - cx
    y = yy.ravel().astype(np.float64) - cy

    visible = rgba[:, 3] > 0
    x = x[visible]
    y = y[visible]
    rgba = rgba[visible]

    theta = math.radians(angle_degrees)

    R = np.array(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)],
        ],
        dtype=np.float64,
    )

    corners = np.array(
        [
            [-cx, -cy],
            [w - 1 - cx, -cy],
            [-cx, h - 1 - cy],
            [w - 1 - cx, h - 1 - cy],
        ],
        dtype=np.float64,
    )

    rotated_corners = corners @ R.T
    min_x, min_y = np.floor(rotated_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(rotated_corners.max(axis=0)).astype(int)

    out_w = max_x - min_x + 3
    out_h = max_y - min_y + 3
    out = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    total = len(x)

    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        m = end - start

        points = np.column_stack((x[start:end], y[start:end])).reshape(-1)

        # Explicit Kronecker operator: I_m ⊗ R
        K = np.kron(np.eye(m, dtype=np.float64), R)
        transformed = (K @ points).reshape(m, 2)

        new_x = np.rint(transformed[:, 0] - min_x + 1).astype(int)
        new_y = np.rint(transformed[:, 1] - min_y + 1).astype(int)

        valid = (
            (new_x >= 0)
            & (new_x < out_w)
            & (new_y >= 0)
            & (new_y < out_h)
        )

        out[new_y[valid], new_x[valid]] = rgba[start:end][valid]

    return Image.fromarray(out, "RGBA")


def add_text(frame):
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 42)
        subtitle_font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()

    panel = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.rounded_rectangle(
        (55, 55, 1025, 180),
        radius=24,
        fill=(0, 0, 0, 150),
    )
    frame = Image.alpha_composite(frame, panel)

    draw = ImageDraw.Draw(frame)
    draw.text((85, 78), TITLE, font=title_font, fill=(255, 255, 255, 255))
    draw.text((85, 132), SUBTITLE, font=subtitle_font, fill=(235, 235, 235, 255))

    return frame


def main():
    background = fit_background(Image.open(BACKGROUND_FILE), CANVAS_SIZE)
    sprite = prepare_sprite(Image.open(SPRITE_FILE), SPRITE_HEIGHT)

    frames = []
    print(f"Generating {TOTAL_FRAMES} frames...")

    for frame_index in range(TOTAL_FRAMES):
        angle = 360.0 * frame_index / TOTAL_FRAMES

        rotated = kronecker_rotate_sprite(sprite, angle)

        frame = background.copy()

        center_x = CANVAS_SIZE // 2
        center_y = int(CANVAS_SIZE * 0.61)
        left = center_x - rotated.width // 2
        top = center_y - rotated.height // 2

        frame.alpha_composite(rotated, dest=(left, top))
        frame = add_text(frame)

        frames.append(np.asarray(frame.convert("RGB")))

        print(
            f"Frame {frame_index + 1:02d}/{TOTAL_FRAMES} "
            f"angle={angle:6.1f} degrees"
        )

    print("Encoding MP4...")

    imageio.mimsave(
        OUTPUT_FILE,
        frames,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
