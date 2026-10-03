
Required:
    spongebob.png
    krusty_krab.png

Install:
    python -m pip install numpy pillow imageio imageio-ffmpeg

Run:
    python linkedin_kronecker_genius.py
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio


SPRITE_FILE = "spongebob.png"
BACKGROUND_FILE = "krusty_krab.png"
OUTPUT_FILE = "spongebob_kronecker_genius.mp4"

CANVAS = 1080
FPS = 24
DURATION = 8
TOTAL_FRAMES = FPS * DURATION

SPRITE_HEIGHT = 360
CHUNK_SIZE = 650


def load_font(size, bold=False):
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_TITLE = load_font(42, bold=True)
FONT_SUB = load_font(24)
FONT_MONO = load_font(26)
FONT_SMALL = load_font(20)


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


def ease_in_out(t):
    # smoothstep-ish cinematic easing
    return 0.5 - 0.5 * math.cos(math.pi * t)


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
    c, s = math.cos(theta), math.sin(theta)

    R = np.array(
        [[c, -s],
         [s,  c]],
        dtype=np.float64
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

        # Explicit Kronecker-product operator
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

    return Image.fromarray(out, "RGBA"), R


def add_vignette(frame):
    arr = np.asarray(frame).astype(np.float32)
    h, w = arr.shape[:2]

    yy, xx = np.indices((h, w))
    dx = (xx - w / 2) / (w / 2)
    dy = (yy - h / 2) / (h / 2)
    r = np.sqrt(dx * dx + dy * dy)

    factor = np.clip(1.0 - 0.28 * np.maximum(r - 0.25, 0), 0.68, 1.0)
    arr[:, :, :3] *= factor[:, :, None]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


def draw_hud(frame, angle, R):
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # top title panel
    draw.rounded_rectangle(
        (48, 44, 1032, 168),
        radius=26,
        fill=(12, 14, 18, 165),
        outline=(255, 255, 255, 32),
        width=1,
    )

    draw.text(
        (78, 66),
        "KRONECKER ROTATION",
        font=FONT_TITLE,
        fill=(255, 255, 255, 255),
    )

    draw.text(
        (80, 120),
        "Applying  Iₙ ⊗ R(θ)  to image-space coordinates",
        font=FONT_SUB,
        fill=(220, 224, 232, 255),
    )

    # left equation panel
    draw.rounded_rectangle(
        (54, 760, 520, 1012),
        radius=24,
        fill=(10, 12, 16, 178),
        outline=(255, 255, 255, 28),
        width=1,
    )

    draw.text(
        (82, 790),
        "p′ = (Iₙ ⊗ R(θ)) p",
        font=FONT_MONO,
        fill=(255, 255, 255, 255),
    )

    c, ms = R[0, 0], R[0, 1]
    s, cc = R[1, 0], R[1, 1]

    matrix_text = (
        f"R(θ) = [{c: .3f}  {ms: .3f}]\n"
        f"       [{s: .3f}  {cc: .3f}]"
    )

    draw.multiline_text(
        (82, 850),
        matrix_text,
        font=FONT_SMALL,
        fill=(218, 222, 230, 255),
        spacing=8,
    )

    draw.text(
        (82, 958),
        "explicit block rotation operator",
        font=FONT_SMALL,
        fill=(170, 176, 186, 255),
    )

    # right telemetry panel
    draw.rounded_rectangle(
        (738, 790, 1024, 1008),
        radius=24,
        fill=(10, 12, 16, 178),
        outline=(255, 255, 255, 28),
        width=1,
    )

    draw.text(
        (770, 818),
        f"θ = {angle:6.1f}°",
        font=FONT_MONO,
        fill=(255, 255, 255, 255),
    )

    draw.text(
        (770, 870),
        f"chunk = {CHUNK_SIZE}",
        font=FONT_SMALL,
        fill=(213, 218, 226, 255),
    )

    draw.text(
        (770, 910),
        "transform: Kronecker",
        font=FONT_SMALL,
        fill=(213, 218, 226, 255),
    )

    draw.text(
        (770, 950),
        "space: pixel coordinates",
        font=FONT_SMALL,
        fill=(170, 176, 186, 255),
    )

    return Image.alpha_composite(frame, overlay)


def main():
    background = fit_background(Image.open(BACKGROUND_FILE), CANVAS)
    sprite = prepare_sprite(Image.open(SPRITE_FILE), SPRITE_HEIGHT)

    writer = imageio.get_writer(
        OUTPUT_FILE,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    print(f"Generating {TOTAL_FRAMES} frames...")

    for frame_index in range(TOTAL_FRAMES):
        t = frame_index / (TOTAL_FRAMES - 1)

        # two full rotations with smooth acceleration/deceleration
        angle = 720.0 * ease_in_out(t)

        rotated, R = kronecker_rotate_sprite(sprite, angle)

        frame = background.copy()

        # subtle translucent wash for cinematic readability
        wash = Image.new("RGBA", frame.size, (0, 0, 0, 32))
        frame = Image.alpha_composite(frame, wash)

        center_x = CANVAS // 2
        center_y = 525

        left = center_x - rotated.width // 2
        top = center_y - rotated.height // 2

        # soft glow/shadow
        shadow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        shadow_sprite = rotated.copy()
        alpha = shadow_sprite.getchannel("A").point(lambda p: int(p * 0.30))
        shadow_sprite.putalpha(alpha)
        shadow.alpha_composite(shadow_sprite, dest=(left + 14, top + 16))
        frame = Image.alpha_composite(frame, shadow)

        frame.alpha_composite(rotated, dest=(left, top))

        frame = add_vignette(frame)
        frame = draw_hud(frame, angle, R)

        writer.append_data(np.asarray(frame.convert("RGB")))

        print(
            f"Frame {frame_index + 1:03d}/{TOTAL_FRAMES} "
            f"angle={angle:7.2f}°"
        )

    writer.close()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
