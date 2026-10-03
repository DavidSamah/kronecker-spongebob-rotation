from PIL import Image
import numpy as np
import math

spongebob = Image.open("spongebob.png").convert("RGBA")

pixels = np.array(spongebob)

height, width, channels = pixels.shape

yy, xx = np.indices((height, width))

cx = (width - 1) / 2
cy = (height - 1) / 2

x_centered = xx - cx
y_centered = yy - cy

alpha = pixels[:, :, 3]
visible = alpha > 0

visible_x = x_centered[visible]
visible_y = y_centered[visible]
visible_pixels = pixels[visible]

angle_degrees = 30
theta = math.radians(angle_degrees)

R = np.array([
    [math.cos(theta), -math.sin(theta)],
    [math.sin(theta),  math.cos(theta)]
])

chunk_size = 1000

rotated_chunks = []

for start in range(0, len(visible_x), chunk_size):

    end = min(start + chunk_size, len(visible_x))

    x_chunk = visible_x[start:end]
    y_chunk = visible_y[start:end]

    points = np.column_stack(
        (x_chunk, y_chunk)
    ).reshape(-1)

    m = len(x_chunk)

    K = np.kron(
        np.eye(m),
        R
    )

    rotated = K @ points

    rotated_chunks.append(
        rotated.reshape(-1, 2)
    )

all_rotated = np.vstack(rotated_chunks)

# Find bounds of rotated coordinates
min_x = np.floor(all_rotated[:, 0].min()).astype(int)
max_x = np.ceil(all_rotated[:, 0].max()).astype(int)

min_y = np.floor(all_rotated[:, 1].min()).astype(int)
max_y = np.ceil(all_rotated[:, 1].max()).astype(int)

rotated_width = max_x - min_x + 1
rotated_height = max_y - min_y + 1

print("Rotated canvas size:")
print(rotated_width, "x", rotated_height)

# Create transparent output canvas
rotated_array = np.zeros(
    (rotated_height, rotated_width, 4),
    dtype=np.uint8
)

# Shift coordinates so they start at 0
new_x = np.rint(
    all_rotated[:, 0] - min_x
).astype(int)

new_y = np.rint(
    all_rotated[:, 1] - min_y
).astype(int)

# Put original colors at rotated positions
rotated_array[
    new_y,
    new_x
] = visible_pixels

rotated_image = Image.fromarray(
    rotated_array,
    "RGBA"
)

rotated_image.save(
    "spongebob_rotated.png"
)

print("Saved: spongebob_rotated.png")