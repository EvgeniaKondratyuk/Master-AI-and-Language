# ------------------------------------------------------------
# Python Libraries – Image Processing, Part 2:
# Image Subsampling and Quantization
# Corrected and runnable version of the code shown in the video
# ------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_SOURCE = "/content/drive/MyDrive/Colab Notebooks/LTU Python Course/"
filename = "flowers_color_crop.png"

image_color = cv2.imread(path_SOURCE + filename)
if image_color is None:
    raise FileNotFoundError(f"Could not read {path_SOURCE + filename}")
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
# (If you load a gray file directly, use cv2.imread(path, cv2.IMREAD_GRAYSCALE);
#  without the flag OpenCV returns 3 channels and "height, width = .shape" fails.)

height, width = image_gray.shape


def show_row(images, titles, vmax=255):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(t)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ----------------------------------------------
# 1. Spatial subsampling: 300 -> 150 -> 75 -> 37
# ----------------------------------------------
sizes = [(width // f, height // f) for f in (1, 2, 4, 8)]          # dsize is (WIDTH, HEIGHT)
small = [cv2.resize(image_gray, s, interpolation=cv2.INTER_NEAREST) for s in sizes]
# Blow each one back up to the original size so the loss of detail is visible
shown = [cv2.resize(im, (width, height), interpolation=cv2.INTER_NEAREST) for im in small]
show_row(shown, [f"{s[0]} x {s[1]}" for s in sizes])

# --------------------------------------------------------------
# 2. Three ways to downsample by 4 (impulse / average / maximum)
# --------------------------------------------------------------
f = 4
impulse = image_gray[::f, ::f]                                       # take every 4th pixel
average = cv2.resize(image_gray, (width // f, height // f), interpolation=cv2.INTER_AREA)
h4, w4 = (height // f) * f, (width // f) * f                          # crop to a multiple of f
maxpool = image_gray[:h4, :w4].reshape(h4 // f, f, w4 // f, f).max(axis=(1, 3))
show_row([impulse, average, maxpool], ["impulse (every 4th)", "average (INTER_AREA)", "max-pool"])

# ------------------------------------------------------------
# 3. Upsampling = interpolation: nearest vs linear vs cubic
# ------------------------------------------------------------
low_res_image = cv2.resize(image_gray, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
ups = [cv2.resize(low_res_image, (width, height), interpolation=m)
       for m in (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC)]
show_row(ups, ["INTER_NEAREST (blocky)", "INTER_LINEAR (smooth)", "INTER_CUBIC (smooth, sharper)"])

# The same with scale factors instead of a target size:
img_res = cv2.resize(image_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)   # < 1: reduce
img_big = cv2.resize(image_gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)  # > 1: enlarge
print("fx=fy=0.5 ->", img_res.shape, "  fx=fy=2 ->", img_big.shape)

# ------------------------------------------
# 4. Intensity quantization: 256 -> 64/16/4
# ------------------------------------------
quantized = []
for levels in (64, 16, 4):
    step = 256 // levels                         # 4, 16, 64
    q = (image_gray // step) * step              # e.g. 4 levels: 0, 64, 128, 192
    quantized.append(q)
    print(f"{levels:>2} levels -> step {step:>2}, values used: {len(np.unique(q))}")
show_row([image_gray] + quantized,
         ["Original 256 gray levels", "64 gray levels", "16 gray levels", "4 gray levels"])

# --------------------------------------------
# 5. Bilevel (1-bit) images with a threshold
# --------------------------------------------
results, titles = [], []
for threshold_value in (64, 127, 192):
    _, image_bilevel = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)
    results.append(image_bilevel)
    titles.append(rf'Thresholded image $\theta$ = {threshold_value}')   # raw string: \t is a TAB!
show_row(results, titles)
