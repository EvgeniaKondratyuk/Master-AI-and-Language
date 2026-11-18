# ------------------------------------------------------------
# Python Libraries – Image Processing, Part 3: Colors and Color Spaces
# Corrected and runnable version of the code shown in the video
# ------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_SOURCE = "/content/drive/MyDrive/Colab Notebooks/LTU Python Course/"
filename = "flowers_color_crop.png"

img_in = cv2.imread(path_SOURCE + filename)            # OpenCV order: B, G, R
if img_in is None:
    raise FileNotFoundError(f"Could not read {path_SOURCE + filename}")


def show_row(images, titles, cmaps=None):
    cmaps = cmaps or [None] * len(images)
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    for ax, im, t, cm in zip(axes, images, titles, cmaps):
        ax.imshow(im, cmap=cm, vmin=0, vmax=255) if im.ndim == 2 else ax.imshow(im)
        ax.set_title(t)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ------------------------------------------
# 1. RGB: separate the three color channels
# ------------------------------------------
img_RGB = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
image_channelR = img_RGB[:, :, 0]
image_channelG = img_RGB[:, :, 1]
image_channelB = img_RGB[:, :, 2]
show_row([img_RGB, image_channelR, image_channelG, image_channelB],
         ["RGB", "R channel", "G channel", "B channel"], [None, 'gray', 'gray', 'gray'])

# Same thing straight from the BGR image (note the reversed order!)
b, g, r = cv2.split(img_in)
assert np.array_equal(r, image_channelR)

# Tinted view, like on the slides: keep one channel, zero the others
tinted = []
for k in range(3):
    t = np.zeros_like(img_RGB)
    t[:, :, k] = img_RGB[:, :, k]
    tinted.append(t)
show_row(tinted, ["R only", "G only", "B only"])

# ---------------------------------------
# 2. HSV: hue, saturation, value
# ---------------------------------------
img_HSV = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)      # convert from the BGR original
image_channelH = img_HSV[:, :, 0]                      # video had image_HSV (undefined)
image_channelS = img_HSV[:, :, 1]
image_channelV = img_HSV[:, :, 2]
print("H range:", image_channelH.min(), "-", image_channelH.max(), "(OpenCV hue is 0..179)")
show_row([img_RGB, image_channelH, image_channelS, image_channelV],
         ["RGB", "Hue (0-179)", "Saturation", "Value"], [None, 'hsv', 'gray', 'gray'])

# Hue-based selection: keep strongly coloured yellow pixels (H about 20-35 in OpenCV units)
mask_yellow = cv2.inRange(img_HSV, (20, 100, 100), (35, 255, 255))
yellow_only = cv2.bitwise_and(img_RGB, img_RGB, mask=mask_yellow)
show_row([img_RGB, mask_yellow, yellow_only], ["RGB", "mask: yellow hues", "yellow pixels only"],
         [None, 'gray', None])

# ---------------------------------------
# 3. Convert to gray scale
# ---------------------------------------
img_Gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)   # RGB input -> RGB2GRAY
img_Gray2 = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)   # BGR input -> BGR2GRAY (same result)
img_Wrong = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2GRAY)  # mismatch: R and B weights swapped
print("RGB2GRAY == BGR2GRAY on matching input:", np.array_equal(img_Gray, img_Gray2))
print("mean abs error of the mismatched version:",
      np.abs(img_Gray.astype(int) - img_Wrong.astype(int)).mean().round(1))
show_row([img_Gray, img_Wrong], ["correct gray", "wrong flag (R/B weights swapped)"], ['gray', 'gray'])

# ---------------------------------------
# 4. Indexed color: 256-colour palette
# ---------------------------------------
from PIL import Image
pal_img = Image.fromarray(img_RGB).quantize(colors=256)     # palette + 8-bit index per pixel
indices = np.array(pal_img)                                  # 2D array of palette indices 0..255
palette = np.array(pal_img.getpalette()[:256 * 3]).reshape(-1, 3)
print("index image:", indices.shape, indices.dtype, " palette:", palette.shape)
print("palette entry 0 (R, G, B):", palette[0], " HTML: #%02X%02X%02X" % tuple(palette[0]))
rebuilt = palette[indices].astype(np.uint8)                  # look every index up in the table
show_row([img_RGB, rebuilt], ["24-bit RGB", "8-bit indexed (256 colours)"])
