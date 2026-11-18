# ------------------------------------------------------------
# Python Libraries – Image Processing, Part 4: Histogram Operations
# Corrected and runnable version of the code shown in the video
# ------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_SOURCE = "/content/drive/MyDrive/Colab Notebooks/LTU Python Course/"
filename = "flowers_color_crop.png"          # or a scanned document (the video uses a postcard)

image_color = cv2.imread(path_SOURCE + filename)
if image_color is None:
    raise FileNotFoundError(f"Could not read {path_SOURCE + filename}")
img_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)


def show(images, titles, hist=True):
    """Images in the top row, their histograms underneath."""
    rows = 2 if hist else 1
    fig, axes = plt.subplots(rows, len(images), figsize=(4 * len(images), 3.5 * rows), squeeze=False)
    for k, (im, t) in enumerate(zip(images, titles)):
        axes[0, k].imshow(im, cmap='gray', vmin=0, vmax=255)
        axes[0, k].set_title(t)
        axes[0, k].axis('off')
        if hist:
            axes[1, k].hist(im.ravel(), bins=256, range=(0, 256))
            axes[1, k].set_xlim(0, 255)
            axes[1, k].set_title('Intensity Histogram')
    plt.tight_layout()
    plt.show()


# ------------------------------------------------
# 0. The histogram itself (slide example, 4x4)
# ------------------------------------------------
tiny = np.array([[15, 30, 96, 120],
                 [30, 183, 150, 96],
                 [96, 150, 57, 30],
                 [30, 57, 30, 15]], np.uint8)
values, counts = np.unique(tiny, return_counts=True)
print("value : count", dict(zip(values.tolist(), counts.tolist())))   # 30 -> 5, 57 -> 2

hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).ravel()   # 256 bins
print("most frequent gray level:", int(hist.argmax()))

# ------------------------------------------------
# 1. Thresholding: I_new = I_init > 127
# ------------------------------------------------
threshold_value = 127
_, image_bilevel = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
print("tiny > 127:\n", (tiny > 127).astype(np.uint8))

# ------------------------------------------------
# 2. Image negative: I_new = 255 - I_init
# ------------------------------------------------
image_negative = 255 - img_gray
print("negative of tiny:\n", 255 - tiny)

# ------------------------------------------------
# 3. Brightness: + / - delta, clipped to 0..255
# ------------------------------------------------
delta = 50
image_lighter = np.where(img_gray <= 255 - delta, img_gray + delta, 255).astype(np.uint8)
image_darker = np.where(img_gray > delta, img_gray - delta, 0).astype(np.uint8)
# Equivalent one-liners (saturating arithmetic):
assert np.array_equal(image_lighter, cv2.add(img_gray, delta))
assert np.array_equal(image_darker, cv2.subtract(img_gray, delta))
print("why clipping matters: uint8 220 + 50 =", np.array([220], np.uint8) + 50)   # [14]: wraps around!

show([img_gray, image_bilevel, image_negative, image_lighter, image_darker],
     ['Original', 'Threshold 127', 'Negative', 'Lighter (+50)', 'Darker (-50)'])

# ------------------------------------------------
# 4. Contrast stretching
# ------------------------------------------------
val_min, val_max = int(img_gray.min()), int(img_gray.max())
image_stretch = 255 * (np.float32(img_gray) - val_min) / (val_max - val_min)
image_stretch = image_stretch.astype(np.uint8)

# "Over-stretch": choose a narrower window; everything outside is clipped
val_min2, val_max2 = 100, 150
image_stretch2 = 255 * (np.float32(img_gray) - val_min2) / (val_max2 - val_min2)
image_stretch2 = np.clip(image_stretch2, 0, 255).astype(np.uint8)

# ------------------------------------------------
# 5. Histogram equalization (global and CLAHE)
# ------------------------------------------------
image_eq = cv2.equalizeHist(img_gray)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
image_clahe = clahe.apply(img_gray)

show([img_gray, image_stretch, image_stretch2, image_eq, image_clahe],
     [f'Original [{val_min},{val_max}]', 'Stretched', 'Over-stretched [100,150]',
      'Equalized', 'CLAHE'])

# ------------------------------------------------
# 6. Nonlinear: gamma correction and log transform
# ------------------------------------------------
gammas = (0.5, 1.0, 2.0)
gamma_imgs = [(np.power(img_gray / 255.0, g) * 255).astype(np.uint8) for g in gammas]

c = 255 / np.log(1 + float(img_gray.max()))          # scale so the maximum maps to 255
image_log = (c * np.log(1 + img_gray.astype(np.float32))).astype(np.uint8)

show(gamma_imgs + [image_log], [f'gamma = {g}' for g in gammas] + ['log transform'])
