# ------------------------------------------------------------
# Python Libraries – Image Processing, Part 5: Image Smoothing
# Corrected and runnable version of the code shown in the video
# ------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_SOURCE = "/content/drive/MyDrive/Colab Notebooks/LTU Python Course/"
filename = "Arvid Wallin-4.png"            # the old photograph used in the video

img_in = cv2.imread(path_SOURCE + filename)
if img_in is None:
    raise FileNotFoundError(f"Could not read {path_SOURCE + filename}")
img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)


def show(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 5))
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap='gray' if im.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(t)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ------------------------------------------------
# 0. Correlation by hand on the slide's 4x4 example
# ------------------------------------------------
A = np.array([[5, 19, 32, 40],
              [10, 61, 50, 32],
              [32, 50, 19, 10],
              [10, 19, 10, 5]], np.float32)
ones = np.ones((3, 3), np.float32)
print("sum of 3x3 neighbourhood, zero padding:\n",
      cv2.filter2D(A, -1, ones, borderType=cv2.BORDER_CONSTANT))     # top-left = 95
print("3x3 mean, zero padding:\n",
      np.round(cv2.filter2D(A, -1, ones / 9, borderType=cv2.BORDER_CONSTANT), 2))

# ------------------------------------------------
# 1. Uniform (box) smoothing with cv2.filter2D
# ------------------------------------------------
kernel3 = np.ones((3, 3), np.float32) / 9
image_smooth3 = cv2.filter2D(img_in, -1, kernel3)          # video: Image_smooth3 (capital I)

kernel5 = np.ones((5, 5), np.float32) / 25
image_smooth5 = cv2.filter2D(img_in, -1, kernel5)

kernel9 = np.ones((9, 9), np.float32) / 81
image_smooth9 = cv2.filter2D(img_in, -1, kernel9)

show([img_in, image_smooth3, image_smooth5, image_smooth9],
     ['Original Image', '3x3 Smoothing', '5x5 Smoothing', '9x9 Smoothing'])

# ------------------------------------------------
# 2. Weighted (Gaussian-like) 3x3 kernel: 1/16 [1 2 1; 2 4 2; 1 2 1]
# ------------------------------------------------
kernel_w = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]], np.float32) / 16
print("weights sum to", kernel_w.sum())                    # must be 1 to keep brightness
image_weighted = cv2.filter2D(img_in, -1, kernel_w)

# ------------------------------------------------
# 3. Other filters
# ------------------------------------------------
box_blur = cv2.boxFilter(img_in, -1, ksize=(9, 9), normalize=True)
gaussian_blur = cv2.GaussianBlur(img_in, ksize=(25, 25), sigmaX=3)
median_blur = cv2.medianBlur(img_in, ksize=9)
bilateral_blur = cv2.bilateralFilter(img_in, 9, 25, 25)     # d, sigmaColor, sigmaSpace

show([box_blur, gaussian_blur, median_blur, bilateral_blur],
     ['9x9 Box', 'sigma = 3 Gaussian', '9x9 Median', '9-25x25 Bilateral'])

# ------------------------------------------------
# 4. Which filter for which noise? (salt-and-pepper vs Gaussian noise)
# ------------------------------------------------
gray = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
rng = np.random.default_rng(0)

noisy_gauss = np.clip(gray + rng.normal(0, 20, gray.shape), 0, 255).astype(np.uint8)
noisy_sp = gray.copy()
r = rng.random(gray.shape)
noisy_sp[r < 0.03] = 0                                      # pepper
noisy_sp[r > 0.97] = 255                                    # salt

for name, noisy in (("Gaussian noise", noisy_gauss), ("salt & pepper", noisy_sp)):
    results = {
        "noisy": noisy,
        "box 5x5": cv2.blur(noisy, (5, 5)),
        "Gaussian 5x5": cv2.GaussianBlur(noisy, (5, 5), 0),
        "median 5": cv2.medianBlur(noisy, 5),
        "bilateral": cv2.bilateralFilter(noisy, 9, 50, 50),
    }
    print(name, "- PSNR vs clean image (higher = better):",
          {k: round(cv2.PSNR(gray, v), 1) for k, v in results.items()})
    show(list(results.values()), [f"{name}: {k}" for k in results])
