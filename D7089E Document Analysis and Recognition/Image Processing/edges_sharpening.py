# ------------------------------------------------------------
# Python Libraries – Image Processing, Part 6:
# Edge Detectors and Image Sharpening
# Corrected and runnable version of the code shown in the video
# ------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_SOURCE = "/content/drive/MyDrive/Colab Notebooks/LTU Python Course/"
file_manuscript = "BL_Royal_Vincent_of_Beauvais.jpg"   # illuminated manuscript (edges)
file_photo = "Arvid Wallin-4.png"                       # old photograph (sharpening)


def load(name, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(path_SOURCE + name, flags)
    if img is None:
        raise FileNotFoundError(f"Could not read {path_SOURCE + name}")
    return img


def show(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap='gray' if im.ndim == 2 else None)
        ax.set_title(t)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


img_RGB = cv2.cvtColor(load(file_manuscript), cv2.COLOR_BGR2RGB)
img_gray = load(file_manuscript, cv2.IMREAD_GRAYSCALE)

# ------------------------------------------------
# 1. Simple difference kernels, Prewitt and Roberts with filter2D
# ------------------------------------------------
g = img_gray.astype(np.float32)                          # float: keep negative responses
kernels = {
    "diff x [1, -1]": np.array([[1, -1]], np.float32),
    "diff y [1; -1]": np.array([[1], [-1]], np.float32),
    "Prewitt x": np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32),
    "Prewitt y": np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32),
    "Roberts cross 1": np.array([[1, 0], [0, -1]], np.float32),    # classic diagonal Roberts
    "Roberts cross 2": np.array([[0, 1], [-1, 0]], np.float32),
}
responses = {name: np.abs(cv2.filter2D(g, -1, k)) for name, k in kernels.items()}
show(list(responses.values()), list(responses.keys()))

# ------------------------------------------------
# 2. Sobel x / y and gradient magnitude
# ------------------------------------------------
sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)   # d/dx -> VERTICAL edges
sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)   # d/dy -> HORIZONTAL edges

gradient_magnitude = cv2.magnitude(sobelx, sobely)         # sqrt(Gx^2 + Gy^2)
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)  # |x|, round, saturate -> uint8
gradient_direction = np.degrees(np.arctan2(sobely, sobelx))   # edge orientation (bonus)

show([img_RGB, img_gray, cv2.convertScaleAbs(sobelx), cv2.convertScaleAbs(sobely), gradient_magnitude],
     ['Manuscript Illumination', 'Gray Scale', 'Sobel Edge - X', 'Sobel Edge - Y', 'Sobel Edge Detection'])

# ------------------------------------------------
# 3. Canny edge detector
# ------------------------------------------------
image_cropped = img_gray[550:950, 300:700].copy()         # rows 550-949, cols 300-699
t_lower = 50
t_upper = 150
image_canny = cv2.Canny(image_cropped, t_lower, t_upper)
image_canny_smooth = cv2.Canny(cv2.GaussianBlur(image_cropped, (5, 5), 0), t_lower, t_upper)
show([image_cropped, image_canny, image_canny_smooth],
     ['Cropped', f'Canny ({t_lower}, {t_upper})', 'Gaussian 5x5 + Canny'])

# ------------------------------------------------
# 4. Image sharpening
# ------------------------------------------------
img_in = load(file_photo)
img_gray_p = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
img_RGB_p = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

kernel_lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)   # sum 0: edges only
kernel5 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)      # sum 1: image + edges
kernel9 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32)  # stronger (8-neighbour)

image_cropped_p = img_gray_p[30:180, 130:270]
sharpened_image = cv2.filter2D(image_cropped_p, -1, kernel5)
sharpened_image9 = cv2.filter2D(image_cropped_p, -1, kernel9)
edges_only = cv2.convertScaleAbs(cv2.filter2D(image_cropped_p.astype(np.float32), -1, kernel_lap))

# Unsharp masking (bonus): original + amount * (original - blurred)
blur = cv2.GaussianBlur(image_cropped_p, (0, 0), 2)
unsharp = cv2.addWeighted(image_cropped_p, 1.5, blur, -0.5, 0)

show([img_RGB_p[30:180, 130:270], image_cropped_p, edges_only, sharpened_image, sharpened_image9, unsharp],
     ['Original', 'Gray Scale', 'Laplacian (sum 0)', 'Sharpened Image, 5', 'Sharpened Image, 9',
      'Unsharp mask'])
