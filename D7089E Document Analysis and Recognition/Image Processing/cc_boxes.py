# ------------------------------------------------------------
# Part 7 + Part 8 together: characters, words and lines as boxes
# (the application slide of Part 8, plus bounding boxes)
# ------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_SOURCE = "/content/drive/MyDrive/Colab Notebooks/LTU Python Course/"
filename = "typeset_text.png"   # any scan of printed text

img = cv2.imread(path_SOURCE + filename)
if img is None:
    raise FileNotFoundError(f"Could not read {path_SOURCE + filename}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Text -> white, background -> black (Otsu picks the threshold)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


def boxes_for(binary, se_shape=None, min_area=10, connectivity=8):
    """Optionally dilate, label components, and return their (x, y, w, h) boxes."""
    if se_shape is not None:
        binary = cv2.dilate(binary, np.ones(se_shape, np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
    boxes = []
    for i in range(1, n):                         # start at 1: label 0 is the background
        x, y, w, h, area = stats[i]
        if area >= min_area:                      # drop specks of noise
            boxes.append((x, y, w, h))
    return boxes


levels = {
    "characters (no dilation)": boxes_for(bw, None, min_area=10),
    "words (3x7 dilation)":     boxes_for(bw, (3, 7), min_area=30),
    "lines (3x12 dilation)":    boxes_for(bw, (3, 12), min_area=100),
}

fig, axes = plt.subplots(len(levels), 1, figsize=(9, 10))
for ax, (title, boxes) in zip(axes, levels.items()):
    vis = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{title}: {len(boxes)} boxes")
    ax.axis("off")
plt.tight_layout()
plt.show()
