# ------------------------------------------------------------
# Part 7 application: smear typeset text into words and lines
# with wide, flat structuring elements (3x7 and 3x12)
# ------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_SOURCE = "/content/drive/MyDrive/Colab Notebooks/LTU Python Course/"
filename = "typeset_text.png"   # any scanned page of printed text

img = cv2.imread(path_SOURCE + filename, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not read {path_SOURCE + filename}")

# Otsu picks the threshold; THRESH_BINARY_INV makes text white (255) on black,
# because OpenCV morphology grows and shrinks the WHITE pixels.
_, text_white = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

se_words = np.ones((3, 7), np.uint8)    # 3 rows x 7 columns: bridges gaps between letters
se_lines = np.ones((3, 12), np.uint8)   # 3 rows x 12 columns: also bridges gaps between words

words = cv2.dilate(text_white, se_words)
lines = cv2.dilate(text_white, se_lines)

# Each white blob is now (roughly) one word or one line; count them
n_words = cv2.connectedComponents(words)[0] - 1   # -1 removes the background label
n_lines = cv2.connectedComponents(lines)[0] - 1
print(f"blobs after 3x7: {n_words}   blobs after 3x12: {n_lines}")

fig, axes = plt.subplots(3, 1, figsize=(8, 10))
for ax, (title, im) in zip(axes, [("Binarized (text = white)", text_white),
                                 ("3x7 dilation -> words", words),
                                 ("3x12 dilation -> lines", lines)]):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
