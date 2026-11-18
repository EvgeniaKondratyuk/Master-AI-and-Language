# ------------------------------------------------------------
# Python Libraries – Image Processing, Part 7: Image Morphology
# Corrected and runnable version of the code shown in the video
# ------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_SOURCE = "/content/drive/MyDrive/Colab Notebooks/LTU Python Course/"
filename = "greekBW.png"

img_in = cv2.imread(path_SOURCE + filename)
if img_in is None:
    raise FileNotFoundError(f"Could not read {path_SOURCE + filename}")

image_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

# Convert to binary using a threshold.
# Dark ink (< 127) becomes 1 = "on", paper becomes 0 = "off".
threshold_value = 127
image_bilevel = (image_gray < threshold_value).astype(np.uint8)  # video had image_bilevel here (bug)

# Define a structuring element (3x3 square of ones)
se = np.ones((3, 3), np.uint8)

# Dilate / erode: pass the number of passes as iterations=, not as the 3rd positional argument
img_dilate = cv2.dilate(image_bilevel, se, iterations=1)
img_erode = cv2.erode(image_bilevel, se, iterations=1)

# Open (erode then dilate) / close (dilate then erode)
img_open = cv2.morphologyEx(image_bilevel, cv2.MORPH_OPEN, se)
img_close = cv2.morphologyEx(image_bilevel, cv2.MORPH_CLOSE, se)

# Display: 1 = ink, so use gray_r to show ink as black on white
results = [("Original", image_bilevel), ("Dilation", img_dilate), ("Erosion", img_erode),
           ("Opening", img_open), ("Closing", img_close)]
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, (title, im) in zip(axes, results):
    ax.imshow(im, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()

# Save: scale 0/1 to 0/255 first, otherwise the PNG looks completely black
cv2.imwrite(path_SOURCE + "greek_dilate.png", 255 - img_dilate * 255)
