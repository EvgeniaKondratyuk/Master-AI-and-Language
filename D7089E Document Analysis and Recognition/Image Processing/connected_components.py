# ------------------------------------------------------------
# Python Libraries – Image Processing, Part 8: Connected Components
# Corrected and runnable version of the code shown in the video
# ------------------------------------------------------------
import cv2
import numpy as np

try:                                   # in Colab
    from google.colab.patches import cv2_imshow
except ImportError:                    # outside Colab: show with matplotlib instead
    import matplotlib.pyplot as plt
    def cv2_imshow(img):
        plt.figure(figsize=(8, 3))
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

path_SOURCE = "/content/drive/MyDrive/Colab Notebooks/LTU Python Course/"
filename = "word_MountainsG.png"

img_in = cv2.imread(path_SOURCE + filename)
if img_in is None:
    raise FileNotFoundError(f"Could not read {path_SOURCE + filename}")
cv2_imshow(img_in)

# Convert to gray scale
img_inG = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

# Threshold so it becomes binary. INV makes the dark text white (255):
# connectedComponents labels the NON-ZERO pixels, so the text must be the white part.
ret, img_BW = cv2.threshold(img_inG, 127, 255, cv2.THRESH_BINARY_INV)
cv2_imshow(img_BW)

# Label the components (4-connectivity, as in the video)
num_labels, labels_im = cv2.connectedComponents(img_BW, connectivity=4)  # video had imgBW (bug)
print("Num CCs: ", num_labels)          # note that one label (0) is the background
print("Objects:  ", num_labels - 1)

# Display: map component labels to a hue value
# (OpenCV hue runs 0..179, so 179 = the last colour of the wheel)
label_hue = np.uint8(179 * labels_im / max(np.max(labels_im), 1))  # max(..,1) avoids /0 on an empty image
blank_ch = 255 * np.ones_like(label_hue)                             # full saturation and value
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
# Convert HSV -> BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
# Set the background label to black
labeled_img[label_hue == 0] = 0
cv2_imshow(labeled_img)
