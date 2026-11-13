# Mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')
path_SOURCE = "/content/drive/MyDrive/path/to/your/source/directory"  # Update this path to your source directory

# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# sample image read
image_in = cv2.imread(path_SOURCE + 'file.png')
image_color = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

# display image in color
plt.figure(figsize=(6, 6))
plt.imshow(image_color)
plt.title('Color Image')
plt.axis('off')
plt.show()

# display image in grayscale
plt.figure(figsize=(6, 6))
plt.imshow(image_gray, cmap = 'grey', vmin = 0, vmax = 255)
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Undestanding Image Arrays and Pixels
height, width = image_gray.shape
print('Gray Image')
print(height, width)

height, width, planes = image_color.shape
print('Color Image')
print(height, width, planes)

print(image_gray[0:2, 0:5])

# accessing image values

pixel_value = image_gray[height//2, width//2]
print(f'Pixel value: {pixel_value}')

# a range of pixels
print('Pixel value range')
print(image_gray[0:2, 0:5])
image_cropped = image_gray[0:2, 0:5].copy()

# display image subset
plt.figure(figsize=(6, 6))
plt.imshow(image_gray[0:2, 0:5], cmap = 'gray', vmin = 0, vmax = 255)
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# save images
cv2.imwrite(path_SOURCE + 'astronaut_gray.png', image_gray)
cv2.imwrite(path_SOURCE + 'astronaut_cropped.png', image_cropped)