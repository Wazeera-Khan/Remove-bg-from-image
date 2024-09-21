import cv2
import numpy as np

# Load the image
image = cv2.imread('image.png')

# Create an initial mask
mask = np.zeros(image.shape[:2], np.uint8)

# Define the background and foreground models
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Define the rectangle around the foreground object
rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)

# Apply the grabCut algorithm
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask to segment the foreground and background
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply the mask to the image to remove the background
foreground = image * mask2[:, :, np.newaxis]

# Show the image
cv2.imshow('Foreground', foreground)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('image_no_background.png', foreground)
