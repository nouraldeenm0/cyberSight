"""
contrast = evalContrast(image)
contrast_stretched_image = contrast_stretch(image)
histogram_equalized_image = equalize_histogram(image)
gray-scale_transformed_image = gray_scale_transform(image)

Finally, for each of the three outputs, the contrast is re-evaluated. As per that, the following
components are to be implemented

1. Co-occurrence matrix and calculate contrast


"""

"""
import cv2 # sudo apt install libgl1 is required
import numpy as np

# Read the image
img = cv2.imread('contrast.jpg')

# Convert the image to a numpy array
np_img = np.array(img)

# Print the shape of the numpy array
print(np_img.shape)
"""

from PIL import Image
with Image.open("contrast.jpg") as im:
    im.convert("L")
    im.show()