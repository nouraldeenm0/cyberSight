import cv2 # sudo apt install libgl1 is required
import numpy as np
from PIL import Image

# Get 2D numpy array from image
np_img = np.array(cv2.imread('contrast.jpg'))

# Get 1D array representing histogram

# Show image as grayscale
# with Image.open("contrast.jpg") as im:
    # im.convert("L").show()


print(np_img.shape)