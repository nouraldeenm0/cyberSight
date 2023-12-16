from typing import List
import numpy as np
import cv2

class QuadTree:
    def __init__(self, node: np.ndarray):
        self.node = node
        self.children = []

    def __init__(self, node: np.ndarray, children: List['QuadTree'] = None):
        self.node = node
        self.children = children

    def __str__(self):
        # print node and children recursively in a tree-like structure
        return str(self.node) + '\n' + str(self.children)
         
def create_image() -> np.ndarray:
    image_data = [
        [1, 1, 1, 1, 1, 1, 1, 2],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [3, 1, 4, 9, 9, 8, 1, 0],
        [1, 1, 8, 8, 8, 4, 1, 0],
        [1, 1, 6, 6, 6, 3, 1, 0],
        [1, 1, 5, 6, 6, 3, 1, 0],
        [1, 1, 5, 6, 6, 2, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0]
    ]
    return np.array(image_data, dtype='uint8')

def generate_image(size: int) -> np.ndarray:
    return np.random.randint(0, 256, (size, size), dtype='uint8')

def split_one_iteration(img: np.ndarray, threshold: int) -> List[np.ndarray]:
    max_val: int = img.max()
    min_val: int = img.min()
    is_homogeneous: bool = max_val - min_val < threshold

    if is_homogeneous:
        return [img]

    img_dimensions: tuple = img.shape # returns a tuple of (height, width)
    w_midpnt: int = img_dimensions[1] // 2
    h_midpnt: int = img_dimensions[0] // 2

    return [img[:h_midpnt, :w_midpnt], img[:h_midpnt, w_midpnt:],
            img[h_midpnt:, :w_midpnt], img[h_midpnt:, w_midpnt:]]

def split(img: np.ndarray, threshold: int) -> List[np.ndarray]:
    max_val: int = img.max()
    min_val: int = img.min()

    # Base case: if image is homogeneous, return it wrapped in a list
    if max_val - min_val < threshold:
        return [img]

    # Recursive case: split image into four quadrants and recurse
    img_dimensions: tuple(int, int) = img.shape # returns a tuple of (height, width)
    w_midpnt: int = img_dimensions[1] // 2
    h_midpnt: int = img_dimensions[0] // 2

    quadrants: List[np.ndarray] = [img[:h_midpnt, :w_midpnt], img[:h_midpnt, w_midpnt:],
                                   img[h_midpnt:, :w_midpnt], img[h_midpnt:, w_midpnt:]]
    result: List[np.ndarray] = []
    for quadrant in quadrants:
        split_img = split(quadrant, threshold)

        # The extend() method appends all items from the given list (split_img) to the end of the current list (result)
        result.extend(split_img)
    return result

def splitQ(img: np.ndarray, threshold: int) -> QuadTree:
    """Split the image into quadrants if the difference between max and min pixel values is above the threshold."""
    max_val: int = img.max()
    min_val: int = img.min()

    # Base case: if image is homogeneous, return it as a QuadTree with no children
    if max_val - min_val < threshold:
        return QuadTree(img)

    # Recursive case: split image into four quadrants and recurse
    img_dimensions: tuple = img.shape # returns a tuple of (height, width)
    w_midpnt: int = img_dimensions[1] // 2
    h_midpnt: int = img_dimensions[0] // 2

    quadrants: List[QuadTree] = [splitQ(img[:h_midpnt, :w_midpnt], threshold), splitQ(img[:h_midpnt, w_midpnt:], threshold),
                                 splitQ(img[h_midpnt:, :w_midpnt], threshold), splitQ(img[h_midpnt:, w_midpnt:], threshold)]
    return QuadTree(img, quadrants)

def merge(segments: List[np.ndarray], threshold: int) -> List[np.ndarray]:
    merged_segments: List[np.ndarray] = []
    while segments:
        segment: np.ndarray = segments.pop(0)

        for i, other_segment in enumerate(segments):
            if np.abs(segment.mean() - other_segment.mean()) < threshold:
                segment: np.ndarray = np.concatenate((segment, other_segment))
                segments.pop(i)
                break
        merged_segments.append(segment)
    return merged_segments

# # Load image as grayscale
# img: np.ndarray = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE) # numpy n-dimensional array
# 
# # Split image into homogeneous segments
# segments: List[np.ndarray] = split(img, threshold=10)

# # Merge adjacent segments
# merged_segments: np.ndarray = np.concatenate(merge(segments, threshold=10))




img = create_image()
print(img)

print('-' * 100)

segments = splitQ(img=img, threshold=2)
print(segments)

print('-' * 100)

# merge
# print(merge(segments, 10))