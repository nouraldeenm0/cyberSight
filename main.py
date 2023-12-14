from typing import List
import numpy as np
import cv2

def split(img: np.ndarray, threshold: int) -> List[np.ndarray]:
    max_val = img.max()
    min_val = img.min()

    # Base case: if image is homogeneous, return it wrapped in a list
    if max_val - min_val < threshold:
        return [img]

    # Recursive case: split image into four quadrants and recurse
    mid_row, mid_col = img.shape[0] // 2, img.shape[1] // 2
    quadrants = [img[:mid_row, :mid_col], img[:mid_row, mid_col:], img[mid_row:, :mid_col], img[mid_row:, mid_col:]]
    return [quad for quadrant in quadrants for quad in split(quadrant, threshold)]

def merge(segments: List[np.ndarray], threshold: int) -> List[np.ndarray]:
    merged_segments = []
    while segments:
        segment = segments.pop(0)
        for i, other_segment in enumerate(segments):
            if np.abs(segment.mean() - other_segment.mean()) < threshold:
                segment = np.concatenate((segment, other_segment))
                segments.pop(i)
                break
        merged_segments.append(segment)
    return merged_segments

# Load image as grayscale
img: np.ndarray = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE) # numpy n-dimensional array


# Split image into homogeneous segments
segments: List[np.ndarray] = split(img, threshold=10)

# Merge adjacent segments
merged_segments: np.ndarray = np.concatenate(merge(segments, threshold=10))
