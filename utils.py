from typing import Optional, Union, Tuple
import cv2
import bz2
import mediapipe as mp
import numpy as np
import math
from math import pi
import torch
import os, sys, time, re

import numpy as np
from numpy.linalg import norm as l2norm
from skimage import transform as trans

def read_image_from_bz2(file_path):
    try:
        # Read compressed data from the BZ2 file
        with bz2.BZ2File(file_path, 'rb') as file:
            decompressed_data = file.read()
        # Convert the decompressed binary data to a NumPy array
        image_array = np.frombuffer(decompressed_data, dtype=np.uint8)
        # Decode the NumPy array to an OpenCV image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error: {e}")
        return None

def read_image_from_file(file_path):
    try:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"Error: {e}")
        return None

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def crop_face(detection_result, image, margin_percentage=0.5) -> np.ndarray:
    image_copy = np.copy(image.numpy_view())
    detection = detection_result.detections[0]
    bboxC = detection.bounding_box
    x, y, w, h = bboxC.origin_x, bboxC.origin_y, bboxC.width, bboxC.height

    # Calculate margin
    margin_x = int(w * margin_percentage)
    margin_y = int(h * margin_percentage)

    # Adjust cropping coordinates with margin
    x -= margin_x
    y -= margin_y
    w += 2 * margin_x
    h += 2 * margin_y

    # Ensure coordinates are within the image boundaries
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image.width - x)
    h = min(h, image.height - y)

    # Crop the face from the image
    cropped_face = image_copy[y:y+h, x:x+w]
    
    landmark = []
    for keypoint in detection_result.detections[0].keypoints[:4]:
        keypoint_x, keypoint_y = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, image.width, image.height)
        landmark.append([max(keypoint_x - x, 0), max(keypoint_y - y, 0)])
    landmark = np.array(landmark, dtype=np.float32)
    return cropped_face, landmark


# arcface_dst = np.array(
#     [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
#      [41.5493, 92.3655], [70.7299, 92.2041]],
#     dtype=np.float32)

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [56.1396, 92.2848]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (4, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=112):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped