from typing import Optional
import cv2
import bz2
import numpy as np
from math import pi
import torch
import os, sys, time, re

from backbones import get_model
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

# assume face 
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
    for keypoint in detection_result.detections[0].keypoints:
        keypoint_x, keypoint_y = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, image.width, image.height)
        landmark.append([max(keypoint_x - x, 0), max(keypoint_y - y, 0)])
    landmark = np.array(landmark, dtype=np.float32)
    return cropped_face, landmark

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [56.1396, 92.2848], [11.7080, 62.8398], [100.1185, 62.8406]],
    dtype=np.float32)

def calculate_euler_angles(rotation_matrix):
    # Extract pitch, yaw, and roll angles from rotation matrix
    if rotation_matrix[2, 0] != 1 and rotation_matrix[2, 1] != -1:
        pitch_x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw_y = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
        roll_z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        yaw_y = 0 # anything (we default this to zero)
        if rotation_matrix[2, 0] == -1: 
            pitch_x = pi/2 
            roll_z = yaw_y + atan2(rotation_matrix[0, 1], rotation_matrix[0, 2]) 
        else: 
            pitch_x = -pi/2 
            roll_Z = -1*yaw_y + atan2(-1*rotation_matrix[0, 1],-1*rotation_matrix[0, 2]) 
    # convert from radians to degrees
    roll_z = roll_z*180/pi 
    pitch_x = pitch_x*180/pi
    yaw_y = yaw_y*180/pi
    return pitch_x, yaw_y, roll_z

def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (6, 2)
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
    x, y, z = calculate_euler_angles(tform.params[0:3, :])
    if y < -10:
        head_pose="left"
    elif y > 10:
        head_pose="right"
    elif x < -10:
        head_pose="down"
    elif x > 10:
        head_pose="up"
    else:
        head_pose="forward"
    
    M = tform.params[0:2, :]
    return M, head_pose

def norm_crop(img, landmark, image_size=112):
    M, head_pose = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, head_pose