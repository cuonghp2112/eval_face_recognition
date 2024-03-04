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

def crop_face_landmarks(detection_result, image, margin_percentage=0.3) -> np.ndarray:
    image_copy = np.copy(image.numpy_view())
    landmarks = detection_result.face_landmarks[0] # only take the first face

    x = int(landmarks[127].x*image.width)
    y = int(landmarks[10].y*image.height)
    w = int(landmarks[356].x*image.width) - x
    h = int(landmarks[152].y*image.height) - y

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
    for keypoint in [landmarks[468],landmarks[473],landmarks[1],landmarks[61],landmarks[291]]:
        keypoint_x, keypoint_y = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, image.width, image.height)
        landmark.append([max(keypoint_x - x, 0), max(keypoint_y - y, 0)])
    landmark = np.array(landmark, dtype=np.float32)
    return cropped_face, landmark


arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

# arcface_dst = np.array(
#     [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
#      [56.1396, 92.2848]],
#     dtype=np.float32)

def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
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

def head_pose_estimation(results, image):
    img_w = image.width
    img_h = image.height
    img_c = image.channels
    face_2d = []
    face_3d = []
    for face_landmarks in results.face_landmarks:
        for idx, lm in enumerate(face_landmarks):
            if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                if idx ==1:
                    nose_2d = (lm.x * img_w,lm.y * img_h)
                    nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                x,y = int(lm.x * img_w),int(lm.y * img_h)

                face_2d.append([x,y])
                face_3d.append(([x,y,lm.z]))


        #Get 2d Coord
        face_2d = np.array(face_2d,dtype=np.float64)

        face_3d = np.array(face_3d,dtype=np.float64)

        focal_length = 1 * img_w

        cam_matrix = np.array([[focal_length,0,img_h/2],
                              [0,focal_length,img_w/2],
                              [0,0,1]])
        distortion_matrix = np.zeros((4,1),dtype=np.float64)

        success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)


        #getting rotational of face
        rmat,jac = cv2.Rodrigues(rotation_vec)

        angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        #here based on axis rot angle is calculated
        if y < -10:
            text="left"
        elif y > 10:
            text="right"
        elif x < -20:
            text="down"
        elif x > 20:
            text="up"
        else:
            text="forward"

        return text