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
    
    return cropped_face

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

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
    img_w, img_h, img_c = image.shape
    face_2d = []
    face_3d = []
    arcface_landmarks = {}
    for face_landmarks in results.multi_face_landmarks:
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33:
                arcface_landmarks["l_eye"] = [int(lm.x * img_w),int(lm.y * img_h)]
            elif idx == 263:
                arcface_landmarks["r_eye"] = [int(lm.x * img_w),int(lm.y * img_h)]
            elif idx == 1:
                arcface_landmarks["nose"] = [int(lm.x * img_w),int(lm.y * img_h)]
            elif idx == 61:
                arcface_landmarks["l_m"] = [int(lm.x * img_w),int(lm.y * img_h)]
            elif idx == 291:
                arcface_landmarks["r_m"] = [int(lm.x * img_w),int(lm.y * img_h)]
                
            if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                if idx ==1:
                    nose_2d = (lm.x * img_w,lm.y * img_h)
                    nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                x,y = int(lm.x * img_w),int(lm.y * img_h)

                face_2d.append([x,y])
                face_3d.append(([x,y,lm.z]))


        #Get 2d Coord
        face_2d = np.array(face_2d,dtype=np.float64)
        arcface_landmarks = np.array([arcface_landmarks["l_eye"],arcface_landmarks["r_eye"],
                                     arcface_landmarks["nose"],arcface_landmarks["l_m"],
                                     arcface_landmarks["r_m"]],dtype=np.float64)

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
        elif x < -10:
            text="down"
        elif x > 10:
            text="up"
        else:
            text="forward"

        return text, arcface_landmarks