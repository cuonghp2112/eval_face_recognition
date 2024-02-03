import random
import os, sys
import numpy as np
from typing import Optional, Union
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import torch
from arcface_pytorch import ArcfacePytorch
from arcface_ort import ArcFaceORT
from utils import crop_face, norm_crop


class FaceRecogPipeline:
    def __init__(self, face_det_model_path: str,face_recog_model_path: str, device="cpu"):

        base_options = python.BaseOptions(model_asset_path=face_det_model_path)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)
        
        if face_recog_model_path.endswith(".pth"):
            face_recog = ArcfacePytorch(face_recog_model_path, device=device)
        elif face_recog_model_path.endswith(".onnx"):
            face_recog = ArcFaceORT(face_recog_model_path, device=device)
            face_recog.check()
        else:
            print("unsupported model")
            face_recog = None

        self.face_recog = face_recog

    
    def face_detect(self, img_mp: mp.Image):
        detection_result = self.detector.detect(img_mp)
        return detection_result
        
    
    def get_face_embedding(self, cropped_face: Union[None, np.ndarray]):
        if cropped_face is None:
            return None
        return self.face_recog.forward(cropped_face)
    
    def predict(self, img_mp: mp.Image):
        detection_result = self.face_detect(img_mp)
        if len(detection_result.detections)==0:
            print("No faces found")
            return
        cropped_face, landmarks = crop_face(detection_result, img_mp)
        aligned_face = norm_crop(cropped_face, landmarks)
        face_embedding = self.get_face_embedding(aligned_face)
        return face_embedding