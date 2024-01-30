import random
import os, sys
import numpy as np
from typing import Optional, Union
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy.linalg import norm as l2norm
import torch
from iresnet import iresnet50
from utils import crop_face, norm_crop


class FaceRecogPipeline:
    def __init__(self, face_det_model_path: str,face_recog_model_path: str, device="cpu",
                 min_detection_confidence=0.2, min_tracking_confidence=0.2):

        base_options = python.BaseOptions(model_asset_path=face_det_model_path)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)
        
        net = iresnet50(False, fp16=False)
        net.load_state_dict(torch.load(face_recog_model_path, map_location=device))
        net.eval()
        net.to(device)
        self.face_recog = net
        self.device = device

    
    def face_detect(self, img_mp: mp.Image):
        detection_result = self.detector.detect(img_mp)
        return detection_result
        
    
    def get_face_embedding(self, cropped_face: Union[None, np.ndarray]):
        if cropped_face is None:
            return None
        img = np.transpose(cropped_face, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img.div_(255).sub_(0.5).div_(0.5)
        feat = self.face_recog(img).detach().cpu().numpy()
        norm = l2norm(feat)
        feat = feat/norm
        return feat
    
    def predict(self, img_mp: mp.Image):
        detection_result = self.face_detect(img_mp)
        if len(detection_result.detections)==0:
            print("No faces found")
            return
        cropped_face, landmarks = crop_face(detection_result, img_mp)
        aligned_face = norm_crop(cropped_face, landmarks)
        face_embedding = self.get_face_embedding(aligned_face)
        return face_embedding