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
from utils import crop_face, norm_crop, head_pose_estimation


class FaceRecogPipeline:
    def __init__(self, face_det_model_path: str,face_recog_model_path: str, device="cuda:0",
                 min_detection_confidence=0.1, min_tracking_confidence=0.1, embedding_dir:Union[None, str]=None):

        base_options = python.BaseOptions(model_asset_path=face_det_model_path)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=min_detection_confidence,
                                                         min_tracking_confidence=min_tracking_confidence)
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
    
    def get_face_mesh(self, cropped_face: Union[None, np.array]):
        if cropped_face is None:
            return None
        return self.face_mesh.process(cropped_face)
    
    def get_head_pose(self, landmarks: Union[None, np.array], img_mp: mp.Image):
        head_pose, landmarks_2d = head_pose_estimation(landmarks,img_mp)
        return head_pose, landmarks_2d
        
    
    def get_face_embedding(self, cropped_face: Union[None, np.ndarray]):
        if cropped_face is None:
            return None
        img = np.transpose(cropped_face, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img.div_(255)
        feat = self.face_recog(img).detach().cpu().numpy()
        norm = l2norm(feat)
        feat = feat/norm
        return feat
    
    def predict(self, img_mp: mp.Image):
        detection_result = self.face_detect(img_mp)
        if len(detection_result.detections)==0:
            print("No faces found")
            return
        cropped_face = crop_face(detection_result, img_mp)
        landmarks = self.get_face_mesh(cropped_face)
        if landmarks is None or landmarks.multi_face_landmarks is None:
            print("No face mesh found")
            return None
        head_pose, landmarks_2d = self.get_head_pose(landmarks, cropped_face)
        aligned_face = norm_crop(cropped_face, landmarks_2d)
        face_embedding = self.get_face_embedding(aligned_face)
        return face_embedding, head_pose