import os
import glob
import numpy as np
import cv2
import sys
import onnxruntime

class ArcFaceORT:
    def __init__(self, model_path, device:str="cpu"):
        self.model_path = model_path
        # providers = None will use available provider, for onnxruntime-gpu it will be "CUDAExecutionProvider"
        self.providers = ['CPUExecutionProvider'] if device == "cpu" else None

    #input_size is (w,h), return error message, return None if success
    def check(self, track='cfat', test_img = None):

        if not os.path.exists(self.model_path):
            return "model_path not exists"
        if os.path.isdir(self.model_path):
            return "model_path should be onnx file"

        self.model_file = self.model_path
        print('use onnx-model:', self.model_file)
        try:
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = onnxruntime.InferenceSession(self.model_file, providers=self.providers)
        except:
            return "load onnx failed"
        
        input_cfg = session.get_inputs()[0]
        input_shape = input_cfg.shape
        print('input-shape:', input_shape)
        if len(input_shape)!=4:
            return "length of input_shape should be 4"

        self.image_size = tuple(input_shape[2:4][::-1])
        print('image_size:', self.image_size)
        input_name = input_cfg.name
        outputs = session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        if len(output_names)!=1:
            return "number of output nodes should be 1"
        
        self.session = session
        self.input_name = input_name
        self.output_names = output_names

        input_size = (112,112)
        if input_size!=self.image_size:
            return "input-size is inconsistant with onnx model input, %s vs %s"%(input_size, self.image_size)

        self.input_mean = 127.5
        self.input_std = 127.5

        return None

    def forward(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.image_size

        blob = cv2.dnn.blobFromImages(imgs, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name : blob})[0]
        return net_out