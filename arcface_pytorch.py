import torch
import numpy as np
from iresnet import iresnet50
from numpy.linalg import norm as l2norm

class ArcfacePytorch:
    def __init__(self, model_path:str, fp16:bool=False, device:str="cpu"):
        self.net = iresnet50(False, fp16=fp16)
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.net.eval()
        self.net.to(device)

    def forward(self, cropped_face: np.ndarray):
        img = np.transpose(cropped_face, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img.div_(255).sub_(0.5).div_(0.5)
        feat = self.net(img).detach().cpu().numpy()
        norm = l2norm(feat)
        feat = feat/norm
        return feat