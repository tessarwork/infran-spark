import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from feature_extractor.backbone import Backbone
from tqdm import tqdm
from PIL import Image



def get_embedding(backbone, image, input_size, transform, device):
     
    image_tensor = transform(image)

    with torch.no_grad(): 
        embedding = F.normalize(backbone(image_tensor.unsqueeze(0).to(device))).cpu()

    return embedding

