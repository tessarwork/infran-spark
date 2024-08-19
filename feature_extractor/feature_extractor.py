import sys
import os
import torch
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from face_detector import face_alignment
import base64
import io
import cv2
import logging

from feature_extractor.embeddings import get_embedding
from feature_extractor.backbone import Backbone

model_path = "/Users/taufiq/infran-spark/feature_extractor/model/backbone_ir50_ms1m_epoch120.pth"
input_size = [112, 112]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

backbone = Backbone(input_size)
backbone.load_state_dict(torch.load(model_path, map_location=device))
backbone.to(device)
backbone.eval()

transform  = transforms.Compose(
    [
        transforms.Resize([int(128 * input_size[0] / input_size[1]), int(128 * input_size[0] / input_size[1])]), 
        transforms.CenterCrop([input_size[0], input_size[1]]), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ], 
)

def fitur(image_data, face_detector = True, face_alignment = face_alignment):
     
    Embedding_d = np.zeros([1, 512])
    if face_detector: 
         img_wrp = face_alignment(image_data)
    else:
        imgData = base64.b64decode(image_data)
        img_wrp = Image.open(io.BytesIO(imgData))
    
    if not isinstance(img_wrp, Image.Image):
        raise TypeError(f"Expected a PIL.Image, but got {type(img_wrp)}")
    embedding = get_embedding(backbone, img_wrp, input_size, transform, device)
    Embedding_d[0, :] = embedding
    err_code = 0
    message = "Succes Feature Extraction"
    # except Exception as err: 
    
    #     message = "Error Feature Extraction Model"
    #     err_code = 1

    return Embedding_d.dumps()



    