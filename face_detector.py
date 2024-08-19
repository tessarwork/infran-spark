from PIL import Image
from infranlib.applications.align.detector import detect_faces
# from infranlib.applications.align.visualization_utils import show_results

import torch
import base64
from infranlib.applications.align.align_trans import get_reference_facial_points, warp_and_crop_face

import numpy as np


import io


crop_size = 112
scale = crop_size / 112.0
input_size = None
reference = get_reference_facial_points(default_square=True) * scale

def face_alignment(image_data):
    global reference
    global crop_size
    global input_size

    img_warp = None
    img_str = None
    
    message = "Face Detection"

    imgData = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(imgData))
    img = img.convert('RGB')
    with torch.no_grad(): 
        _, landmarks = detect_faces(img)
    
    if(len(landmarks) == 0): 
        message = "003"
    else:
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]]for j in range(5)]
        warped_face = warp_and_crop_face(
            np.array(img), 
            facial5points,
            reference, 
            crop_size=(crop_size, crop_size),     
        )
        img_warp = Image.fromarray(warped_face)
        buffered = io.BytesIO()
        img_warp.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

    return img_str


    
    