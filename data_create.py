from face_detector import face_alignment
from feature_extractor.feature_extractor import fitur

import numpy as np
import pandas as pd
import io
import base64

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

import csv
import os

face_detect  = False

output_path = "/"

a = np.zeros([1, 10])
print(a)
# a.dumps()

base64_images = []

def pil_collate_fn(batch):
    # This will be a list of tuples (image, label)
    images, labels = zip(*batch)
    return list(images), labels

transform_image = datasets.ImageFolder("/home/tessar/lfw_2")
dataloader = DataLoader(transform_image, batch_size=1, shuffle=False, collate_fn=pil_collate_fn)

class_names = transform_image.classes
label_1 = []
file_names = []  # List to store file names

# Prepare to store images if needed
base64_images = []  # If you are handling images

for idx, (img, label) in enumerate(dataloader):
    img_pil = img[0]
    label_name = class_names[label[0]]
      # Store image if needed
    label_1.append(label_name)
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue())

    base64_images.append(img_base64)

    # Get the file path
    file_path, _ = transform_image.samples[idx]
    file_name = os.path.basename(file_path)

    file_names.append(file_name)  # Corrected to append file_name

# print(len(file_names))
print(len(class_names))

# Detect faces
face_detect_list = []
for img_string in base64_images: 
    x = face_alignment(img_string)
    face_detect_list.append(x)

# Extract features
matrix_embedding = []
for face_align in face_detect_list:
    y = fitur(face_align, face_detect)
    embedding_feature = np.load(io.BytesIO(y), allow_pickle=True)
    embedding_feature = list(pd.DataFrame(embedding_feature).values[0])
    matrix_embedding.append(embedding_feature)

matrix_embedding = np.array(matrix_embedding)

# Calculate similarity
similarity = cosine_similarity(matrix_embedding)
similarity = similarity.clip(min=0, max=1)

csv_file_path = 'output_with_vertical_similarity_fulllfw.csv'  # Specify your desired CSV file path

with open(csv_file_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write header
    csv_writer.writerow(['File Name', 'Compared To', 'Similarity Score'])
    
    # Write data rows
    for idx in range(len(file_names)):
        for compared_idx in range(len(file_names)):
            # Include self-comparison by removing the condition that excludes it
            row = [
                file_names[idx],  
                file_names[compared_idx], 
                similarity[idx, compared_idx]
            ]
            csv_writer.writerow(row)