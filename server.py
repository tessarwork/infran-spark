import grpc
from concurrent import futures
from proto import test_pb2, test_pb2_grpc

import base64
from datetime import datetime
from PIL import Image
import PIL
import os
import face_detector
from feature_extractor.feature_extractor import fitur 
import numpy as np
import io
import pandas as pd
import csv

class TestServicer(test_pb2_grpc.TestResponseServicer): 
    def Add(self, request, context):
        result = request.num1 + request.num2
        return test_pb2.AddResponse(result=result)
    
    def Decode(self, request, context): 
        Timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        ImgData = request.ImgData
        img = base64.b64decode(ImgData)
        path = '/home/tessar/infran-spark/images'
        if not os.path.exists(path): 
            os.makedirs(path)
        file_path = os.path.join(path, f'image_{Timestamp}.jpg')
        with open(file_path, 'wb') as file: 
            file.write(img)

        return test_pb2.DecodeResponse(file_path = file_path, Timestamp = Timestamp)
    
    def FaceDetect(self, request, context):
        face_detect = False
        path = '/home/tessar/infran-spark/embeddings'

        ImgData = request.ImgData
        TrxID = request.TrxID
        time_stamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        img = face_detector.face_alignment(ImgData)
        feature_extract = fitur(img, face_detector=face_detect)
        embedding_feature = np.load(io.BytesIO(feature_extract), allow_pickle=True)
        embedding_feature = list(pd.DataFrame(embedding_feature).values[0])
        # embedding_feature_bytes = np.array(embedding_feature).tobytes()
        if not os.path.exists(path): 
            os.makedirs(path)
        Embeddings = os.path.join(path, f'embedding_{time_stamp}.csv')
        with open(Embeddings, 'w') as file: 
            writer = csv.writer(file)
            writer.writerow(embedding_feature)

        return test_pb2.FaceDetectResponse(TrxID = TrxID, Embeddings = Embeddings)
        


        
    


    
def serve(): 
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    test_pb2_grpc.add_TestResponseServicer_to_server(TestServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__': 
    serve()
       
