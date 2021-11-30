from headshots import capture_headshots
from train_model import train_for_facial_recog
import facial_req
from facial_req import FaceRecognize
from emotion_detection.emotion1 import *
import cv2


face_recognize = FaceRecognize(False)

# face_recognize.face_recog()

# IMAGE_PATH = "./emotion_detection/images/3.jpg"
# emotionImage(IMAGE_PATH) # If you are using this on an image please provide the path


# capture_headshots()


train_for_facial_recog()

# print("is new face: ", face_recognize.isNewFace)
# while True:
#     if face_recognize.isNewFace:
#         capture_headshots()
#         train_for_facial_recog()
#         face_recognize.isNewFace = False
#         print("after middel is new face: ", face_recognize.isNewFace)


#     else:
#         face_recognize.face_recog()
#         print("after is new face: ", face_recognize.isNewFace)