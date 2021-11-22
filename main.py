from headshots import capture_headshots
from train_model import train_for_facial_recog
from facial_req import FaceRecognize
from emotion_detection.emotion1 import *
import cv2


face_recognize = FaceRecognize(False)

# face_recognize.face_recog()



# IMAGE_PATH = "./emotion_detection/images/3.jpg"
# emotionImage(IMAGE_PATH) # If you are using this on an image please provide the path





# train_for_facial_recog()

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

    


cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()