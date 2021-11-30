import time

from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime
import pytz
import requests
import json

  
# it will get the time zone 
# of the specified location
IST = pytz.timezone('Asia/Kolkata')
# todays_date = today = date.today()
datetime_ist = datetime.now(IST)  


# Rest api end points
ROOT_SENTI_API_URI = "http://20.102.100.20:5000/facial-senti-api"
RAW_SENTI_API_URI = ROOT_SENTI_API_URI + "/raw_sentiments"
headers_ = {'Content-Type': 'application/json'}

 # Load the model
model = Sequential()
classifier = load_model('./emotion_detection/ferjj.h5') # This model has a set of 6 classes

# We have 6 labels for the model
class_labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
classes = list(class_labels.values())
# print(class_labels)


face_classifier = cv2.CascadeClassifier('./emotion_detection/Haarcascades/haarcascade_frontalface_default.xml')




# This function is for designing the overlay text on the predicted image boxes.
def text_on_detected_boxes(text,text_x,text_y,image,font_scale = 1,
                           font = cv2.FONT_HERSHEY_SIMPLEX,
                           FONT_COLOR = (0, 0, 0),
                           FONT_THICKNESS = 2,
                           rectangle_bgr = (0, 255, 0)):



    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
    # Set the Coordinates of the boxes
    box_coords = ((text_x-10, text_y+4), (text_x + text_width+10, text_y - text_height-5))
    # Draw the detected boxes and labels
    cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(image, text, (text_x, text_y), font, fontScale=font_scale, color=FONT_COLOR,thickness=FONT_THICKNESS)




# Detection of the emotions on an image:

def face_detector_image(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) # Convert the image into GrayScale image
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    allfaces = []
    rects = []
    for (x, y, w, h) in faces:
        # x = left, y = top, x + w = right, y+h = bottom
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        allfaces.append(roi_gray)
        rects.append((x, w, y, h))
    return rects, allfaces, img


def emotionImage(emp_details, imgPath):
    img = cv2.imread(imgPath)
    rects, faces, image = face_detector_image(img)

    i = 0
    for face in faces:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]

        # convert numpy array to dictionary
        # predsDict = dict(enumerate(preds.flatten(), 0))
        # print("preds: ", predsDict)

        predsDict = { }
        json_predsDict = { }
        # class_labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
        for i in range(len(preds)):

            if i == 0:
                predsDict["Angry"] = preds[i]
                json_predsDict["Angry"] = str(preds[i])
            if i == 1:
                predsDict["Fear"] = preds[i]
                json_predsDict["Fear"] = str(preds[i])
            if i == 2:
                predsDict["Happy"] = preds[i]
                json_predsDict["Happy"] = str(preds[i])
            if i == 3:
                predsDict["Neutral"] = preds[i]
                json_predsDict["Neutral"] = str(preds[i])
            if i == 4:
                predsDict["Sad"] = preds[i]
                json_predsDict["Sad"] = str(preds[i])
            if i == 5:
                predsDict["Surprised"] = preds[i]
                json_predsDict["Surprised"] = str(preds[i])

        sortedpredsDict = dict(sorted(predsDict.items(), key = lambda x: x[1], reverse=True)) # , reverse=True


        label = "no_emotion"

        listsortedpreds = list(sortedpredsDict)

        # Note: order of elif block is important here
        if listsortedpreds[0] == "Happy":
            label = "likely_happy"
        elif (listsortedpreds[0] == "Neutral") and (listsortedpreds[1] == "Happy" or predsDict["Happy"] > 0.05):
            label = "likely_happy"
        elif listsortedpreds[0] == "Surprised" and listsortedpreds[1] == "Happy":
            label = "likely_happy" 
        elif listsortedpreds[0] == "Neutral" and listsortedpreds[5] == "Happy":
            label = "likely_neutral"
        else:
            label = "likely_not_happy"

        current_datetime = datetime_ist.strftime('%d-%m-%Y %H:%M:%S %Z %z')
        current_date = datetime_ist.strftime('%Y-%m-%d')
        current_time = datetime_ist.strftime('%H:%M:%S')
        emp_name_n_id = emp_details.split("-",1)
        print("json_predsDict: ", json_predsDict)
        try:
            # predsDict_ = json.dumps(json_predsDict, indent=4)
            
            data_ = {
                    "emotion_scores": json_predsDict,
                    "date": current_date,
                    "time": current_time,
                    "emp_name": emp_name_n_id[0],
                    "emp_id": emp_name_n_id[1],
                    "overall_sentiment": label}

            data_ = json.dumps(data_, indent=4)

            print ("data: ", data_)
            r = requests.post(url = RAW_SENTI_API_URI, data = data_, headers = headers_)
            print ("Return of post request", r.json())
        except Exception as e:
            print ("Problem while making post request to url ", RAW_SENTI_API_URI, ", Problem: ", e)


        # label = class_labels[preds.argmax()]
        # label_position = (rects[i][0] + int((rects[i][1] / 2)), abs(rects[i][2] - 10))
        i = + 1

        


        # Overlay our detected emotion on the picture

        # text_on_detected_boxes(label, label_position[0],label_position[1], image)


    cv2.imshow("Emotion Detector", image)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

# ToDo
# def get_emotion_from_image(frame):
    



# Detection of the expression on video stream
def face_detector_video(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        roi_gray = gray[y:y + h, x:x + w]

    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    return (x, w, y, h), roi_gray, img


def emotionVideo(cap):


    while True:

        ret, frame = cap.read()
        rect, face, image = face_detector_video(frame)
        if np.sum([face]) != 0.0:
            roi = face.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (rect[0] + rect[1]//50, rect[2] + rect[3]//50)

            text_on_detected_boxes(label, label_position[0], label_position[1], image) # You can use this function for your another opencv projects.
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(image, str(fps),(5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(image, "No Face Found", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('All', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def test_dict():
    d = {"b": 2, "a": 1,  "c": 3}
    
    sorted(d.items(), key=lambda x: x[1])

    print(d)

if __name__ == '__main__':

    # camera = cv2.VideoCapture(0) # If you are using an USB Camera then Change use 1 instead of 0.
    # emotionVideo(camera)

    # IMAGE_PATH = "./emotion_detection/images/girl_smiling_1.jpg"
    IMAGE_PATH = "./emotion_detection/images/2.jpg"
    emotionImage(IMAGE_PATH) # If you are using this on an image please provide the path


    # test_dict()
