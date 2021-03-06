import cv2
from pathlib import Path
import requests
import json


# Rest api end points
ROOT_SENTI_API_URI = "http://20.102.100.20:5000/facial-senti-api"
NEW_EMP_SENTI_API_URI = ROOT_SENTI_API_URI + "/add_emp"
headers_ = {'Content-Type': 'application/json'}

def capture_headshots():
    # Ask name for the person
    first_name = input ("New face detected. \nPlease enter the first name of the person this face belongs to: ")
    first_name = first_name.strip()
    last_name = input ("Please enter the last name: ")
    last_name = last_name.strip()

    name = first_name + "_" + last_name
    emp_id = input ("Please enter employee id: ")
    images_dir_name = name + "-" + emp_id
    Path("dataset/"+ images_dir_name).mkdir(parents=True, exist_ok=True)
    data_ = {"emp_id": emp_id, "emp_name": name}
    data_ = json.dumps(data_, indent=4)

    r = requests.post(url = NEW_EMP_SENTI_API_URI, data = data_, headers = headers_)
    print ("Return of post request", r)

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("press space to take a photo", 500, 300)

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("press space to take a photo", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "dataset/"+ images_dir_name +"/image_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_headshots()