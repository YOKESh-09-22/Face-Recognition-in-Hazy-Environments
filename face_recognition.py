import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_TRIPLEX

# Initialize and start real-time video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Load the mapping from user names to numeric IDs
label_map = {'priya': 0, 'yokesh': 1, 'zaiba': 2, 'yuvaraj': 3, 'rahul': 4, 'sharmila': 5, 'harshini':6}

# Create a reverse mapping for user IDs to names
reverse_label_map = {v: k for k, v in label_map.items()}

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define min window size to be recognized as a face
    minW = int(0.1 * cam.get(3))
    minH = int(0.1 * cam.get(4))

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(minW, minH),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        label, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if confidence is less than 100 ==> "0" is a perfect match 
        if label in reverse_label_map:
            user_name = reverse_label_map[label]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            user_name = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, user_name, (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)  

    cv2.imshow('camera', img) 

    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()





