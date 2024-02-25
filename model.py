
import cv2
import os
import time

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_TRIPLEX

names = ['priya', 'yokesh', 'zaiba', 'yuvaraj', 'sharmila', 'rahul', 'harshini']

# Initialize and start real-time video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

output_folder = "recognized_faces"
os.makedirs(output_folder, exist_ok=True)

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less than 100 ==> "0" is a perfect match
        if confidence < 100:
            recognized_name = names[label]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            recognized_name = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, recognized_name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        # Save the recognized image
        user_folder = os.path.join(output_folder, recognized_name)
        os.makedirs(user_folder, exist_ok=True)

        output_path = os.path.join(user_folder, f"recognized_{recognized_name}.jpg")
        cv2.imwrite(output_path, img[y:y + h, x:x + w])

        # Break the loop after recognizing the face
        break

    cv2.imshow('camera', img)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()
