import cv2
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_TRIPLEX

names = ['priya', 'yokesh', 'zaiba', 'yuvaraj']

def dehaze(input_image):
    # Load the input image
    input_image_path = input_image
    img = cv2.imread(input_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
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
        output_folder = "recognized_faces"
        user_folder = os.path.join(output_folder, recognized_name)
        os.makedirs(user_folder, exist_ok=True)

        output_path = os.path.join(user_folder, f"recognized_{recognized_name}.jpg")
        cv2.imwrite(output_path, img[y:y + h, x:x + w])

    # Display the result
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

input_path=r"Dehazed.jpg"
dehaze(input_path)