import cv2
import os
import image_dehazer  # Assuming you have the image_dehazer module

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

    # Call the dehazer function
    HazeCorrectedImg, _ = image_dehazer.remove_haze(img)

    faces = faceCascade.detectMultiScale(
        HazeCorrectedImg,  # Use the dehazed image for face detection
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(HazeCorrectedImg, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less than 100 ==> "0" is a perfect match
        if confidence < 100:
            recognized_name = names[label]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            recognized_name = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(HazeCorrectedImg, recognized_name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(HazeCorrectedImg, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        # Save the recognized image
        output_folder = "recognized_faces"
        user_folder = os.path.join(output_folder, recognized_name)
        os.makedirs(user_folder, exist_ok=True)

        output_path = os.path.join(user_folder, f"recognized_{recognized_name}.jpg")
        cv2.imwrite(output_path, HazeCorrectedImg[y:y + h, x:x + w])

    # Save the dehazed image in the "dehazed_images" folder
    dehazed_folder = "dehazed_images"
    os.makedirs(dehazed_folder, exist_ok=True)
    dehazed_path = os.path.join(dehazed_folder, f"dehazed_{os.path.basename(input_image)}")
    cv2.imwrite(dehazed_path, HazeCorrectedImg)

    # Display the result
    cv2.imshow('Result', HazeCorrectedImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

input_path = r"Dehazed.jpg"
dehaze(input_path)
