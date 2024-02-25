import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    labels = []  # Change 'ids' to 'labels'
    label_map = {}  # Mapping from user names to integer labels
    current_id = 0

    for imagePath in imagePaths:
        user_name = os.path.split(imagePath)[-1].split(".")[0]  # Extracting user name without index
        label = label_map.setdefault(user_name, current_id)  # Assign a unique label to each user name

        if label == current_id:
            current_id += 1

        PIL_img = Image.open(imagePath).convert('L')  # Convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            labels.append(label)  # Change 'ids' to 'labels'

    return faceSamples, labels  # Change 'ids' to 'labels'

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, labels = getImagesAndLabels(path)
recognizer.train(faces, np.array(labels))  # Change 'ids' to 'labels'

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

# Print the number of faces trained and end the program
print("\n [INFO] {0} faces trained.".format(len(np.unique(labels))))  # Change 'ids' to 'labels'
