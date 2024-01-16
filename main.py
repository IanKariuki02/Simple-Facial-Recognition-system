import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Function to detect and return faces
def get_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if faces is not None:
        for (x, y, w, h) in faces:
            return gray[y:y + w, x:x + h], faces[0]
    return None, None



anna_images = ["picture/Anna/anna.jpg"]
nicholas_images = ["picture/Nicholas/nicholas.jpg"]




def preprocess_images(image_paths, label):
    faces = []
    labels = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            continue
        face, _ = get_face(img)
        if face is not None:
            faces.append(face)
            labels.append(label)
    return faces, labels


faces_anna, labels_anna = preprocess_images(anna_images, 0)  # Label 0 for Anna
faces_nicholas, labels_nicholas = preprocess_images(nicholas_images, 1)  # Label 1 for Nicholas


faces = faces_anna + faces_nicholas
labels = labels_anna + labels_nicholas


face_recognizer.train(faces, np.array(labels))

known_face_names = ["Anna", "Nicholas"]

students = known_face_names.copy()

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '_Attendance.csv', 'w+', newline='')
Inwriter = csv.writer(f)


recognized_students = set()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    face, rect = get_face(frame)
    if face is not None:
        label, confidence = face_recognizer.predict(face)
        if confidence < 120:
            name = known_face_names[label]

            if name not in recognized_students:
                current_time = datetime.now().strftime("%H:%M:%S")
                Inwriter.writerow([name, current_time])
                print("Image detected and captured")

                cv2.putText(frame, "Face detected and captured", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if rect is not None:
            (x, y, w, h) = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
