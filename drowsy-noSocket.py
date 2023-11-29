import cv2
import face_recognition
import dlib
from scipy.spatial import distance as dist
import numpy as np
import os
import socket
import time

startTime = time.time()

# Drowsiness detection function using Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Directory containing the images of different people
directory = "images"

reference_encodings = {}

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = face_recognition.load_image_file(os.path.join(directory, filename))
        encoding = face_recognition.face_encodings(image)[0]
        person_name = os.path.splitext(filename)[0]
        reference_encodings[person_name] = encoding

# Initialize the webcam
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

tolerance = 0.6
drowsy_threshold = 0.25

# Flag to keep track of the current drowsiness state
drowsy_flag = False

drowsinessFrameCount = 0

while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    drowsiness_detected = False

    for i, face_encoding in enumerate(face_encodings):
        match_found = False
        for name, reference_encoding in reference_encodings.items():
            matches = face_recognition.compare_faces([reference_encoding], face_encoding, tolerance=tolerance)

            if matches[0]:
                match_found = True
                cv2.putText(frame, f'MATCH: {name}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                face_location = face_locations[i]
                top, right, bottom, left = face_location
                face = frame[top * 4:bottom * 4, left * 4:right * 4]

                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # Detect facial landmarks
                shape = predictor(gray, dlib.rectangle(0, 0, face.shape[0], face.shape[1]))

                left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
                right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)

                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < drowsy_threshold:
                    drowsiness_detected = True

    if drowsiness_detected and not drowsy_flag:        
        drowsy_flag = True

    elif not drowsiness_detected and drowsy_flag:       
        drowsy_flag = False
        print("Drowsiness 0 sent to server")
        drowsinessFrameCount = 0

    if drowsiness_detected:
        cv2.putText(frame, 'DROWSY', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        drowsinessFrameCount += 1
        print(drowsinessFrameCount)
        if drowsinessFrameCount > 30:
            print("Drowsiness 1 sent to server")
            drowsinessFrameCount = 0
    if not drowsiness_detected:
        cv2.putText(frame, 'NOT DROWSY', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()