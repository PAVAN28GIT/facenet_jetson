import cv2 as cv
import numpy as np
import pickle
import mediapipe as mp
from keras_facenet import FaceNet
import time
from collections import Counter

import threading
import subprocess

# Load the pre-trained model using pickle
with open('face_recognition_model', 'rb') as f:
    loaded_model, encoder = pickle.load(f)

data = np.load('faces_embeddings_done_4classes.npz')
known_embeddings = data['embeddings']
known_labels = data['labels']

# Load the FaceNet embedder
embedder = FaceNet()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def get_embedding(face_image):
    face_image = face_image.astype('float32')  # 3D (160x160x3)
    face_image = np.expand_dims(face_image, axis=0)
    yhat = embedder.embeddings(face_image)
    return yhat[0]  # 512D Image

def process_frame(frame, threshold=0.7):
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = detector.process(frame_rgb)
    recognized_names = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            x2, y2 = x + w, y + h
            if x < 0 or y < 0 or x2 > iw or y2 > ih:
                continue
            face_region = frame_rgb[y:y2, x:x2]
            if face_region.size == 0:
                continue
            face_region = cv.resize(face_region, (160, 160))
            test_image_embed = get_embedding(face_region).reshape(1, -1)
            distances = [np.linalg.norm(test_image_embed - known_embed) for known_embed in known_embeddings]
            min_distance = min(distances)
            class_label = known_labels[distances.index(min_distance)] if min_distance < threshold else "Unknown"
            recognized_names.append(class_label)
            cv.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, str(class_label), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame, recognized_names

def real_time_face_recognition(duration):
    cap = cv.VideoCapture(0)  # Open webcam
    recognized_names_list = []
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            break
        processed_frame, recognized_names = process_frame(frame)
        recognized_names_list.extend(recognized_names)
        cv.imshow('Real-Time Face Recognition', processed_frame)
        
        # Check if the specified duration has passed
        if time.time() - start_time > duration:
            break
        
        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()  # Release the webcam
    cv.destroyAllWindows()

    if recognized_names_list:
        most_common_name = Counter(recognized_names_list).most_common(1)[0][0]
    else:
        most_common_name = "Unknown"

    return most_common_name

def add_new_face():
    subprocess.call(["python3", "capture_new_subject.py"])

def main():
    # Define the duration for the face recognition
    duration = 20  # Duration in seconds

    # Run the real-time face recognition for the specified duration
    most_frequent_name = real_time_face_recognition(duration)
    if most_frequent_name == "Unknown":
        print("No Face recognized")
        add_new = input("Do you want to add a new face? (yes/no): ").lower()
        if add_new == "yes":
            add_new_face()
    else:
        print(f"Hello {most_frequent_name}, WELCOME!!")

if __name__ == "__main__":
    main()
