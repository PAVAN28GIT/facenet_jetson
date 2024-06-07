import cv2 as cv
import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
import mediapipe as mp
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def extract_face(self, filename):
        img = cv.imread(filename)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Detect faces in the image
        results = self.detector.process(img_rgb)

        if not results.detections:
            raise ValueError("No faces detected in the image.")
        
        # Sort faces by x-coordinate
        detections = sorted(results.detections, key=lambda d: d.location_data.relative_bounding_box.xmin)
        
        # Select the face with the smallest x-coordinate
        detection = detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)

        # Draw bounding box
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return face_arr, img

    def load_faces(self, directory):
        FACES = []
        for im_name in os.listdir(directory):
            if im_name.startswith('.'):  # Skip hidden files
                continue
            try:
                image_path = os.path.join(directory, im_name)
                single_face, _ = self.extract_face(image_path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            if sub_dir.startswith('.'):  # Skip hidden directories/files
                continue
            path = os.path.join(self.directory, sub_dir)
            if not os.path.isdir(path):  # Ensure it's a directory
                continue
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(len(labels))
            self.X.extend(FACES)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)

    def display_image_with_bboxes(self, filename):
        _, img_with_bboxes = self.extract_face(filename)
        img_with_bboxes = cv.cvtColor(img_with_bboxes, cv.COLOR_RGB2BGR)
        plt.imshow(cv.cvtColor(img_with_bboxes, cv.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

time1 = time.time()
faceloading = FACELOADING('./train')
X, Y = faceloading.load_classes()
time2 = time.time()
print(f"{time2 - time1}s")

embedder = FaceNet()

def get_embedding(face_image):
    face_image = face_image.astype('float32')  # 3D (160x160x3)
    face_image = np.expand_dims(face_image, axis=0)
    yhat = embedder.embeddings(face_image)
    return yhat[0]  # 512D Image

EMBEDDED_X = []

for image in X:
    EMBEDDED_X.append(get_embedding(image))

EMBEDDED_X = np.asarray(EMBEDDED_X)

known_embeddings = EMBEDDED_X
known_labels = Y
np.savez_compressed('faces_embeddings_done_4classes.npz', embeddings=known_embeddings, labels=known_labels)

# Label Encoding of Images
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
print(Y)

# Splitting the data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=40)

# Train the SVC model using flattened data

# Train the SVC model using flattened data
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

# Predictions on training and testing data
ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)

# Evaluate model performance
train_accuracy = accuracy_score(Y_train, ypreds_train)
test_accuracy = accuracy_score(Y_test, ypreds_test)

print("Accuracy of the Training Model is:", train_accuracy)
print("Accuracy of the Testing Model is:", test_accuracy)

# Print a classification report
print(classification_report(Y_test, ypreds_test))

# Visualize a confusion matrix
conf_matrix = confusion_matrix(Y_test, ypreds_test)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Saving the pre-trained model using pickle
with open('face_recognition_model', 'wb') as f:
    pickle.dump((model, encoder), f)
