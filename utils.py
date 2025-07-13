import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

def load_face_detector(prototxt_path="face_detector/deploy.prototxt",
                       weights_path="face_detector/res10_300x300_ssd_iter_140000.caffemodel"):
    """Load OpenCV DNN face detector from disk."""
    net = cv2.dnn.readNet(prototxt_path, weights_path)
    return net

def detect_faces_dnn(frame, net, conf_threshold=0.5):
    """Detect faces in a frame using OpenCV DNN model."""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))  # Mean subtraction for normalization
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            boxes.append((startX, startY, endX, endY))
    return boxes

def preprocess_face(frame, box):
    """Crop and preprocess a single face for mask classification."""
    (startX, startY, endX, endY) = box
    face = frame[startY:endY, startX:endX]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    face = face / 255.0  # Normalize
    return face
