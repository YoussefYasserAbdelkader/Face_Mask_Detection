import cv2
import numpy as np
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array

# Load face detector and mask classifier
print("[INFO] Loading models...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = load_model("mask_detector.model")

# Initialize webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting real-time detection... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_array = img_to_array(face_resized)
        face_preprocessed = preprocess_input(face_array)
        face_input = np.expand_dims(face_preprocessed, axis=0)

        # Predict mask status
        (mask, withoutMask) = model.predict(face_input)[0]
        label = "With Mask" if mask > withoutMask else "Without Mask"
        color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)

        # Display label and bounding box
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Show frame
    cv2.imshow("Face Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
