import cv2
import imutils
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load face detector
prototxt_path = "face_detector/deploy.prototxt"
weights_path = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxt_path, weights_path)

# Load mask detector model
model = load_model("mask_detector.model")

# Define labels and colors
labels = ["With Mask", "Without Mask"]
colors = [(0, 255, 0), (0, 0, 255)]  # Green for mask, red for no mask

# Start video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]

    # Create blob from image and pass through face detector
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Loop over all detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure bounding boxes are within frame dimensions
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            # Extract the face ROI, preprocess it, and predict
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face = face / 255.0

            (mask, withoutMask) = model.predict(face, verbose=0)[0]
            label_index = 0 if mask > withoutMask else 1
            label = labels[label_index]
            color = colors[label_index]

            # Display the label and bounding box rectangle
            cv2.putText(frame, f"{label}: {max(mask, withoutMask) * 100:.2f}%",
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Show the frame
    cv2.imshow("Face Mask Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to exit
    if key == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
