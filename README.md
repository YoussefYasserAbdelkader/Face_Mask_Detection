# 😷 Face Mask Detection using MobileNetV2

A deep learning-based real-time face mask detection system using TensorFlow, Keras, OpenCV, and MobileNetV2. It detects whether a person is **wearing a mask** or **not** using a webcam or video feed.

---

## 📌 Features

- Real-time face mask detection via webcam
- Lightweight and accurate MobileNetV2 backbone
- Data augmentation to improve generalization
- Visual training plots (loss and accuracy)
- Easy to train and test on custom data

---

## 🧾 Directory Structure

Face_Mask_Detection/
├── dataset/ # Face mask dataset
│ ├── with_mask/
│ └── without_mask/
│
├── face_detector/ # Face detection model (Caffe-based)
│ ├── deploy.prototxt
│ └── res10_300x300_ssd_iter_140000.caffemodel
│
├── train_model.py # Model training script
├── detect_mask_video.py # Real-time video inference script
├── mask_detector.model # Saved Keras model (HDF5)
├── training_plot.png # Accuracy/loss plot
├── requirements.txt # Python dependencies
└── README.md



---

## 🔧 Setup Instructions

### ✅ 1. Clone the Repository

git clone https://github.com/your-username/Face_Mask_Detection.git
cd Face_Mask_Detection

✅ 2. Create and Activate Virtual Environment (Windows)

py -3.10 -m venv venv310
venv310\Scripts\activate
✅ 3. Install Dependencies

pip install -r requirements.txt
✅ 4. Prepare the Dataset
Download the dataset from Kaggle:
🔗 https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

Place the two folders inside a directory named dataset/:


dataset/
├── with_mask/
└── without_mask/
🧠 Train the Model

python train_model.py
Saves the model as: mask_detector.model

Generates training visualization: training_plot.png

📥 Download Face Detector Files

mkdir face_detector

Invoke-WebRequest -Uri "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt" -OutFile ".\face_detector\deploy.prototxt"

Invoke-WebRequest -Uri "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel" -OutFile ".\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
🎥 Run Real-Time Detection

python detect_mask_video.py
✅ Output: Webcam will display frames with bounding boxes and labels — Mask (green) or No Mask (red).

📈 Sample Training Plot


🧰 Dependencies

tensorflow
keras
opencv-python
scikit-learn
matplotlib
numpy
