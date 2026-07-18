# Vehicle Speed Estimator & Analytics System

An end-to-end computer vision and data analytics intelligence system built to ingest traffic video streams, stabilize frames, detect moving vehicles, classify them using an optimized Machine Learning framework, and track their speeds to audit traffic enforcement metrics.

## 🌟 Key Engineering Features
* **Digital Image Stabilization:** Minimizes camera vibration artifacts by utilizing Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`) alongside robust affine transformation constraints (`cv2.estimateAffinePartial2D`).
* **Foreground Estimation:** Extracts moving targets efficiently via a Mixture of Gaussians background subtractor (`MOG2`).
* **Geometric Centroid Tracking:** Implements custom tracking mechanics using frame-by-frame Euclidean Distance optimization to persist object profiles across sequential frame matrices.
* **HOG-SVM Inference Framework:** Features an optimized `LinearSVC` model calibrated to classify vehicles as `car`, `biker`, or `truck` using Histogram of Oriented Gradients (HOG) data maps extracted from image patches.
* **Automated Logging Ecosystem:** Tracks metrics in real-time, instantly logging over-speed violations to a metrics dashboard for data visibility.

## 🛠️ Tech Stack & Dependencies
* **Frontend Dashboard:** Streamlit
* **Computer Vision:** OpenCV (`opencv-python-headless`), Scikit-Image (`skimage`)
* **Machine Learning & Core Math:** Scikit-Learn (`sklearn`), NumPy, Joblib
* **Data Pipelines:** Streamlit Caching API, Google Drive Hosting Integrations

## 🚀 Setup & Execution Instructions

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/Abeenashan203/Speed_Predictor.git](https://github.com/Abeenashan203/Speed_Predictor.git)
   cd Speed_Predictor
