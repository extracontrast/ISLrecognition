# 🇮🇳 Indian Sign Language (ISL) Recognition using BoVW + ORB + ML Classifiers

This project implements **Indian Sign Language (ISL) hand gesture recognition** using  
**ORB feature extraction**, a **Bag of Visual Words (BoVW)** pipeline, and several  
**traditional machine learning classifiers**, both **before and after PCA**.

The system was trained and evaluated on **42,745 images** of Indian hand signs.

---

## 📌 Project Overview

This work explores the full machine-learning pipeline for image-based gesture recognition:

- ORB keypoint & descriptor extraction  
- Visual vocabulary generation using MiniBatchKMeans  
- BoVW histogram construction  
- Dimensionality reduction with PCA  
- Classifier benchmarking (SVM, KNN, LR, NB, MLP)

---

## 🛠️ 1. Environment Setup & Data Acquisition

All experiments were performed using **Google Colab**.

### 🔹 Kaggle Setup
1. Install the Kaggle library  
2. Upload `kaggle.json` and create `~/.kaggle/`  
3. Set file permissions to `600`  
4. Download dataset:  
   **Dataset:** `prathumarikeri/indian-sign-language-isl` (~281MB)

### 🔹 Dataset Extraction
- Unzipped `indian-sign-language-isl.zip`  
- Verified extraction, e.g.:  
  `Indian/V/819.jpg`

---

## 📦 2. Dependencies & Data Preprocessing

### 🔹 Major Libraries Used
- `opencv-python` (cv2)  
- `numpy`, `pandas`, `scipy`, `matplotlib`  
- `tensorflow` (for initial experimentation)  
- `scikit-learn` (SVM, NB, KNN, LR, PCA, metrics, etc.)

### 🔹 Image Preparation
- Collected **42,745 JPEG images**
- Converted images to **grayscale**
- Resized to **200 × 200**
- Extracted labels from folder structure
- Stored as NumPy arrays **X** (images) and **Y** (labels)

---

## 🔍 3. Exploratory Data Analysis (EDA)

✔ Mean images per class  
✔ Variance images per class  
✔ Standard Deviation images per class  

---

## 🧩 4. Feature Engineering

### 🔹 ORB Feature Extraction
- Used: `cv2.ORB_create(nfeatures=2000)`
- Extracted:
  - **Keypoints** + **Descriptors** for all images  
  - Total descriptors collected: **4,399,616**

### 🔹 Bag of Visual Words (BoVW)
- Codebook size: **150** clusters  
- Clustering via **MiniBatchKMeans**  
- Generated BoVW histograms → final feature matrix

### 🔹 Data Split
- **Train:** 25,647  
- **Test:** 17,098  
- Validation: 0  
- Dataset shuffled before training

---

## 📊 5. Model Evaluation

### ⭐ Models Evaluated (Before PCA)
| Classifier | Accuracy | Precision | F1 | Recall |
|-----------|----------|-----------|----|--------|
| **SVM** | 0.9989 | 0.9989 | 0.9989 | 0.9989 |
| **Logistic Regression** | 0.9988 | 0.9989 | 0.9989 | 0.9989 |
| **KNN** | 0.9989 | 0.9989 | 0.9989 | 0.9989 |
| **MLP** | 0.9989 | 0.9989 | 0.9989 | 0.9989 |
| **Naive Bayes** | 0.9916 | 0.9922 | 0.9921 | 0.9922 |

---

## 🔻 PCA (Dimensionality Reduction)
- PCA components: **25**
- Applied on BoVW features

### ⭐ Models Evaluated (After PCA)
| Classifier | Accuracy | Precision | F1 | Recall |
|-----------|----------|-----------|----|--------|
| **KNN** | 0.9987 | 0.9988 | 0.9988 | 0.9987 |
| **MLP** | 0.9977 | 0.9978 | 0.9978 | 0.9978 |
| **SVM** | 0.9959 | 0.9960 | 0.9959 | 0.9961 |
| **Logistic Regression** | 0.9953 | 0.9955 | 0.9955 | 0.9956 |
| **Naive Bayes** | 0.9832 | 0.9845 | 0.9844 | 0.9841 |


---

## 🏁 Final Comparison

- PCA slightly reduced performance but made training faster  
- **KNN remained the top performer (Accuracy: 0.9987 after PCA)**  
- Naive Bayes performance decreased the most with PCA  
- ORB + BoVW proved highly effective for ISL classification

---

## 📜 Summary

This project demonstrates:

- The strength of ORB+BoVW for gesture image classification  
- Extremely high accuracy across multiple ML models  
- PCA can reduce features significantly with minimal accuracy loss  
- KNN and MLP classifiers are consistently top performers  

---
