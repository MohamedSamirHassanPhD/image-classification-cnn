# Image Classification using CNNs

This repository contains an implementation of an **image classification pipeline using Convolutional Neural Networks (CNNs)**.
The project was developed fully in **Google Colab**, starting from data loading and preprocessing, to training and evaluating the model.

---

## 📌 Project Overview

The goal of this assignment is to build and train a CNN model capable of classifying images into multiple categories.
The notebook includes:

* Dataset loading and preprocessing
* Building a custom CNN architecture
* Training with TensorFlow/Keras
* Accuracy and loss visualization
* Evaluation on test data
* Predictions and visualization of results

This project demonstrates core deep learning concepts used for computer vision tasks.

---

## 📂 Notebook Link

You can open and run the full notebook here:

🔗 **Google Colab:**
[https://colab.research.google.com/drive/1_CHMmKxSoCcglZ2gFfwCeO96MaqRyAVi?usp=sharing](https://colab.research.google.com/drive/1_CHMmKxSoCcglZ2gFfwCeO96MaqRyAVi?usp=sharing)

---

## 🧠 Model Architecture

The CNN used in this project consists of:

* Convolution layers (Conv2D)
* MaxPooling layers
* Flatten layer
* Dense layers
* Softmax output layer

The model is trained using **categorical crossentropy** and optimized using **Adam**.

---

## 🗂 Dataset

The notebook uses a standard computer vision dataset available through Keras or uploaded manually.
The dataset is processed using:

* Normalization
* Train–validation split
* One-hot encoded labels

---

## 🚀 How to Run

1. Open the Colab notebook.
2. Run all cells sequentially.
3. The training will begin and display accuracy/loss curves.
4. Evaluate the model on test images.
5. Visualize predictions at the end.

---

## 📊 Results

The notebook includes:

* Training accuracy
* Validation accuracy
* Loss curves
* Sample predictions
* Confusion matrix *(if enabled)*

Performance varies depending on dataset size and number of epochs.

---

## 🔧 Requirements

* Python 3
* TensorFlow / Keras
* NumPy
* Matplotlib

(Already installed automatically in Google Colab)

---

## 📌 Future Improvements

* Use data augmentation for better generalization
* Add Batch Normalization and Dropout
* Try transfer learning (VGG16, MobileNet, ResNet)
* Export the trained model for deployment

---

## 👨‍💻 Author

**Mohamed Samir Hassan, MSc, PhD Researcher**
Image Processing — Deep Learning — Machine Learning

---

## ⭐ Contributions

Feel free to open issues or submit pull requests to improve this repository.
