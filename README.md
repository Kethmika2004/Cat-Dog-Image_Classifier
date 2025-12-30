# 🐱🐶 Cats vs Dogs Classifier – Deep Learning Project

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange)](https://www.tensorflow.org/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🚀 Project Overview
Welcome to my **Cats vs Dogs classifier**! 🐾  

This project demonstrates **image classification using a Convolutional Neural Network (CNN)**. The model can accurately distinguish cats 🐱 from dogs 🐶, even with a small dataset, by using **data augmentation**, **Batch Normalization**, and **Dropout layers**.  

**Goal:** Achieve robust predictions on unseen images with >70% accuracy.  

---

## 🖼 Dataset
The dataset is structured as follows:
```
cats_and_dogs/
├── train/               # Training images
│   ├── cats/
│   └── dogs/
├── validation/          # Validation images
│   ├── cats/
│   └── dogs/
└── test/                # Test images for predictions
```


Source: [FreeCodeCamp Cats and Dogs Dataset](https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip)

---

## 🔧 Features
- **CNN Architecture:** 4 convolutional blocks with BatchNorm & Dropout.  
- **Data Augmentation:** Random rotations, shifts, zooms, shears, flips.  
- **Callbacks:** EarlyStopping & ReduceLROnPlateau for better training.  
- **Visualization:**  
  - Training & validation accuracy/loss  
  - Test images with **confidence percentages**  

---

## 🎨 Example Predictions

### Training & Validation Accuracy/Loss

![accuracy_plot](images/accuracy_plot.png)  
*Model learning curves showing convergence and stability.*

---

## ⚡ How It Works

1. **Data Preparation:** Rescale and augment training images.  
2. **Model Building:** Sequential CNN with 4 Conv2D blocks, BatchNorm, MaxPooling, Flatten, Dense, and Dropout.  
3. **Training:**  
   ```python
   model.fit(train_data_gen,
             validation_data=val_data_gen,
             epochs=50,
             callbacks=[EarlyStopping, ReduceLROnPlateau])
4. **Evaluation & Visualization:** Plot training curves, predict on test images, and visualize results.

---

## 💡 Results

  - **Typical accuracy on test set:** >70%
  - **Prediction confidence:** Shown as percentages for each image
  - The model generalizes well due to augmentation and robust architecture.

---

## 🚀 Future Improvements

  - Integrate Transfer Learning (EfficientNet, MobileNet) for higher accuracy (~85–90%).
  - Expand dataset with more diverse images.
  - Hyperparameter tuning: batch size, learning rate, optimizer.
  - Create a web app interface for real-time predictions.

---

## 🛠 Tech Stack

  - Python 3.10+
  - TensorFlow 2.x / Keras
  - Matplotlib / NumPy
