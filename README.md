# JPL Frost Classification Project

## Overview
This repository contains the final project by Yu-Shuo Chen, titled "Frost Classification," developed as part of a machine learning course or research initiative. The project focuses on classifying images of frost patterns using deep learning techniques, specifically Convolutional Neural Networks (CNNs). The dataset used in this project consists of images of frost and non-frost patterns, and the goal is to train a model to accurately distinguish between these two classes.

The project is implemented in Python using Jupyter Notebook and leverages popular libraries such as TensorFlow, Keras, NumPy, and Matplotlib. The notebook includes data preprocessing, model building, training, evaluation, and visualization of results.

## Project Objectives
- **Data Preprocessing**: Load, preprocess, and augment frost and non-frost images to prepare them for model training.
- **Model Development**: Design and implement a CNN model to classify frost images.
- **Training and Evaluation**: Train the model on the prepared dataset and evaluate its performance using metrics such as accuracy, loss, and confusion matrix.
- **Visualization**: Visualize the training process and model predictions to assess performance and interpret results.

## Dataset
The dataset consists of images stored in two directories:
- **Frost Images**: Images depicting frost patterns.
- **Non-Frost Images**: Images without frost patterns.

The images are preprocessed by resizing them to a uniform size (e.g., 128x128 pixels), normalizing pixel values, and applying data augmentation techniques to enhance model robustness.

## Methodology
1. **Data Loading and Preprocessing**:
   - Images are loaded from specified directories using TensorFlow's `ImageDataGenerator`.
   - Data augmentation is applied, including random rotations, flips, and zooms, to increase dataset diversity.
   - Images are normalized to have pixel values between 0 and 1.

2. **Model Architecture**:
   - A Convolutional Neural Network (CNN) is constructed using Keras with the following layers:
     - Convolutional layers with ReLU activation for feature extraction.
     - MaxPooling layers to reduce spatial dimensions.
     - Dropout layers to prevent overfitting.
     - Dense layers for classification, culminating in a softmax output layer for binary classification (frost vs. non-frost).
   - The model is compiled with the Adam optimizer and categorical cross-entropy loss function.

3. **Training**:
   - The model is trained on the augmented dataset using a specified number of epochs and batch size.
   - Training and validation accuracy/loss are monitored to assess model performance.

4. **Evaluation**:
   - The model's performance is evaluated on a test set using accuracy, loss, and a confusion matrix.
   - Precision, recall, and F1-score may also be calculated to provide a comprehensive assessment.

5. **Visualization**:
   - Plots of training and validation accuracy/loss over epochs are generated using Matplotlib.
   - A confusion matrix is visualized to show the model's classification performance.
   - Example predictions on test images are displayed alongside their true labels.

## Requirements
To run this project, you need the following software and libraries:
- **Python**: Version 3.6 or higher
- **Jupyter Notebook**: For running the `.ipynb` file
- **Libraries**:
  - TensorFlow (>=2.0)
  - Keras (included with TensorFlow)
  - NumPy
  - Matplotlib
  - Scikit-learn (for confusion matrix and additional metrics)

You can install the required libraries using pip:
```bash
pip install tensorflow numpy matplotlib scikit-learn
