# Doctor-Singh: AI-Based Pneumonia Detection System

**Doctor-Singh** is an AI-powered system designed to assist in detecting pneumonia from chest X-ray images using Convolutional Neural Networks (CNN). Built using TensorFlow and Keras, this project automates the analysis of X-rays to classify them as either **Pneumonia** or **Normal**.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Training the Model](#training-the-model)
6. [Making Predictions](#making-predictions)
7. [File Structure](#file-structure)

## Overview

Doctor-Singh is a medical AI tool designed to detect pneumonia from chest X-ray images. It uses deep learning algorithms (CNNs) to analyze medical images and provide predictions. The goal is to create an AI model that supports doctors and medical professionals by providing quick and reliable diagnoses from X-ray images.

## Prerequisites

Ensure that you have the following installed:
- Python 3.6 or higher
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib (optional for visualization)
- OpenCV (optional for image processing)

To install the required libraries, use:

```bash
pip install tensorflow numpy matplotlib opencv-python
```
## Installation

To get started with the **Doctor-Singh** project, follow these steps:

### 1. Clone the Repository
First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/your-username/Doctor-Singh.git
cd Doctor-Singh
```
### 2. Install the dependencies (if you haven't already):

#### You can create one with your dependencies. Run(powershell):
```powershell
pip freeze > requirements.txt
```
#### Then install:
```powershell
pip install -r requirements.txt
```
## Dataset

This project uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle.

The Chest X-Ray Images (Pneumonia) dataset from Kaggle is organized into three main directories: train, test, and val, each containing subdirectories for the two classes: NORMAL and PNEUMONIA
### Downloading the Dataset:
#### 1. Download the Dataset:

1. Visit the Chest X-Ray Images (Pneumonia) dataset page on Kaggle.
link:https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

 
2. Click on the "Download" button to obtain the dataset.


#### 2. Extract the Dataset:
1. After downloading, extract the contents of the ZIP file to access the images.


#### 3. Organize the Dataset:

```format
chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```
For more details, refer to the Chest X-Ray Images (Pneumonia) dataset page on Kaggle.
## Training the Model

The model can be trained using the provided training script. Make sure your dataset is organized and paths are correctly set. Run the training code:
```bash
python train_model.py
```
The model will be trained and saved as doctor_singh_model.h5. You can adjust the number of epochs and batch size as needed.

## Making Predictions

Once the model is trained, you can use it to predict pneumonia in new chest X-ray images. The following code will help you make predictions:
```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


model = tf.keras.models.load_model('pneumonia_detection_model.h5')  # Load the saved model


def predict_image(img_path):
  img = image.load_img(img_path, target_size=(150, 150))   # Resize image
  img_array = image.img_to_array(img) / 255.0   # Convert to array and normalize
  img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

  prediction = model.predict(img_array)

  if prediction[0] > 0.5:
        print('Pneumonia')
  else:
        print('Normal')

if __name__ == '__main__':
    img_path = 'path_to_image.jpeg'  # Image path
    predict_image(img_path)
```
Make sure to replace 'path_to_image.jpeg' with the path to your image.

## File Structure

Here is a quick overview of the project structure:
```bash
Doctor-Singh/
│
├── chest_xray/                # X-ray images dataset  
│   ├── train/                 # Images used for training the model  
│   ├── val/                   # Images for validation (checking during training)  
│   ├── test/                  # Images for testing the final model  
│
├── model/                     # Folder to store trained models  
│   ├── pneumonia_detection_model.h5  # The saved model file  
│
├── train_model.py             # Script to train the model  
├── predict.py                 # Script to use the model for predictions  
└── requirements.txt           # List of required libraries  
```
