# Handwritten Roll Number Recognition

## Overview

This project automates the recognition of handwritten roll numbers using machine learning and computer vision techniques. The system accurately identifies and extracts roll numbers from images.

## Code Description

### 1. Training the Digit Recognition Model (Double_Digit_Prediction.py)

This component trains a Convolutional Neural Network (CNN) model using the MNIST dataset. The model architecture is based on the LeNet architecture, consisting of convolutional, pooling, and fully connected layers. After training, the model is saved as a `.pkl` file for later use in predictions.

**Note:** Before running the web application, execute this file to generate the trained model (`.pkl` file).

### 2. Web Application for Real-time Prediction (app.py)

A Flask-based web application that enables real-time prediction of handwritten roll numbers from uploaded images. The application processes images by converting them to grayscale, applying thresholding, and detecting contours. Detected digit regions are cropped, resized, and fed into the trained model for prediction.

**Note:** Ensure the trained model (`.pkl` file) is created by running `Double_Digit_Prediction.py` before starting the web application.

## Features

- **Roll Number Recognition:** Accurately identifies and extracts handwritten roll numbers.
- **Machine Learning Model:** Utilizes a trained CNN for robust and accurate digit recognition.
- **Web Application:** Provides real-time predictions through a Flask-based web interface.
- **User-Friendly Interface:** Simple and intuitive for users to upload images and get predictions.

## Requirements

- Python 3.x
- Flask
- OpenCV
- NumPy
- TensorFlow
- Keras

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```
Run the Double_Digit_Prediction.py script to generate the trained model (.pkl file).

## Usage

Run the Flask application:
```bash
python app.py
```

Access the application in your web browser at http://localhost:5000.
Upload images containing roll numbers to get predictions.

## Credits
This project uses the MNIST dataset for training the digit recognition model. The CNN model is based on the LeNet architecture.
