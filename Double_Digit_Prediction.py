#Necessary Imports
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.optimizers import SGD
import random
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, plot_model
import pickle

#Loading MNIST data for model training
np.random.seed(45)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_classes = 10

#Preparing the images for the model
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
y_train = keras.utils.to_categorical(y_train, num_classes = 10)
y_test = keras.utils.to_categorical(y_test, num_classes = 10)

#Preprocessing data by using augmentation
datagen = ImageDataGenerator(rotation_range=20,
                            zoom_range=0.10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,)
datagen.fit(X_train)

batches = datagen.flow(X_train, y_train, batch_size = 100)
X_batch, y_batch = next(batches)

#Defining the model
def leNet_model():
    # create model
    model = Sequential()
    model.add(keras.Input(shape=(28, 28, 1)))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
      
    # Compile model
    model.compile(SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Declaring the model and fitting it
model = leNet_model()

X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, test_size = 0.1)
history = model.fit(datagen.flow(X_train1, y_train1, batch_size=25),
                            steps_per_epoch=2000,
                            epochs=10,
                            validation_data=(X_val1, y_val1), shuffle = 1)

#Displaying test accuracy
score = model.evaluate(X_test, y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])


# save the model as a pickle file
model_pkl_file = "digit_predict.pkl"  

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model, file)