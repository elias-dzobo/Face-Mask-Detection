#!/usr/bin/env python
# coding: utf-8

#   BUILDING A MASK DETECTION **MODEL**

# importing all the libraries and modules

# In[5]:


from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import numpy as np
import matplotlib.pyplot as plt

# Build A Neural Network
# - 2 pairs of convoluted and Maxpool layers 
# - A Flatten and Dropout layer to convert data to 1D

# In[ ]:


model = Sequential (
    [
     Conv2D(100, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
     MaxPooling2D(2,2),
     Conv2D(100, (3,3), activation = 'relu'),
     MaxPooling2D(2,2),
     Flatten(),
     Dropout(0.5),
     Dense(50, activation = 'relu'),
     Dense(2, activation= 'softmax')
    ]
)


# Compilig the model

# In[ ]:


model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['acc']
)


# IMAGE DATA GENERATOR / AUGMENTATION
# 
# lets users artificially expand the size of their training dataset by creating modified versions of the images

# In[ ]:


TRAINING_DIR = "./train/train"
train_datagen = ImageDataGenerator(
    rescale = 1.0/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest' 
)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size = 10,
    target_size = (150, 150)

)


VALIDATION_DIR = "./test/test"
validation_datagen = ImageDataGenerator(
    rescale = 1.0/255
)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size = 10,
    target_size = (150, 150)
)


# Call back function to save best model after each epoch training

# In[ ]:


checkpoint = ModelCheckpoint('model2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')


# In[ ]:


history = model.fit_generator(
    train_generator,
    epochs = 10,
    validation_data = validation_generator,
    callbacks = [checkpoint]
)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = list(range(10))

plt.figure(figsize = (16, 10))
plt.subplot(1,2,2)

plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
