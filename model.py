import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

def dataAugmentation(images):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    datagen.fit(images)

    return datagen

def buildBaselineModel():
    model = tf.keras.Sequential()

    return model

def buildBetterModel():
    model = tf.keras.Sequential()

    return model

def trainModel(model, images, labels, batch_size, epochs):
    images, imagesVal, labels, labelsVal = train_test_split(images, labels, test_size=0.2, random_state=0, stratify=labels)

    datagen = dataAugmentation(images)

    history = model.fit(
        datagen.flow(images, labels, batch_size=batch_size),
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(imagesVal, labelsVal),
        verbose=1
    )
    return history