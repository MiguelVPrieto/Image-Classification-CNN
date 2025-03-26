import tensorflow as tf
import numpy as np

def preprocess(image, label):
    image = tf.image.resize(tf.cast(image, tf.float32), (224, 224))
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image, label

def importData(batch_size=64):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds = (train_ds
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(buffer_size=10000)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    test_ds = (test_ds
               .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))

    return train_ds, test_ds