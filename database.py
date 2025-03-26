import tensorflow as tf
import numpy as np

def resizeImages(images, size=(224, 224), batch_size=200):
    dataset = tf.data.Dataset.from_tensor_slices(images)

    dataset = (dataset
               .map(lambda x: tf.image.resize(tf.cast(x, tf.float16), size),
                    num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))

    resized_batches = []
    for batch in dataset:
        resized_batches.append(batch.numpy())

    return np.concatenate(resized_batches, axis=0)

def importData():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

    x_train = resizeImages(x_train)
    x_test = resizeImages(x_test)

    return (x_train, y_train), (x_test, y_test)
