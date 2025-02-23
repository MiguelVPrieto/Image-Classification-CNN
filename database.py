import tensorflow as tf

def importData():
    return tf.keras.datasets.cifar100.load_data(label_mode='fine')