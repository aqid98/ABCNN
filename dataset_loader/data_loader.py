import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow.keras.utils as np_utils

def load_mnist_dataset():
    ''' Utility function loads the MNIST Dataset, returns reshaped data
    References:
    - [tf datasets](https://www.tensorflow.org/api_docs/python/tf/keras/datasets)
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255


    X_train = X_train.reshape(60000, 1, 28, 28)
    X_test = X_test.reshape(10000, 1, 28, 28)

    return X_train, y_train, X_test, y_test