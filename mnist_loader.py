from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.preprocessing import minmax_scale

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.squeeze(np.array([np.reshape(x, (1, 784)) for x in x_train]))
    x_train = minmax_scale(x_train, (-1,1))

    y_train = np.array(y_train)

    x_test = np.squeeze(np.array([np.reshape(x, (1, 784)) for x in x_test]))
    x_test = minmax_scale(x_test, (-1,1))

    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test

def get_data():
    return NotImplemented

def normalize_data(data, min_value:int = -1):
    ''' 
        min_value should be the same as the activation function's lower limit.
        max_value is normalized to 1.
    '''
    data = data - 1
    max_value = np.max(np.max(data))
    return data/max_value