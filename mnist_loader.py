from tensorflow.keras.datasets import mnist
import numpy as np

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.array([np.reshape(x, (1, 784)) for x in x_train])
    # y_train = [np.reshape(y, x_train.shape()[0]) for y in y_train]
    y_train = np.array(y_train)
    x_test = np.array([np.reshape(x, (1, 784)) for x in x_test])
    # y_test = [np.reshape(y, x_test.shape()[0]) for y in y_test]
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test

# def load_data():
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     return np.reshape(x_train,(60000,784)), y_train, np.reshape(x_test,(10000,784)), y_test