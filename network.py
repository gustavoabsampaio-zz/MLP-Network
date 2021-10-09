from sys import base_exec_prefix
import numpy as np
import activation_functions as af

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, inpuit):
        raise NotImplemented
    
    def backward(self, grad_output, learning_rate):
        raise NotImplemented

class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = 2 * np.random.rand(input_size, output_size) - 1.0
        self.bias = 2 * np.random.rand(1, output_size) - 1.0
        
    def forward(self, input):
        self.input = input
        # print("bias: ", self.bias.shape)
        # print("weights: ", self.weights.shape)
        # print("input: ", self.input.shape)
        # print("weights: ", self.weights.shape, input.shape, self.bias.shape)
        # print("OUTPUT: ", np.shape(self.weights), np.shape(input))
        self.output = np.dot(input, self.weights) + self.bias
        return np.squeeze(self.output)
    
    def backward(self, grad_output, learning_rate):
        input_grad = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis = 0).reshape((1,-1))
        
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return input_grad

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        
    # returns the activated input
    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        # print(self.weights.shape, input.shape, self.bias.shape)
        return self.output
    
    # returns grad_input = dE / dX given grad_output = dE / dY
    # learning_rate is not used because there is no "learnable" parameters
    def backward(self, grad_output, learning_rate):
        # print("activation prime input, grad_out: ", (self.activation_prime(self.input) * grad_output).shape)
        act_prime = self.activation_prime(self.input)
        res = act_prime * grad_output
        return res

class SoftMaxLayer:
    def __init__(self):
        self.activation = af.softmax
        self.activation_prime = af.softmax_prime

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)

    def backward(self, output_gradient, leraning_rate):
        n_amostras = output_gradient.shape[0]
        n_classes = output_gradient.shape[1]

        input_gradient = np.zeros((n_amostras, n_classes))

        for k in range(n_amostras):
            Jacobian = self.activation_prime(self.output[k])
            input_gradient[k,:] = np.dot(output_gradient[k,:], Jacobian)
        return input_gradient


class Network:
    def __init__(self):
        self.layers = []
        self.loss = self.mse
        self.loss_prime = self.mse_prime
        
    def add(self, layer):
        self.layers.append(layer)
        
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
        
    def predict(self, input_data):
        samples = len(input_data)
        result = []
        
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
                result.append(output)        
                
        return result
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def mse(self, y_true, y_pred):
        err = np.power(y_true - y_pred, 2)
        return np.mean(err)

    def mse_prime(self, y_true, y_pred):
        grad_output = 2 * (y_pred - y_true) / 1
        return grad_output

    def fit(self, x_train, y_train, epochs, learning_rate, batch_size):
        # batches = self.data_split(x_train, batch_size)
        
        for i in range(epochs):
            err = 0.0
            output_list = []
            batches = [x_train[k:k + batch_size] for k in range(0, len(x_train), batch_size)]
            for batch in batches:
                n_samples = batch.shape[0]
                for j in range(n_samples):
                    sample = batch[j]
                    output = self.forward(sample)
                    output_list.append(output)
                    y_pred = np.argmax(output)
                    err += self.loss(y_train[j], y_pred)
                    for output in output_list:
                        grad_out = self.loss_prime(y_train[j], output)
                    for layer in reversed(self.layers):
                        grad_out = layer.backward(grad_out, learning_rate)
            # err = self.mse(y_train, output_list)
            # err /= n_samples
            # print(err)
            # print(type(err))
            # print(f'epoch {i+1}/{epochs} error = {err:.3f}')
