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


        return self.output
    
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
        return self.activation_prime(self.input) * grad_output

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

    def mse(self, y_true, y_pred):
        # print(np.shape(y_true))
        n = np.shape(y_true)
        if n:
            n = n[0]
        else:
            n = 1
        # print(y_true[0])
        # print(y_pred[0])
        err = sum((y_true - y_pred) ** 2)
        err = (1.0/ n) * err
        return err

    def mse_prime(self, y_true, y_pred):
        # n = np.shape(y_true)
        # if n:
        #     n = n[0]
        # else:
            # n = 1
        # n = y_true.shape[0]
        grad_output = 2 * (y_pred - y_true) / 1
        return grad_output
        
    # def data_split(self, data, size):
    #     split_data = []
    #     n = data.shape[0]
    #     step = n//size
    #     if n % size > 0:
    #         size += 1
    #     for x in range(0, size, step):
    #         if  x != size-1:
    #             split_data.append(data[x:x+step])
    #         else:
    #             split_data.append(data[x:])
    #     print(split_data[0])
    #     return split_data

    def fit(self, x_train, y_train, epochs, learning_rate, batch_size):
        # batches = self.data_split(x_train, batch_size)
        
        for i in range(epochs):
            err = 0.0
            outputs = []
            batches = [x_train[k:k + batch_size] for k in range(0, len(x_train), batch_size)]
            for batch in batches:
                n_samples = batch.shape[0]
                for j in range(n_samples):
                    output = batch[j]
                    for layer in self.layers:
                        # print(output)
                        output = layer.forward(output)
                    outputs.append(output)
                    # err += self.loss(y_train[j], output)
                    for result in output:
                        grad_out = self.loss_prime(y_train[j], output)
                    for layer in reversed(self.layers):
                        grad_out = layer.backward(grad_out, learning_rate)
            # err = self.mse(y_train, outputs)
            # err /= n_samples
            # print(err)
            # print(type(err))
            # print(f'epoch {i+1}/{epochs} error = {err:.3f}')




