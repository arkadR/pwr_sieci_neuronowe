from nn import NeuralNetwork
import numpy as np
import math
from weight_initialization import WeightInitialization
from activation_functions import ActivationFunction
import matplotlib
from matplotlib import pyplot as plt

class ConvolutionalNetwork:

    def __init__(self, layers, init_x, output_layer):
        self.layers = layers
        self.output_shape = self.feed_forward(init_x, show=False)
        output_size = self.output_shape.flatten().shape[0]
        self.nn = NeuralNetwork.create_random([output_size, output_layer], [ActivationFunction.Sigmoid])
    
    def feed_forward(self, X, show=False):
        output = X
        for layer in self.layers:
            output = layer.feed_forward(output)
            # if (show):
            #     plt.imshow(output[0])
            #     plt.show()
        return output

    def predict(self, X):
        conv_output = self.feed_forward(X, show=True)
        x = conv_output.reshape(conv_output.shape[0], -1).T
        (_, A) = self.nn.feed_forward(x)
        return np.reshape(np.argmax(A[-1], axis=0), (-1,1))

    def train(self, X, Y, X_val, Y_val, epochs):
        results = []
        res = self.predict(X_val)
        acc = np.count_nonzero(Y_val == res) / Y_val.shape[0]
        results.append((0, acc))
        for e in range(epochs):
            for i in range(len(X)):
                conv_output = self.feed_forward(X[i:i+1,:,:], show=False)
                x = conv_output.reshape(conv_output.shape[0], -1).T
                A = self.nn.train_and_propagate(x, Y[i:i+1,:])
                err = np.reshape(A, newshape=(-1, self.output_shape.shape[1], self.output_shape.shape[2]))
                # err = np.reshape(A, newshape=(-1, 13, 13)) #TODO
                for layer in reversed(self.layers):
                    err = layer.train_and_propagate(err)
            res = self.predict(X_val)
            acc = np.count_nonzero(Y_val == res) / Y_val.shape[0]
            print(f"epoch: {e}, acc: {acc}")
            results.append((e+1, acc))
        return results

        
class ConvolutionalLayer:
    def feed_forward(self, input):
        pass

    def train_and_propagate(self, error):
        pass

    



class Convolution(ConvolutionalLayer):
    def __init__(self, kernel_size, step, kernel=None):
        if (kernel is None):
            self.kernel_size = kernel_size
            self.kernel = np.random.normal(0, 1, size=kernel_size)
        else:
            self.kernel_size = kernel.shape
            self.kernel = kernel
        
        self.step = step 
        self.prev_input = None

    def __iter_areas(self, image):
        output_size = (
            image.shape[0],
            math.floor((image.shape[1] - (self.kernel_size[0]-1)) / self.step),
            math.floor((image.shape[2] - (self.kernel_size[1]-1)) / self.step)
        )
        for y_idx in range(output_size[2]):
            for x_idx in range(output_size[1]):
                y_top = y_idx * self.step
                x_left = x_idx * self.step
                yield image[
                    :, 
                    y_top:y_top+self.kernel_size[1], 
                    x_left:x_left+self.kernel_size[0]], y_idx, x_idx

    def feed_forward(self, input):
        self.prev_input = input
        output_size = (
            input.shape[0],
            math.floor((input.shape[1] - (self.kernel_size[0]-1)) / self.step),
            math.floor((input.shape[2] - (self.kernel_size[1]-1)) / self.step)
        )
        output = np.zeros(output_size)
        for y_idx in range(output_size[2]):
            for x_idx in range(output_size[1]):
                y_top = y_idx * self.step
                x_left = x_idx * self.step
                slice = input[
                    :, 
                    y_top:y_top+self.kernel_size[1], 
                    x_left:x_left+self.kernel_size[0]]
                res = np.sum(slice * self.kernel, axis=(1,2))
                output[:, y_idx, x_idx] = res
        return output
    
    def train_and_propagate(self, error):
        err = self.kernel * 0.0
        for area, y, x in self.__iter_areas(self.prev_input):
            for dim in range(self.prev_input.shape[0]):
                err += error[dim, y, x] * area[dim]
        self.kernel += 0.001 * err 
        # print(error[0])


class MaxPooling(ConvolutionalLayer):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.prev_input = None
        self.prev_output_size = None

    def __iter_areas(self, image):
        y_slices = image.shape[1] // self.pool_size[0]
        x_slices  = image.shape[2] // self.pool_size[1]
        for i in range(y_slices):
            for j in range(x_slices):
                y_start, y_end = i*self.pool_size[0], (i+1)*self.pool_size[0]
                x_start, x_end = j*self.pool_size[1], (j+1)*self.pool_size[1]
                yield image[:, y_start:y_end, x_start:x_end], i, j
    
    def feed_forward(self, input):
        self.prev_input = input
        output_size = (
            input.shape[0],
            math.floor(input.shape[1] / self.pool_size[0]), 
            math.floor(input.shape[2] / self.pool_size[1]), 
        )
        output = np.zeros(output_size)
        for y_idx in range(0, output_size[1]):
            for x_idx in range(0, output_size[2]):
                y_top = y_idx * self.pool_size[0]
                x_left = x_idx * self.pool_size[1]
                slice = input[
                    :, 
                    y_top:y_top+self.pool_size[0],
                    x_left:x_left+self.pool_size[1]]
                res = np.max(slice, axis=(1,2))
                output[:, y_idx, x_idx] = res
        return output

    def train_and_propagate(self, A):
        err = self.prev_input * 0
        for area, y, x in self.__iter_areas(self.prev_input):
            max_a = np.max(area, axis=(1,2))
            for dim in range(area.shape[0]):
                for y_ in range(area.shape[1]):
                    for x_ in range(area.shape[2]):
                        # if (area[dim, y_, x_] == max_a[dim]):
                        err[dim, y * self.pool_size[0] + y_, x * self.pool_size[1] + x_] = max_a[dim] 
        return err      
