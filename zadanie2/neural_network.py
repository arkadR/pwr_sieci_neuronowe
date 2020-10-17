import numpy as np
from activation_functions import ActivationFunction

class NeuralNetwork:

    def __init__(self, sizes, W, b, activation_functions):
        self.sizes = sizes
        self.W = W
        self.b = b
        self.activation_functions = activation_functions

    @staticmethod
    def create_random(sizes, activation_functions):
        # W = [np.random.randn(sizes[i+1], sizes[i]) * 0.01 for i in range(len(sizes) - 1)]
        W = [np.random.rand(sizes[i+1], sizes[i]) for i in range(len(sizes) - 1)]
        b = [np.zeros(shape = (sizes[i+1], 1)) for i in range(len(sizes) - 1)]
        return NeuralNetwork(sizes, W, b, activation_functions)

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as file:
            npz = np.load(file, allow_pickle=True)
            return NeuralNetwork(npz['sizes'], npz['W'], npz['b'], npz['activation_functions'])

    def predict(self, X):
        A = self.__feed_forward(X)
        return np.reshape(np.argmax(A, axis=0), (-1,1))

    def __feed_forward(self, X):
        Z = []
        A = [X]
        for w, b, act in zip(self.W, self.b, self.activation_functions):
            Z.append(np.dot(w, A[-1]) + b)
            act = ActivationFunction.get(act)[0]
            A.append(act(Z[-1]))
        return A[-1]

    def save_to_file(self, path):
        np.savez_compressed(path, 
            sizes = self.sizes,
            W = self.W, 
            b = self.b, 
            activation_functions = self.activation_functions)