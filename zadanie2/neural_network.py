import numpy as np
from activation_functions import ActivationFunction

class NeuralNetwork:

    def __init__(self, sizes, W, b, activation_functions):
        self.sizes = sizes
        self.W = W
        self.b = b
        self.activation_functions, self.activation_primes = zip(*[ActivationFunction.get(f) for f in activation_functions])
        self.activation_functions_dump = activation_functions

    @staticmethod
    def create_random(sizes, activation_functions):
        # W = [np.random.randn(sizes[i+1], sizes[i]) * 0.01 for i in range(len(sizes) - 1)]
        W = [np.random.rand(sizes[i+1], sizes[i]) for i in range(len(sizes) - 1)]
        b = [np.random.rand(sizes[i+1], 1) for i in range(len(sizes) - 1)]
        return NeuralNetwork(sizes, W, b, activation_functions)

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as file:
            npz = np.load(file, allow_pickle=True)
            return NeuralNetwork(npz['sizes'], npz['W'], npz['b'], npz['activation_functions'])

    def predict(self, X):
        (_, A) = self.__feed_forward(X)
        return np.reshape(np.argmax(A[-1], axis=0), (-1,1))

    def __feed_forward(self, X):
        Z = []
        A = [X]
        for w, b, act in zip(self.W, self.b, self.activation_functions):
            Z.append(np.dot(w, A[-1]) + b)
            A.append(act(Z[-1]))
        return Z, A

    def gradient_descent(self, X, Y, eta):
        (Z, A) = self.__feed_forward(X)
        (dW, db) = self.__backpropagation(Y, Z, A)
        print(eta * np.array(dW))
        self.W += eta * np.array(dW)
        self.b += eta * np.array(db)
        for (w, b) in zip(self.W, self.b):
            b = b / (np.abs(w).max())
            w = w / (np.abs(w).max())

    def __backpropagation(self, Y, Z, A):
        dA = self.__activation_error(Y, Z, A)
        batch_size = Y.shape[0]
        db = [np.sum(d) / batch_size for d in dA]
        dW = [np.dot(d, a.T) / batch_size for d, a in zip(dA, A)]
        return (dW, db)

    def __activation_error(self, Y, Z, A):
        dA = [None] * len(self.W)

        # Convert the inputs into 0-1 array TODO: Consider doing it outside of training
        y_mat = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=10), 
            arr=np.reshape(Y, newshape=(1, -1)),
            axis=0)

        # Error gradient on output layer
        dA[-1] = (y_mat-A[-1]) * self.activation_primes[-1](Z[-1])
        for i, act in reversed(list(zip(range(len(dA)-1), self.activation_primes))):
            dA[i] = self.W[i+1].T.dot(dA[i+1])*act(Z[i])
        return dA

    def save_to_file(self, path):
        np.savez_compressed(path, 
            sizes = self.sizes,
            W = self.W, 
            b = self.b, 
            activation_functions = self.activation_functions)