import numpy as np
import math
import sys
from activation_functions import ActivationFunction

class NeuralNetwork:

    def __init__(self, sizes, W, b, activations):
        self.sizes = sizes
        self.W = W
        self.b = b
        self.activation_functions = [ActivationFunction.get(a)[0] for a in activations]
        self.activation_primes = [ActivationFunction.get(a)[1] for a in activations]
        self.activation_names = activations


    @staticmethod
    def from_file(path):
        with open(path, 'rb') as file:
            npz = np.load(file)
            return NeuralNetwork(npz['sizes'], npz['W'], npz['b'], npz['activations'])

    @staticmethod
    def create_random(sizes, activations, weight_range = 0.01):
        W = [np.random.randn(sizes[i+1], sizes[i]) * weight_range for i in range(len(sizes) - 1)]
        b = [np.zeros(shape = (sizes[i+1], 1)) for i in range(len(sizes) - 1)]
        return NeuralNetwork(sizes, W, b, activations)

    def predict(self, X):
        (_, A) = self.__feed_forward(X)
        return np.reshape(np.argmax(A[-1], axis=0), (-1,1))

    def __feed_forward(self, X):
        Z = []
        A = [X]
        for w, b, act in list(zip(self.W, self.b, self.activation_functions)):
            Z.append(np.dot(w, A[-1]) + b)
            A.append(act(Z[-1]))
        return (np.asarray(Z), A)

    def __activation_error(self, Y, Z, A):
        dA = [None] * len(self.W)
        y_mat = np.transpose([np.bincount([y], minlength=np.shape(A[-1])[0]) for y in Y[0]])
        dA[-1] = -(y_mat-A[-1]) * self.activation_primes[-1](Z[-1])

        for i, act in reversed(list(zip(range(len(dA)-1), self.activation_primes))):
            dA[i] = self.W[i+1].T.dot(dA[i+1])*act(Z[i])
        return dA

    def __backpropagation(self, y, Z, A):
        dA = self.__activation_error(y, Z, A)
        count = y.shape[1]
        db = [np.sum(d) / count for d in dA]
        dW = [np.dot(d, a.T) / count for d, a in zip(dA, A)]
        return (dW, db)

    def __update_parameters(self, dW, db, eta):
        self.W -= eta*np.array(dW)
        self.b -= eta*np.array(db)

    def __gradient_descent(self, X, Y, eta):
        (Z, A) = self.__feed_forward(X)
        (dW, db) = self.__backpropagation(Y, Z, A)
        self.__update_parameters(dW, db, eta)

    def train(
        self, 
        X_train, Y_train, 
        X_val, Y_val, 
        epochs=10, rate=0.1, batch=100
        ):
        batch = min(batch, X_train.shape[1]) 
        batch_count = X_train.shape[1] // batch
        for e in range(epochs):
            acc = 0
            for _ in range(batch_count):
                # Split to batches
                sub = np.random.randint(X_train.shape[1], size=batch)
                X_s, Y_s = X_train[:, sub], Y_train[:, sub]
                # Stochastic Gradient Descent
                self.__gradient_descent(X_s, Y_s, rate)
                # Add train results
                (_, A) = self.__feed_forward(X_s)
                acc += np.count_nonzero(np.argmax(A[-1], axis=0) == Y_s) / (np.shape(Y_s)[1])
            
            acc /= batch_count
            sys.stdout.write(f"\rEpoch: {e+1}, TrainAcc: {acc:9.3f}     ")
            (_, A) = self.__feed_forward(X_val)
            acc = np.count_nonzero(np.argmax(A[-1], axis=0) == Y_val) / np.shape(Y_val)[1]
            sys.stdout.write(f"ValAcc: {acc:9.3f}\n")

    def save_to_file(self, path):
        self.W = [w.astype('float16') for w in self.W]
        self.b = [b.astype('float16') for b in self.b]
        np.savez_compressed(path, 
            sizes = self.sizes,
            W = self.W, b = self.b, activations=self.activation_names)
