import numpy as np
import math
import sys
import time
from dataclasses import dataclass
from activation_functions import ActivationFunction
from weight_initialization import WeightInitialization

class NeuralNetwork:

    def __init__(self, sizes, W, b, activations):
        self.sizes = sizes
        self.W = W
        self.b = b
        self.activation_functions = [ActivationFunction.get(a)[0] for a in activations]
        self.activation_primes = [ActivationFunction.get(a)[1] for a in activations]
        self.activation_names = activations
        self.snapshots = []
        self.best_W = self.W
        self.best_b = self.b
        self.best_val_acc = 0

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as file:
            npz = np.load(file, allow_pickle=True)
            return NeuralNetwork(npz['sizes'], npz['W'], npz['b'], npz['activations'])

    @staticmethod
    def create_random(sizes, activations, weight_range = 0.01, bias_range = 0.01, 
                    weight_init=WeightInitialization.Random):
        W = weight_init(sizes, weight_range)
        b = [np.random.randn(sizes[i+1], 1) * bias_range for i in range(len(sizes) - 1)]
        return NeuralNetwork(sizes, W, b, activations)

    def predict(self, X):
        (_, A) = self.feed_forward(X)
        return np.reshape(np.argmax(A[-1], axis=0), (-1,1))

    def feed_forward(self, X):
        Z = []
        A = [X]
        for w, b, act in list(zip(self.W, self.b, self.activation_functions)):
            Z.append(np.dot(w, A[-1]) + b)
            A.append(act(Z[-1]))
        return (np.asarray(Z), A)

    def __activation_error(self, Y, Z, A):
        dA = [None] * (len(self.W) + 1)
        y_mat = np.transpose([np.bincount([y], minlength=np.shape(A[-1])[0]) for y in Y[0]])
        dA[-1] = -(y_mat-A[-1]) * self.activation_primes[-1](Z[-1])

        for i, act in reversed(list(zip(range(len(dA)-2), self.activation_primes))):
            dA[i+1] = self.W[i+1].T.dot(dA[i+2])*act(Z[i])
        dA[0] = self.W[0].T.dot(dA[1])
        return dA

    def __loss(self, Y, Y_exp):
        return np.square(Y-Y_exp)/len(Y)

    def __backpropagation(self, y, Z, A):
        dA = self.__activation_error(y, Z, A)
        count = y.shape[1]
        db = [np.sum(d) / count for d in dA[1:]]
        dW = [np.dot(d, a.T) / count for d, a in zip(dA[1:], A)]
        return (dW, db, dA)

    def __update_parameters(self, dW, db, eta):
        for i, _ in enumerate(dW):
            self.W[i] -= eta*np.array(dW[i])
            self.b[i] -= eta*np.array(db[i])

    def __gradient_descent(self, X, Y, eta):
        (Z, A) = self.feed_forward(X)
        (dW, db, dA) = self.__backpropagation(Y, Z, A)
        self.__update_parameters(dW, db, eta)

    def train_and_propagate(self, X, Y, eta=0.05):
        (Z, A) = self.feed_forward(X)
        (dW, db, dA) = self.__backpropagation(Y, Z, A)
        self.__update_parameters(dW, db, eta)
        return dA[0]

    def train(
            self, 
            X_train, Y_train, 
            X_val, Y_val, 
            epochs=None, timecap=None, rate=0.1, batch=100, silent=True
            ):
        assert (epochs is not None) ^ (timecap is not None), "Provide only timecap or epochs"
        batch = min(batch, X_train.shape[1]) 
        batch_count = X_train.shape[1] // batch
        epoch = 0
        
        train_acc = self.evaluate(X_train, Y_train)
        val_acc = self.evaluate(X_val, Y_val)
        if silent == False:
            sys.stdout.write(f"\rEpoch: {epoch+1}, TrainAcc: {train_acc:9.3f}, ValAcc: {val_acc:9.3f}\n")
        self.snapshots.append(Snapshot(0, train_acc, val_acc))
        start = time.time()
        while (epochs is not None and epoch < epochs) or (timecap is not None and time.time() - start < timecap):
            epoch += 1
            for _ in range(batch_count):
                # Split to batches
                sub = np.random.randint(X_train.shape[1], size=batch)
                X_s, Y_s = X_train[:, sub], Y_train[:, sub]
                # Stochastic Gradient Descent
                self.__gradient_descent(X_s, Y_s, rate)
                # Break early if time passed
                if timecap is not None and time.time() - start > timecap:
                    break
                
            train_acc = self.evaluate(X_train, Y_train)
            val_acc = self.evaluate(X_val, Y_val)
            if silent == False:
                sys.stdout.write(f"\rEpoch: {epoch+1}, TrainAcc: {train_acc:9.3f}, ValAcc: {val_acc:9.3f}\n")
            self.snapshots.append(Snapshot(time.time() - start, train_acc, val_acc))
            if (val_acc > self.best_val_acc):
                self.best_val_acc = val_acc
                self.best_W, self.best_b = self.W, self.b
        self.W, self.b = self.best_W, self.best_b

    def save_to_file(self, path):
        self.W = [w.astype('float16') for w in self.W]
        self.b = [b.astype('float16') for b in self.b]
        np.savez_compressed(path, 
            sizes = self.sizes,
            W = self.W, b = self.b, activations=self.activation_names)

    def evaluate(self, X, Y):
        y = np.reshape(Y, (-1))
        pred = np.reshape(self.predict(X), (-1))
        return np.count_nonzero(pred == y) / len(y)


@dataclass
class Snapshot:
    time: float
    train_acc: float
    val_acc: float
    # loss: float