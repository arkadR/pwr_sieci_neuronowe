#%% Imports
from nn import NeuralNetwork
from nn_momentum import NeuralNetworkMomentum
from nn_nestrov import NeuralNetworkNestrov
from nn_adagrad import NeuralNetworkAdagrad
from nn_adadelta import NeuralNetworkAdadelta
from activation_functions import ActivationFunction
import matplotlib
from matplotlib import pyplot as plt
from mlxtend.data import loadlocal_mnist
import numpy as np

#%% Load data
X, Y = loadlocal_mnist('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
X_test, Y_test = loadlocal_mnist('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')
Y = np.reshape(Y, newshape=(-1, 1))
X_train = X[:-5000]
Y_train = Y[:-5000]
X_val = X[-5000:]
Y_val = Y[-5000:]

#%%
nn_base = NeuralNetwork.create_random((28*28, 80, 10), [ActivationFunction.Sigmoid, ActivationFunction.Sigmoid])
sizes = nn_base.sizes
weights = nn_base.W
bias = nn_base.b
activations = nn_base.activation_names

# nn = NeuralNetwork(sizes, weights, bias, activations)
# nn.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=30, batch=128, rate=0.01, silent=False)
# print()
# nn_m = NeuralNetworkMomentum(sizes, weights, bias, activations, 0.9)
# nn_m.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=30, batch=32, rate=0.01, silent=False)
# print()
# nn_n = NeuralNetworkNestrov(sizes, weights, bias, activations, 0.9)
# nn_n.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=30, batch=32, rate=0.01, silent=False)
# print()
# nn_adagrad = NeuralNetworkAdagrad(sizes, weights, bias, activations)
# nn_adagrad.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=30, batch=128, rate=0.01, silent=False)
# print()
nn_adadelta = NeuralNetworkAdadelta(sizes, weights, bias, activations)
nn_adadelta.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=30, batch=128, rate=0.01, silent=False)
print()

#%%