#%% Imports
from neural_network import NeuralNetwork
from activation_functions import ActivationFunction
from matplotlib import pyplot as plt
from mlxtend.data import loadlocal_mnist
import numpy as np


#%% Load data
X_train, Y_train = loadlocal_mnist('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
X_test, Y_test = loadlocal_mnist('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')

#%% Show data
print('Train data shape:', X_train.shape)
print('Test data shape:', X_test.shape)

plt.imshow(np.reshape(X_train[20], (28, 28)))
plt.show()

#%% Feed forward
n = NeuralNetwork.create_random([784, 5, 10], [ActivationFunction.Sigmoid, ActivationFunction.SoftMax])
n.predict(X_train.T / 256), Y_train



#%% Saving and reading from file
n = NeuralNetwork.create_random([3, 3, 3], [ActivationFunction.Sigmoid, ActivationFunction.SoftMax])
n.save_to_file('networks/network1.npz')
n2 = NeuralNetwork.from_file('networks/network1.npz')
print('Original: ', np.asarray(n.W))
print('\nFrom file:', n2.W)

#%% Plot network
def plot_network(n):
    for layer in range(len(n.W)):
        for right_neuron_index in range(len(n.W[layer])):
            for left_neuron_index in range(len(n.W[layer][right_neuron_index])):
                # print(left_neuron_index, right_neuron_index, n.W[layer][right_neuron_index][left_neuron_index])
                plt.plot(
                    [layer, layer+1], 
                    [left_neuron_index, right_neuron_index], 
                    color=str((n.W[layer][right_neuron_index][left_neuron_index] + 1) / 2)
                )

n = NeuralNetwork.create_random([10, 5, 7], [ActivationFunction.Sigmoid, ActivationFunction.SoftMax])
plot_network(n)