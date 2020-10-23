#%% Imports
from neural_network import NeuralNetwork
from activation_functions import ActivationFunction
from matplotlib import pyplot as plt
from mlxtend.data import loadlocal_mnist
import numpy as np


#%% Load data
X, Y = loadlocal_mnist('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
X_test, Y_test = loadlocal_mnist('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')
Y = np.reshape(Y, newshape=(-1, 1))

#%% Training
X_train = X[:10000]
Y_train = Y[:10000]
X_val = X[10000:12000]
Y_val = Y[10000:12000]

n = NeuralNetwork.create_random([28*28, 20, 10], [ActivationFunction.Relu, ActivationFunction.Relu, ActivationFunction.Sigmoid])
n.train(X_train.T, Y_train.T, X_val.T, Y_val.T, epochs=10, rate=0.01, batch=64)

#%% Results
Z = n.predict(X_test.T).T[0]
print (Z, Y_test)



#%% Show data
print('Train data shape:', X.shape)
print('Test data shape:', X_test.shape)

plt.imshow(np.reshape(X[20], (28, 28)))
plt.show()

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


# %%
funs = [ActivationFunction.Sigmoid, ActivationFunction.SoftMax, ActivationFunction.SoftMax]
c, v = zip(*[ActivationFunction.get(f) for f in funs])
print(c)
print(v)
