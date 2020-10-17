#%%Setup
from activation_functions import ActivationFunction
from matplotlib import pyplot as plt
import numpy as np
X = np.arange(-4, 4, 0.01)

#%% Sigmoid
Y = ActivationFunction.get(ActivationFunction.Sigmoid)[0](X)
Y_prime = ActivationFunction.get(ActivationFunction.Sigmoid)[1](X)
plt.title('Sigmoid')
plt.plot(X, Y, label='func')
plt.plot(X, Y_prime, label='prime')
plt.legend()
plt.show()

#%% Tanh
Y = ActivationFunction.get(ActivationFunction.HyperbolicTangent)[0](X)
Y_prime = ActivationFunction.get(ActivationFunction.HyperbolicTangent)[1](X)
plt.title('Tanh')
plt.plot(X, Y, label='func')
plt.plot(X, Y_prime, label='prime')
plt.legend()
plt.show()

#%% SoftMax
Y = ActivationFunction.get(ActivationFunction.SoftMax)[0](X)
Y_prime = ActivationFunction.get(ActivationFunction.SoftMax)[1](X)
plt.title('SoftMax')
plt.plot(X, Y, label='func')
plt.plot(X, Y_prime, label='prime')
plt.legend()
plt.show()

#%% Relu
Y = ActivationFunction.get(ActivationFunction.Relu)[0](X)
Y_prime = ActivationFunction.get(ActivationFunction.Relu)[1](X)
plt.title('Relu')
plt.plot(X, Y, label='func')
plt.plot(X, Y_prime, label='prime')
plt.legend()
plt.show()

#%% Leaky Relu
Y = ActivationFunction.get(ActivationFunction.LRelu)[0](X)
Y_prime = ActivationFunction.get(ActivationFunction.LRelu)[1](X)
plt.title('LRelu')
plt.plot(X, Y, label='func')
plt.plot(X, Y_prime, label='prime')
plt.legend()
plt.show()
