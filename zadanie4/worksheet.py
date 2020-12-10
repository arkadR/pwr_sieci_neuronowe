#%%
# from cnn import ConvolutionalNetwork
from nn import NeuralNetwork
from cnn import ConvolutionalNetwork, Convolution
from weight_initialization import WeightInitialization
from activation_functions import ActivationFunction
import matplotlib
from matplotlib import pyplot as plt
from mlxtend.data import loadlocal_mnist
import numpy as np

#%% Load data
X_main, Y_main = loadlocal_mnist('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
X_test, Y_test = loadlocal_mnist('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')
Y_main = np.reshape(Y_main, newshape=(-1, 1))
X_train = (X_main[:1000]/255)
Y_train = Y_main[:1000]
X_val = (X_main[:-100]/255)
Y_val = Y_main[:-100]

X_train = np.reshape(X_train, (-1, 28, 28))
X_val = np.reshape(X_val, (-1, 28, 28))

#%%
def eval_cnn_withsize(size, runs, epochs):
    a_best = None
    for r in range(runs):
        cnn = ConvolutionalNetwork(
        [
            Convolution((size,size), step=1), 
            # MaxPooling((2,2))
        ], init_x=X_train[0:1,:,:], output_layer=10)

        results = cnn.train(X_train, Y_train, X_val, Y_val, epochs)
        a = [acc for (e, acc) in results]
        if (a_best is None or a[-1] > a_best[-1]):
            a_best = a
    return a_best

def bulk_evaluate(epochs = 5, runs = 5):
    a = eval_cnn_withsize(2, runs, epochs)
    plt.plot(range(len(a)), a, label="2x2")
    a = eval_cnn_withsize(3, runs, epochs)
    plt.plot(range(len(a)), a, label="3x3")
    a = eval_cnn_withsize(4, runs, epochs)
    plt.plot(range(len(a)), a, label="4x4")
    a = eval_cnn_withsize(5, runs, epochs)
    plt.plot(range(len(a)), a, label="5x5")
    plt.ylabel("Efektywność na zbiorze walidacyjnym")
    plt.xlabel("Numer epoki")  
    plt.legend()
    plt.savefig(f"graph_cnn.svg")
    plt.show()

bulk_evaluate(5, 3)