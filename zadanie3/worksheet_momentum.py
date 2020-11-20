#%% Imports
from nn import NeuralNetwork
from nn_momentum import NeuralNetworkMomentum
from nn_nestrov import NeuralNetworkNestrov
from nn_adagrad import NeuralNetworkAdagrad
from nn_adadelta import NeuralNetworkAdadelta
from nn_adam import NeuralNetworkAdam
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
# nn_base = NeuralNetwork.create_random((28*28, 80, 10), [ActivationFunction.Sigmoid, ActivationFunction.Sigmoid])
# sizes = nn_base.sizes
# weights = nn_base.W
# bias = nn_base.b
# activations = nn_base.activation_names

# # nn = NeuralNetwork(sizes, weights, bias, activations)
# # nn.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=30, batch=128, rate=0.01, silent=False)
# # print()
# # nn_m = NeuralNetworkMomentum(sizes, weights, bias, activations, 0.9)
# # nn_m.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=30, batch=32, rate=0.01, silent=False)
# # print()
# # nn_n = NeuralNetworkNestrov(sizes, weights, bias, activations, 0.9)
# # nn_n.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=30, batch=32, rate=0.01, silent=False)
# # print()
# # nn_adagrad = NeuralNetworkAdagrad(sizes, weights, bias, activations)
# # nn_adagrad.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=30, batch=128, rate=0.01, silent=False)
# # print()
# # nn_adadelta = NeuralNetworkAdadelta(sizes, weights, bias, activations)
# # nn_adadelta.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=30, batch=128, rate=0.01, silent=False)
# # print()
# nn_adam = NeuralNetworkAdam(sizes, weights, bias, activations)
# nn_adam.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=300, batch=128, rate=0.01, silent=False)
# print()
#%% Bulk eval

def eval_momentum(activation, reps, timecap=60, batch=64):
    best_sdg, best_mom, best_nest = None, None, None
    acc_sdg, acc_mom, acc_nest = None, None, None
    for i in range(reps):
        activations = [activation, ActivationFunction.Sigmoid]
        sizes= (28*28, 80, 10)

        nn = NeuralNetwork.create_random(sizes, activations)
        nn.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=timecap, batch=batch, rate=0.01)
        acc = nn.evaluate(X_test.T, Y_test)
        if (acc_sdg == None or acc > acc_sdg):
            best_sdg, acc_sdg = nn, acc

        nn_m = NeuralNetworkMomentum.create_random(sizes, activations)
        nn_m.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=timecap, batch=batch, rate=0.005)
        acc = nn_m.evaluate(X_test.T, Y_test)
        if (acc_mom == None or acc > acc_mom):
            best_mom, acc_mom = nn_m, acc
            
        nn_n = NeuralNetworkNestrov.create_random(sizes, activations)
        nn_n.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=timecap, batch=batch, rate=0.005)
        acc = nn_n.evaluate(X_test.T, Y_test)
        if (acc_nest == None or acc > acc_nest):
            best_nest, acc_nest = nn_n, acc

    return best_sdg, best_mom, best_nest
        
sdg, mom, nest = eval_momentum(ActivationFunction.Relu, 5, timecap=45)
a1 = [(s.time, s.train_acc, s.val_acc) for s in sdg.snapshots]
Time1, T1, V1 = zip(*a1)
plt.plot(Time1, V1, label="SDG")
a2 = [(s.time, s.train_acc, s.val_acc) for s in mom.snapshots]
Time2, T2, V2 = zip(*a2)
plt.plot(Time2, V2, label="Momentum")
a3 = [(s.time, s.train_acc, s.val_acc) for s in nest.snapshots]
Time3, T3, V3 = zip(*a3)
plt.plot(Time3, V3, label="Momentum Nestrova")
plt.title("Porównanie wpływu momentum na uczenie SDG (ReLU)", wrap=True)
plt.ylim(bottom=0, top=1)
plt.ylabel("Efektywność na zbiorze walidacyjnym")
plt.xlabel("Czas uczenia (s)")
plt.legend()
plt.savefig(f"graph_momentum_relu.svg")
plt.show()

# %%
