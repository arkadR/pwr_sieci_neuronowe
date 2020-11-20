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
    best_sdg, best_adagrad, best_adadelta, best_adam = None, None, None, None
    acc_sdg, acc_adagrad, acc_adadelta, acc_adam = None, None, None, None
    for i in range(reps):
        activations = [activation, ActivationFunction.Sigmoid]
        sizes= (28*28, 80, 10)

        nn = NeuralNetwork.create_random(sizes, activations)
        nn.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=timecap, batch=batch, rate=0.005)
        acc = nn.evaluate(X_test.T, Y_test)
        if (acc_sdg == None or acc > acc_sdg):
            best_sdg, acc_sdg = nn, acc

        nn_gr = NeuralNetworkAdagrad.create_random(sizes, activations)
        nn_gr.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=timecap, batch=batch, rate=0.005)
        acc = nn_gr.evaluate(X_test.T, Y_test)
        if (acc_adagrad == None or acc > acc_adagrad):
            best_adagrad, acc_adagrad = nn_gr, acc
            
        nn_de = NeuralNetworkAdadelta.create_random(sizes, activations)
        nn_de.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=timecap, batch=batch, rate=0.005)
        acc = nn_de.evaluate(X_test.T, Y_test)
        if (acc_adadelta == None or acc > acc_adadelta):
            best_adadelta, acc_adadelta = nn_de, acc

        nn_adam = NeuralNetworkAdam.create_random(sizes, activations)
        nn_adam.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=timecap, batch=batch, rate=0.005)
        acc = nn_adam.evaluate(X_test.T, Y_test)
        if (acc_adam == None or acc > acc_adam):
            best_adam, acc_adam = nn_adam, acc

    return best_sdg, best_adagrad, best_adadelta,  best_adam
        
sdg, adagrad, adadelta, adam = eval_momentum(ActivationFunction.Relu, 5, timecap=90)
Time, T, V = zip(*[(s.time, s.train_acc, s.val_acc) for s in sdg.snapshots])
plt.plot(Time, V, label="SDG")
Time, T, V = zip(*[(s.time, s.train_acc, s.val_acc) for s in adagrad.snapshots])
plt.plot(Time, V, label="Adagrad")
Time, T, V = zip(*[(s.time, s.train_acc, s.val_acc) for s in adadelta.snapshots])
plt.plot(Time, V, label="Adadelta")
Time, T, V = zip(*[(s.time, s.train_acc, s.val_acc) for s in adam.snapshots])
plt.plot(Time, V, label="Adam")
plt.title("Porównanie optymalizatorów współczynnika uczenia (ReLU)", wrap=True)
plt.ylim(bottom=0, top=1)
plt.ylabel("Efektywność na zbiorze walidacyjnym")
plt.xlabel("Czas uczenia (s)")
plt.legend()
plt.savefig(f"graph_optimizers_relu.svg")
plt.show()

# %%
