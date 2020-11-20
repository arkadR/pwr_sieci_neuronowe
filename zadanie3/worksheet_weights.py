#%% Imports
from nn import NeuralNetwork
from weight_initialization import WeightInitialization
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
#%% Bulk eval

def eval_weight_init(activation, reps, timecap=30, batch=128):
    best_rdm, best_xav, best_he = None, None, None
    acc_rdm, acc_xav, acc_he = None, None, None
    sizes = (28*28, 80, 10)
    for i in range(reps):
        nn_rdm = NeuralNetwork.create_random(sizes, [activation, ActivationFunction.Sigmoid], weight_init=WeightInitialization.Random) 
        nn_rdm.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=timecap, batch=batch, rate=0.01)
        acc = nn_rdm.evaluate(X_test.T, Y_test)
        if (acc_rdm == None or acc > acc_rdm):
            best_rdm, acc_rdm = nn_rdm, acc
        
        nn_xav = NeuralNetwork.create_random(sizes, [activation, ActivationFunction.Sigmoid], weight_init=WeightInitialization.Xavier) 
        nn_xav.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=timecap, batch=batch, rate=0.01)
        acc = nn_xav.evaluate(X_test.T, Y_test)
        if (acc_xav == None or acc > acc_xav):
            best_xav, acc_xav = nn_xav, acc

        nn_he = NeuralNetwork.create_random(sizes, [activation, ActivationFunction.Sigmoid], weight_init=WeightInitialization.He) 
        nn_he.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=timecap, batch=batch, rate=0.01)
        acc = nn_he.evaluate(X_test.T, Y_test)
        if (acc_he == None or acc > acc_he):
            best_he, acc_he = nn_he, acc

    return best_rdm, best_xav, best_he
        
rdm, xav, he = eval_weight_init(ActivationFunction.Sigmoid, 5, timecap=15)
a = [(s.time, s.train_acc, s.val_acc) for s in rdm.snapshots]
Time, T, V = zip(*a)
plt.plot(Time, V, label="Random")
a = [(s.time, s.train_acc, s.val_acc) for s in xav.snapshots]
Time, T, V = zip(*a)
plt.plot(Time, V, label="Xavier")
a = [(s.time, s.train_acc, s.val_acc) for s in he.snapshots]
Time, T, V = zip(*a)
plt.plot(Time, V, label="He")
plt.title("Porównanie metod inicjalizacji wag (Sigmoid)", wrap=True)
plt.ylim(bottom=0, top=1)
plt.ylabel("Efektywność na zbiorze walidacyjnym")
plt.xlabel("Czas uczenia (s)")
plt.legend()
plt.savefig(f"graph_weight_init_sigmoid.svg")
plt.show()

# %%
