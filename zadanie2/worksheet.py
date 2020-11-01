#%% Imports
from neural_network import NeuralNetwork
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

#%% Bulk evaluate
def evaluate(
        hidden_count = 80, 
        hidden_activation = ActivationFunction.Relu, 
        output_activation = ActivationFunction.Sigmoid, 
        timecap = 30,
        rate = 0.01,
        batch = 64,
        init_weight_range = 0.01,
        init_bias_range = 0.2):
    
    repetitions = 5
    acc_all = []
    best_net = None
    for i in range(repetitions):
        n = NeuralNetwork.create_random([28*28, hidden_count, 10], [hidden_activation, output_activation], weight_range=init_weight_range, bias_range=init_bias_range)
        n.train(X_train.T, Y_train.T, X_val.T, Y_val.T, timecap=timecap, rate=rate, batch=batch)        
        acc = n.evaluate(X_test.T, Y_test)
        if (acc > max(acc_all, default=0)):
            best_net = n
        acc_all.append(acc)
    return best_net, acc_all

def draw_network_progress(networks, values, param_name, title, labels=None):
    if labels is None: 
        labels = values
    for i, label in enumerate(labels):
        net = networks[i]
        a = [(s.time, s.train_acc, s.val_acc) for s in net.snapshots]
        Time, T, V = zip(*a)
        plt.plot(Time, V, label=f'{param_name}={label}')
    plt.title(title, wrap=True)
    plt.ylim(bottom=0, top=1)
    plt.ylabel("Efektywnośc na zbiorze walidacyjnym")
    plt.xlabel("Czas uczenia (s)")
    plt.legend()
    plt.savefig(f"graph_{param_name}.svg")
    plt.show()


#%% Liczba neuronów w warstwie ukrytej
test_values = [8, 16, 32, 128, 256, 512, 1024]
max_a, std, nets = [], [], []
for hidden_count in test_values:
    net, acc = evaluate(hidden_count=hidden_count)
    max_a.append(np.max(acc))
    std.append(np.std(acc))
    nets.append(net)
plt.errorbar(test_values, max_a, linestyle='None', marker='o')
plt.xlabel('Liczba neuronów w warstwie ukrytej')
plt.ylabel('Efektywność na zbiorze testującym')
plt.title('Najlepsze wyuczone modele w zależności od liczby neuronów w warstwie ukrytej', wrap=True)
plt.savefig("graph1.svg")
plt.show()

draw_network_progress(nets, test_values, 'neuron_count', 'Przebieg uczenia najlepszych sieci w zależności od liczby neuronów w warstwie ukrytej')

#%% Współczynnik uczenia
test_values = [0.001, 0.005, 0.01, 0.015, 0.02, 0.05]
max_a, std, nets = [], [], []
for rate in test_values:
    net, acc = evaluate(rate=rate)
    max_a.append(np.max(acc))
    std.append(np.std(acc))
    nets.append(net)
plt.errorbar(test_values, max_a, linestyle='None', marker='o')
plt.xlabel('Współczynnik uczenia')
plt.ylabel('Efektywność na zbiorze testującym')
plt.title('Najlepsze wyuczone modele w zależności od współczynnika uczenia', wrap=True)
plt.savefig("graph2.svg")
plt.show()

draw_network_progress(nets, test_values, 'rate', 'Przebieg uczenia najlepszych sieci w zależności od współczynnika uczenia')

#%% Początkowe wartości wag
test_values = [0, 0.01, 0.05, 0.1, 0.5, 1]
max_a, std, nets = [], [], []
for weight_range in test_values:
    net, acc = evaluate(init_weight_range=weight_range)
    max_a.append(np.max(acc))
    std.append(np.std(acc))
    nets.append(net)
plt.errorbar(test_values, max_a, linestyle='None', marker='o')
plt.xlabel('Zakres początkowych wartości wag')
plt.ylabel('Efektywność na zbiorze testującym')
plt.title('Najlepsze wyuczone modele w zależności od zakresu początkowych wartości wag', wrap=True)
plt.savefig("graph3.svg")
plt.show()

draw_network_progress(nets, test_values, 'weight_range', 'Przebieg uczenia najlepszych sieci w zależności od zakresu początkowych wartości wag')

#%% Początkowe wartości biasów
test_values = [0, 0.1, 0.2, 0.5, 0.8, 1]
max_a, std, nets = [], [], []
for bias_range in test_values:
    net, acc = evaluate(init_bias_range=bias_range)
    max_a.append(np.max(acc))
    std.append(np.std(acc))
    nets.append(net)
plt.errorbar(test_values, max_a, linestyle='None', marker='o')
plt.xlabel('Zakres początkowych wartości biasów')
plt.ylabel('Efektywność na zbiorze testującym')
plt.title('Najlepsze wyuczone modele w zależności od zakresu początkowych wartości biasów', wrap=True)
plt.savefig("graph4.svg")
plt.show()

draw_network_progress(nets, test_values, 'bias_range', 'Przebieg uczenia najlepszych sieci w zależności od zakresu początkowych wartości biasów')

#%% Funkcje aktywacji
test_values = [
    ActivationFunction.Sigmoid, 
    ActivationFunction.HyperbolicTangent, 
    ActivationFunction.Relu, 
    # ActivationFunction.LRelu,
    ActivationFunction.SoftPlus,  
    # ActivationFunction.Swish,
    ActivationFunction.Linear]
labels=['Sigmoid', 
    'HyperbolicTangent', 
    'Relu', 
    # 'LRelu', 
    'SoftPlus', 
    # 'Swish', 
    'Linear']
max_a, std, nets = [], [], []
for act in test_values:
    net, acc = evaluate(hidden_activation=act)
    max_a.append(np.max(acc))
    std.append(np.std(acc))
    nets.append(net)

test_values = [0, 1, 2, 3, 4]
plt.errorbar(test_values, max_a, linestyle='None', marker='o')
plt.xlabel('Funkcja aktywacji')
plt.xticks(test_values, labels)
plt.ylabel('Efektywność na zbiorze testującym')
plt.title('Najlepsze wyuczone modele w zależności od funkcji aktywacji', wrap=True)
plt.savefig("graph5.svg")
plt.show()

draw_network_progress(nets, test_values, 'activation', 'Przebieg uczenia najlepszych sieci w zależności od funkcji aktywacji', labels=labels)

#%% Wielkości batcha
test_values = [1, 4, 16, 64, 256, 1024]
max_a, std, nets = [], [], []
for batch in test_values:
    net, acc = evaluate(batch=batch)
    max_a.append(np.max(acc))
    std.append(np.std(acc))
    nets.append(net)
plt.errorbar(test_values, max_a, linestyle='None', marker='o')
plt.xlabel('Wielkość batcha')
plt.ylabel('Efektywność na zbiorze testującym')
plt.title('Najlepsze wyuczone modele w zależności od wielkości batcha', wrap=True)
plt.savefig("graph6.svg")
plt.show()

draw_network_progress(nets, test_values, 'batch', 'Przebieg uczenia najlepszych sieci w zależności od wielkości batcha')

#%% Liczba danych treningowych
test_values = [100, 1000, 10000, 20000, 50000]
max_a, std, nets = [], [], []
for count in test_values:
    X_train = X[:count]
    Y_train = Y[:count]
    net, acc = evaluate()
    max_a.append(np.max(acc))
    std.append(np.std(acc))
    nets.append(net)
plt.errorbar(test_values, max_a, linestyle='None', marker='o')
plt.xlabel('Liczba danych treningowych')
plt.ylabel('Efektywność na zbiorze testującym')
plt.title('Najlepsze wyuczone modele w zależności od liczby wzorców treningowych', wrap=True)
plt.savefig("graph7.svg")
plt.show()

draw_network_progress(nets, test_values, 'train_data', 'Przebieg uczenia najlepszych sieci w zależności od liczby wzorców treningowych')

X_train = X[:-10000]
Y_train = Y[:-10000]