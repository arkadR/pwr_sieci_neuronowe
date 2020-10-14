#%%
import numpy as np
from matplotlib import pyplot as plt
import random
from perceptron import perceptron, perceptron_dynamic_bias
from generator import generateTrainData

p = perceptron(lambda x: 1 if x > 0 else 0, bias=1)
X, D = generateTrainData(0, 0.05)

# training
epoch = 1
any_error = True
while any_error == True: 
    any_error = p.train_epoch(X, D)
    epoch = epoch + 1

print('Epochs =', epoch)
print('W =', p.w)


# Plot line
x1 = -1
x2 = 2
y1 = (p.b - p.w[0] * x1) / p.w[1]
y2 = (p.b - p.w[0] * x2) / p.w[1]

plt.plot([x1, x2], [y1, y2], marker = 'o')

# Plot perceptron results
plt.scatter(X[:, 0][D == 0], X[:, 1][D == 0])
plt.scatter(X[:, 0][D == 1], X[:, 1][D == 1])

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Wyniki wyuczonego modelu')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.show()


#%%
import numpy as np
from matplotlib import pyplot as plt
import random
from perceptron import perceptron, perceptron_dynamic_bias
from generator import generateTrainData

def evaluate_training(bias, lower_value, training_rate, weight_range):
    if bias == None:
        p = perceptron_dynamic_bias(
            lambda x: 1 if x > 0 else lower_value, 
            training_rate, 
            weight_range)
    else:    
        p = perceptron(
            lambda x: 1 if x > 0 else lower_value, 
            training_rate, 
            weight_range,
            bias)

    X, D = generateTrainData(lower_value, 0.05)

    # training
    epoch = 1
    any_error = True
    while any_error == True and epoch < 1000: 
        any_error = p.train_epoch(X, D)
        epoch = epoch + 1
    return epoch

def chart(x_label, y_label, chart_name, X_raw, test_values):
    avg = np.average(X_raw, 1)
    std_dev = np.std(X_raw, 1)
    plt.errorbar(test_values, avg, std_dev, linestyle='None', marker='o')
    plt.ylim(bottom=0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(chart_name)
    plt.show()


#%%
# Bias
results = []
test_values = [4, 3.5, 3, 2.5, 2, 1.5, 1, 0.75, 0.5, 0.4, 0.3, 0.25, 0.15]
for i in test_values:
    result = [evaluate_training(i, 0, 0.1, 0.2) for _ in range(100)]
    results.append(result)

chart('Wartość progu początkowego', 'Liczba epok', 'Wpływ wartości progu na szybkość uczenia', results, test_values)

#%%
# Training Rate
results = []
test_values = [0.001, 0.01, 0.02, 0.05, 0.1, 0.12, 0.15, 0.175, 0.2, 0.22, 0.25, 0.27]
for i in test_values:
    result = [evaluate_training(0.5, -1, i, 0.1) for _ in range(100)]
    results.append(result)

chart('Wartość współczynnika uczenia', 'Liczba epok', 'Wpływ wartości współczynnika uczenia na szybkość uczenia', results, test_values)

#%%
# Weight ranges
results = []
test_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for i in test_values:
    result = [evaluate_training(None, 0, 0.1, i) for _ in range(100)]
    results.append(result)

chart('Zakres początkowych wag', 'Liczba epok', 'Wpływ początkowych wartości wag modelu na szybkość uczenia', results, test_values)

#%%
# Lower activation function value
results = []
test_values = [-1, 0]
for i in test_values:
    result = [evaluate_training(None, i, 0.1, 0.1) for _ in range(100)]
    results.append(result)

chart('x', 'y', 'dafs', results, test_values)

#%%
# bipolar vs unipolar
unipolar_errors = []
for i in range(1000):
    p = perceptron_dynamic_bias(lambda x: 1 if x > 0 else 0, 0.0001, 0.01)
    X, D = generateTrainData(0, 0.05)
    p.train(X, D)
    unipolar_errors.append(p.errors_per_epoch)

bipolar_errors = []
for i in range(1000):
    p = perceptron_dynamic_bias(lambda x: 1 if x > 0 else -1, 0.0001, 0.01)
    X, D = generateTrainData(-1, 0.05)
    p.train(X, D)
    bipolar_errors.append(p.errors_per_epoch)

worst_unipolar_run = max(unipolar_errors, key=len)
worst_bipolar_run = max(bipolar_errors, key=len)
plt.plot(range(len(worst_unipolar_run)), worst_unipolar_run, label='Unipolarna')
plt.plot(range(len(worst_bipolar_run)), worst_bipolar_run, label='Bipolarna')
plt.xlabel('Numer epoki')
plt.ylabel('Liczba błędów')
plt.title('Przebieg uczenia dla perceptronów z różnymi funkcjami aktywacji')
plt.legend()
plt.show()

#%%

results_unipolar = []
test_values = [-1, 0]
for i in test_values:
    result = [evaluate_training(None, 0, 0.1, 0.1) for _ in range(100)]
    results.append(result)