#%%
from matplotlib import pyplot as plt
import numpy as np
from neuron import Neuron
from generator import generateTrainData

step_low_value = 0
training_rate = 0.01
weight_range = 0.2

activation_function = lambda x: 1 if x > 0 else step_low_value
X, D = generateTrainData(step_low_value, 0.01)
n = Neuron(activation_function, training_rate, weight_range)

# Training
n.train(X, D, allowed_mean_error=0.05)
print('Epochs =', n.epochs)
print('W =', n.w)

# Plot results
plt.scatter(X[:, 0][D == 0], X[:, 1][D == 0])
plt.scatter(X[:, 0][D == 1], X[:, 1][D == 1])

# Plot line
x1 = -1
x2 = 2
y1 = - (n.w[0] + n.w[1] * x1) / n.w[2]
y2 = - (n.w[0] + n.w[1] * x2) / n.w[2]

plt.plot([x1, x2], [y1, y2], marker = 'o')

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Granica etykiet dla wyuczonego modelu Adaline')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.show()

#%%
def evaluate_training(lower_value, learning_rate, weight_range, allowed_mean_error):
    n = Neuron(
            lambda x: 1 if x > 0 else lower_value, 
            learning_rate, 
            weight_range)

    X, D = generateTrainData(lower_value)

    # Learning
    n.train(X, D, allowed_mean_error=allowed_mean_error)
    return n.epochs, n.errors

def chart(x_label, y_label, chart_name, X_raw, test_values):
    avg = np.average(X_raw, 1)
    std_dev = np.std(X_raw, 1)
    plt.errorbar(test_values, avg, std_dev, linestyle='None', marker='o')
    plt.ylim(bottom=0)
    plt.show()

def chart_error(values, errors_list, title, show_count):
    for value, errors in zip(values, errors_list):
        errors = errors[:show_count]
        plt.plot(range(len(errors)), errors, label=value)
    plt.xlabel('Numer epoki')
    plt.ylabel('Błąd średniokwadratowy dla epoki')
    plt.ylim(bottom=0)
    plt.title(title)
    plt.legend()
    plt.show()

#%%
# Learning Rate
values = [0.01, 0.1, 0.2, 0.5]
errors_list = []
for val in values:
    epochs, errors = evaluate_training(-1, val, 0.1, 0.01)
    errors_list.append(errors)
chart_error(values, errors_list, 'Wpływ współczynnika uczenia na uczenie modelu', 50)

#%%
# Weight range
values = [0.01, 0.2, 0.5, 1]
errors_list = []
for val in values:
    ep, err = 0, [np.inf]
    for i in range(1):
        epochs, errors = evaluate_training(-1, 0.001, val, 0.01)
        if (errors[-1] < err[-1]):
            ep, err = epochs, errors
    errors_list.append(err)
chart_error(values, errors_list, 'Wpływ początkowych wag na uczenie modelu', 100)


#%%
# Unipolar vs Bipolar
values = [0, -1]
errors_list = []
for val in values:
    epochs, errors = evaluate_training(val, 0.001, 0.5, 0.01)
    errors_list.append(errors)
chart_error(['Unipolarna', 'Bipolarna'], errors_list, 'Wpływ początkowych wag na uczenie modelu', 200)


#%%
# Weight range
values = [0.1, 0.5, 0.75, 1]
errors_list = []
for val in values:
    epochs, errors = evaluate_training(-1, 0.001, 0.5, val)
    errors_list.append(errors)
chart_error(values, errors_list, 'Wpływ dopuszczalnego błędu średniokwadratowego na uczenie modelu', 100)

