import numpy as np

class Neuron:

    def __init__(
            self, 
            activation_function,
            training_rate = 0.1,
            weight_range = 0.2,
            ):
        
        self.activation_function = activation_function
        self.w = (np.random.rand(3) * 2 - 1) * weight_range
        self.training_rate = training_rate
        self.epochs = 0
        self.errors = []

    def evaluate(self, x):
        return self.activation_function(self._output(x))

    def _output(self, x):
        return self.w @ np.concatenate(([1], x)).T

    def train(self, X, D, allowed_mean_error):
        mean_error = np.inf
        while mean_error > allowed_mean_error and self.epochs < 1000:
            mean_error = 0.0
            X, D = self._shuffle(X, D)
            for x, d in zip(X, D):
                y = self._output(x)
                err = d - y
                self.w += self.training_rate * err * np.concatenate(([1], x))
                # self.w = np.clip(self.w, -1, 1)
                mean_error += err**2
            mean_error /= len(X)
            self.errors.append(mean_error)
            self.epochs += 1
    
    def _shuffle(self, X1, X2):
        p = np.random.permutation(len(X1))
        return np.array(X1)[p], np.array(X2)[p]