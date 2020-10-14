import numpy as np

class perceptron:

    def __init__(
            self, 
            activation_function,
            training_rate = 0.1,
            weight_range = 0.2,
            bias = -0.5
            ):
        self.activation_function = activation_function
        self.w = np.random.rand(2) * weight_range
        self.b = bias
        self.training_rate = training_rate
        self.errors_per_epoch = []
        self.epoch = 0

    def eval(self, x):
        z = np.sum(self.w * x) - self.b
        return self.activation_function(z)

    def train(self, x, d):
        y = self.eval(x)
        delta = d - y
        if delta != 0:
            self.w = self.w + self.training_rate * (x*delta)
        return delta != 0

    def train_epoch(self, X, D):
        error = False
        for x, d in zip(X, D):
            error = self.train(np.array(x), d) or error
        return error 


class perceptron_dynamic_bias:

    def __init__(
            self, 
            activation_function,
            training_rate = 0.1,
            weight_range = 0.2,
            ):
        self.activation_function = activation_function
        self.w = np.random.rand(3, 1) * weight_range
        self.training_rate = training_rate
        self.errors_per_epoch = []
        self.epoch = 0

    def eval(self, x):
        z = np.sum(self.w * np.array([1, *x]))
        return self.activation_function(z)

    def train_single(self, x, d):
        y = self.eval(x)
        delta = d - y
        if delta != 0:
            self.w = self.w + self.training_rate * (np.array([1, *x])*delta)
        return delta != 0

    def train(self, X, D):
        error_count = np.inf
        while error_count > 0 and self.epoch < 1000:
            error_count = 0
            for x, d in zip(X, D):
                error_count += self.train_single(x, d)
            self.errors_per_epoch.append(error_count)
        return self.epoch