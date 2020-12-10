import numpy as np  

class ActivationFunction:
    Sigmoid = 0
    HyperbolicTangent = 1
    Relu = 2
    LRelu = 3
    SoftPlus = 4
    Swish = 5
    Linear = 6
    SoftMax = 7

    @staticmethod
    def get(function):
        if (function == ActivationFunction.Sigmoid):
            return (
                lambda x: ActivationFunction.__sigmoid(x, 1), 
                lambda x: ActivationFunction.__sigmoid_prime(x, 1))

        elif (function == ActivationFunction.HyperbolicTangent):
            return (
                lambda x: ActivationFunction.__hyperbolic_tangent(x, 1), 
                lambda x: ActivationFunction.__hyperbolic_tangent_prime(x, 1))

        elif (function == ActivationFunction.SoftMax):
            return (
                lambda x: ActivationFunction.__softmax(x), 
                lambda x: ActivationFunction.__softmax_prime(x))

        elif (function == ActivationFunction.Relu):
            return (
                lambda x: ActivationFunction.__relu(x), 
                lambda x: ActivationFunction.__relu_prime(x))

        elif (function == ActivationFunction.LRelu):
            return (
                lambda x: ActivationFunction.__lrelu(x), 
                lambda x: ActivationFunction.__lrelu_prime(x))

        elif (function == ActivationFunction.Swish):
            return (
                lambda x: ActivationFunction.__swish(x), 
                lambda x: ActivationFunction.__swish_prime(x))

        elif (function == ActivationFunction.SoftPlus):
            return (
                lambda x: ActivationFunction.__softplus(x), 
                lambda x: ActivationFunction.__softplus_prime(x))

        elif (function == ActivationFunction.Linear):
            return (
                lambda x: ActivationFunction.__linear(x), 
                lambda x: ActivationFunction.__linear_prime(x))

        raise Exception(f"No activation function {function}")

    @staticmethod
    def __sigmoid(x, beta):
        return 1./(1.+np.exp(-beta * x))
    @staticmethod
    def __sigmoid_prime(x, beta):
        return beta * ActivationFunction.__sigmoid(x, beta)*(1.-ActivationFunction.__sigmoid(x, beta))

    @staticmethod
    def __hyperbolic_tangent(x, beta):
        return (np.exp(beta * x) - np.exp(- beta * x)) / (np.exp(beta * x) + np.exp(- beta * x))
    @staticmethod
    def __hyperbolic_tangent_prime(x, beta):
        return beta * (1 - ActivationFunction.__hyperbolic_tangent(x, beta) ** 2)

    @staticmethod
    def __relu(x):
        return np.maximum(x, 0)
    @staticmethod
    def __relu_prime(x):
        return (x > 0).astype(int)

    @staticmethod
    def __lrelu(x):
        return np.maximum(x, 0.1*x)
    @staticmethod
    def __lrelu_prime(x):
        res = (x > 0).astype(int)
        res = res.astype(float)
        res[res == 0] = 0.1
        return res

    @staticmethod
    def __softmax(x):
        exps = np.exp(x - np.max(x))
        return exps/np.sum(exps, axis=0)
    @staticmethod
    def __softmax_prime(x):
        return ActivationFunction.__softmax(x)*(1-ActivationFunction.__softmax(x))

    @staticmethod
    def __swish(x):
        return x/(1+np.exp(-x))
    @staticmethod
    def __swish_prime(x):
        return (np.exp(x)/(np.exp(x)+1)) * ((x + np.exp(x) + 1)/(np.exp(x) + 1))

    @staticmethod
    def __softplus(x):
        return np.log(1+np.exp(x))
    @staticmethod
    def __softplus_prime(x):
        return 1./(1.+np.exp(-x))

    @staticmethod
    def __linear(x):
        return x
    @staticmethod
    def __linear_prime(x):
        return 1
    