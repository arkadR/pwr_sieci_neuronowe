import numpy as np

class WeightInitialization:
    @staticmethod
    def Random(sizes, weight_range):
        return [np.random.randn(sizes[i+1], sizes[i]) * weight_range
                for i in range(len(sizes) - 1)]

    @staticmethod
    def Xavier(sizes, _):
        return [np.random.normal(0, 2/(sizes[i] + sizes[i+1]), size=(sizes[i+1], sizes[i])) 
                for i in range(len(sizes) - 1)]

    @staticmethod
    def He(sizes, _):
        return [np.random.normal(0, 2/sizes[i+1], size=(sizes[i+1], sizes[i]))
                for i in range(len(sizes) - 1)]