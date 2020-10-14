import random
import numpy as np

def randFloat(start, end):
    return random.random() * (end - start) + start

def generateTrainData(step_lower_value, noise_range = 0.05):
    X = []
    D = []

    for i in range(0, 10):
        x1 = 0 + randFloat(-noise_range, noise_range)
        x2 = 0 + randFloat(-noise_range, noise_range)
        d = step_lower_value
        X.append((x1, x2))
        D.append(d)

    for i in range(0, 10):
        x1 = 1 + randFloat(-noise_range, noise_range)
        x2 = 0 + randFloat(-noise_range, noise_range)
        d = step_lower_value
        X.append((x1, x2))
        D.append(d)

    for i in range(0, 10):
        x1 = 0 + randFloat(-noise_range, noise_range)
        x2 = 1 + randFloat(-noise_range, noise_range)
        d = step_lower_value
        X.append((x1, x2))
        D.append(d)

    for i in range(0, 10):
        x1 = 1 + randFloat(-noise_range, noise_range)
        x2 = 1 + randFloat(-noise_range, noise_range)
        d = 1
        X.append(np.array([x1, x2]))
        D.append(d)

    p = np.random.permutation(len(X))
    return np.array(X)[p], np.array(D)[p]