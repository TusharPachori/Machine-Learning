import numpy as np


def cal_cost(theta, X, y):

    m = len(y)

    predictions = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost
