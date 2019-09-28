import numpy as np


def cal_cost(theta, X, y):

    m = len(y)

    predictions = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost


def minibatch_gradient_descent(X, y, theta, learning_rate=0.01, iterations=10, batch_size=20):
    m = len(y)
    cost_history = np.zeros(iterations)
    n_batches = int(m / batch_size)

    for it in range(iterations):
        cost = 0.0
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        for i in range(0, m, batch_size):
            X_i = X[i:i + batch_size]
            y_i = y[i:i + batch_size]

            X_i = np.c_[np.ones(len(X_i)), X_i]

            prediction = np.dot(X_i, theta)

            theta = theta - (1 / m) * learning_rate * (X_i.T.dot((prediction - y_i)))
            cost += cal_cost(theta, X_i, y_i)
        cost_history[it] = cost

    return theta, cost_history

# X = 2 * np.random.rand(100,1)
# y = 4 +3 * X+np.random.randn(100,1)
#
# lr =0.1
# n_iter = 200
#
# theta = np.random.randn(2,1)
#
#
# theta,cost_history = minibatch_gradient_descent(X,y,theta,lr,n_iter)
#
#
# print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
# print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))

