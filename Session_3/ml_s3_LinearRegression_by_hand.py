import pandas as pd
import numpy as np

# Data Extraction
X = np.array([[10],
             [15],
             [7],
             [4]])
y = np.array([11, 12, 15, 16]).reshape(-1, 1)


# Linear Regression
sample, feature = X.shape

o = np.ones((sample, 1))

X_bias = np.c_[o, X]
sample, feature = X_bias.shape

theta = np.zeros((feature, 1))

# Higher parameter
learning_rate = 0.01
max_iter = 100
cost_function = []
t = 0.0001
repeat = 4

for i in range(max_iter):
    # Hypothesis (predict)
    h = np.dot(X_bias, theta)
    error = h - y
    error = error.reshape(-1,1)
    # Gradiant Descent
    gradient = np.dot(X_bias.T, error) / sample

    # Update weights and bias
    theta = theta - learning_rate * gradient

    rmse = np.sqrt(np.mean(np.power(error, 2)))
    cost_function.append(rmse)

    #Early Stopping
    if np.array(cost_function[-repeat:]) < t:
        break