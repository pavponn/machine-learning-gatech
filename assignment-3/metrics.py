import numpy as np


def negentropy_estimation(y):
    g = np.tanh
    # Calculate Gaussian random variable
    y_gauss = np.random.normal(size=y.shape)
    # Apply non-linearity to the data
    y_transformed = g(y)
    y_gauss_transformed = g(y_gauss)
    # Compute entropy of y and y_gauss
    H_y = -np.mean(np.log(np.cosh(y_transformed)))
    H_y_gauss = -np.mean(np.log(np.cosh(y_gauss_transformed)))
    # Calculate negentropy
    negentropy = H_y_gauss - H_y

    return negentropy
