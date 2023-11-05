import numpy as np
from sklearn.metrics import f1_score


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


def f1_score_stats(clf, X_tr, X_t, y_tr, y_t, averaging):
    y_train = clf.predict(X_tr)
    y_test = clf.predict(X_t)
    train_f1_score = f1_score(y_tr.values, y_train, average=averaging)
    test_f1_score = f1_score(y_t.values, y_test, average=averaging)

    print(f"F1-Score, {averaging} (train): {train_f1_score:.4f}")
    print(f"F1-Score, {averaging} (test) : {test_f1_score:.4f}")
