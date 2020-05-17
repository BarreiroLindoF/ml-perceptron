
import numpy as np

from sklearn import datasets
from sklearn.utils import shuffle


def load_IRIS(p_train=0.7, p_test=0.3, n_classes=2):
    iris = datasets.load_iris()

    X, y = iris.data, iris.target
    idx = np.array(y == 0) | np.array(y == 2)
    X = X[idx, :]
    y = y[idx]

    y[y == 0] = 1
    y[y == 2] = -1

    X, y = shuffle(X, y, random_state=1331)

    n = X.shape[0]


    n_train = int(p_train*n)

    X_train = X[:n_train, :]
    y_train = y[:n_train]
    X_test = X[n_train:, :]
    y_test = y[n_train:]

    return X_train, y_train, X_test, y_test