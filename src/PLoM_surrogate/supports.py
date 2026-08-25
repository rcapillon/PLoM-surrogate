import numpy as np


def support_R(x):
    return np.array([True for _ in range(x.shape[0])])


def support_Rplus(x):
    return np.array([True if x[i] >= 0. else False for i in range(x.shape[0])])


def support_Rminus(x):
    return np.array([True if x[i] <= 0. else False for i in range(x.shape[0])])
