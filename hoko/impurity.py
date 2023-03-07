import numpy as np


def gini_impurity(counts):
    counts = np.array(counts)
    return 1 - ((counts / counts.sum()) ** 2).sum()


def gini_split(counts_left, counts_right):
    N = np.sum(counts_left) + np.sum(counts_right)
    return np.sum(counts_left) / N * gini_impurity(counts_left) + np.sum(
        counts_right
    ) / N * gini_impurity(counts_right)
