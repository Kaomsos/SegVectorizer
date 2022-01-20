from __future__ import annotations
from sklearn.mixture import GaussianMixture
import numpy as np


def get_two_means(X: np.ndarray):
    X = X.reshape(-1, 1)
    fitter = GaussianMixture(2)
    fitter.fit(X)
    fitter.means_.sort()

    # return [thin, thick]
    return fitter.means_.reshape(-1).tolist()
