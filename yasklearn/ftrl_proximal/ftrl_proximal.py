# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import ClassifierMixin


class FTRLProximalClassifier(ClassifierMixin):
    """
    Multi class FTRLProximal

    This code modified the algorithm of the following paper to multi class prediction:
    McMahan, H. B. et al. (2013, August).
    Ad click prediction: a view from the trenches.
    In Proceedings of the 19th ACM SIGKDD (pp. 1222-1230). ACM.
    https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/41159.pdf
    """

    def __init__(self,
                 max_iter=100,
                 alpha=0.01,
                 beta=1.0,
                 l1=1.0,
                 l2=1.0,
                 random_state=None):
        self.k = 0
        self.n = 0

        self.max_iter = max_iter
        self.l1 = l1
        self.l2 = l2
        self.a = alpha
        self.b = beta

        if random_state:
            np.random.seed(random_state)

    def __initialize(self, X, y):
        self.k = len(np.unique(y))  # num of classes
        self.n = len(X[0])  # num of features
        self.bias = np.random.rand(self.k)
        self.w = np.zeros((self.k, self.n), dtype=np.float64)
        self.c = np.zeros((self.k, self.n), dtype=np.float64)
        self.z = np.zeros((self.k, self.n), dtype=np.float64)

    def _predict(self, x):

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        sign = np.ones_like(self.w)
        sign[np.where(self.w < 0)] = -1
        self.z[np.where(sign * self.w <= self.l1)] = 0
        i = np.where(sign * self.w > self.l1)
        self.z[i] = ((sign[i] * self.l1 - self.w[i]) /
                     ((self.b + np.sqrt(self.c[i])) / self.a + self.l2))
        return softmax(np.dot(self.z, x) + self.bias)

    def _train_one(self, x, y):
        pred_result = self._predict(x)

        self.bias -= np.mean(pred_result, axis=0)

        t = np.zeros((self.k, self.n))
        t[y] = x
        e = pred_result[:, np.newaxis] - t
        e2 = e**2
        s = (np.sqrt(self.c + e2) - np.sqrt(self.c)) / self.a
        self.w += e - s * self.z
        self.c += e2

    def fit_partial(self, x, y):
        if self.k == 0:
            raise RuntimeError("At first, run 'fit' method.")

        self._train_one(x, y)
        return self

    def fit(self, X, y):
        if self.k == 0:
            self.__initialize(X, y)

        num_instance = len(X)
        for _ in range(self.max_iter):
            for i in np.random.permutation(num_instance):
                self._train_one(X[i], y[i])
        return self

    def decision_function(self, X):
        if len(X.shape) > 1:
            return np.array(list(map(self._predict, X)))
        return self._predict(X)

    def predict(self, X):
        confidence_scores = self.decision_function(X)
        if len(confidence_scores.shape) > 1:
            return np.argmax(confidence_scores, axis=1)
        return np.argmax(confidence_scores)
