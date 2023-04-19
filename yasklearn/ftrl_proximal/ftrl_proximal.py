# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state


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

        check_random_state(self.random_state)

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


class FTRLProximalRegressor(BaseEstimator, RegressorMixin):
    """FTRL Proximal regressor.

  Args:
    alpha: The learning rate.
    beta: The beta parameter.
    l1: The L1 regularization strength.
    l2: The L2 regularization strength.

  Returns:
    A fitted FTRL Proximal regressor model.
  """

    def __init__(self,
                 max_iter=1000,
                 alpha=0.1,
                 beta=1.0,
                 l1=0.01,
                 l2=0.01,
                 random_state=None):
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.random_state = random_state
        check_random_state(self.random_state)

    def fit(self, X, y):
        """Fit the model to the data.

    Args:
      X: A NumPy array of features.
      y: A NumPy array of labels.

    Returns:
      The fitted model.
    """

        D = X.shape[0]
        V = X.shape[1]
        self.w = np.zeros(V)
        g = np.zeros(V)
        z = np.zeros(V)
        n = np.zeros(V)

        for _ in range(self.max_iter):
            for t in range(D):
                # Update the weights.
                self.w[np.where(z <= self.l1)] = 0
                indices = np.where(z > self.l1)
                self.w[indices] -= ((self.beta + np.sqrt(n[indices]) / self.alpha + self.l2)**-1
                                    * z[indices] - np.sign(z[indices] * self.l1))

                # Compute the gradient.
                g = (y[t] - np.dot(X[t], self.w)) * X[t]
                g2 = g**2

                sigma = 1 / self.alpha * (np.sqrt(n + g2) - np.sqrt(n))

                # Update the z values.
                z = z + g -sigma * self.w

                # Update the gradients.
                n = n + g2

        return self

    def predict(self, X):
        """Predict the labels for the given data.

    Args:
      X: A NumPy array of features.

    Returns:
      A NumPy array of predictions.
    """

        return np.dot(X, self.w)
