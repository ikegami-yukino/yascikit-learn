import numpy as np
from sklearn.utils import check_random_state


class PLSA(object):
    def __init__(self, n_components=10, max_iter=10, t=1.0e-7, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.t = t
        self.random_state = random_state

    def _initialize(self, X):
        random_state = check_random_state(self.random_state)

        # P(z)
        self.Pz = random_state.rand(self.n_components)
        # P(x|z)
        self.Px_z = random_state.rand(self.n_components, X.shape[0])
        # P(y|z)
        self.Py_z = random_state.rand(self.n_components, X.shape[1])

        self.Pz /= np.sum(self.Pz)
        self.Px_z /= np.sum(self.Px_z, axis=1)[:, np.newaxis]
        self.Py_z /= np.sum(self.Py_z, axis=1)[:, np.newaxis]

    def _em_algorithm(self, X):
        tmp = X / np.einsum('k,ki,kj->ij', self.Pz, self.Px_z, self.Py_z)
        tmp[np.isnan(tmp)] = 0
        tmp[np.isinf(tmp)] = 0

        Pz = np.einsum('ij,k,ki,kj->k', tmp, self.Pz, self.Px_z, self.Py_z)
        Px_z = np.einsum('ij,k,ki,kj->ki', tmp, self.Pz, self.Px_z, self.Py_z)
        Py_z = np.einsum('ij,k,ki,kj->kj', tmp, self.Pz, self.Px_z, self.Py_z)

        self.Pz = Pz / np.sum(Pz)
        self.Px_z = Px_z / np.sum(Px_z, axis=1)[:, np.newaxis]
        self.Py_z = Py_z / np.sum(Py_z, axis=1)[:, np.newaxis]

    def _log_likelihood(self, X):
        Pxy = np.einsum('k,ki,kj->ij', self.Pz, self.Px_z, self.Py_z)
        Pxy /= np.sum(Pxy)
        lPxy = np.log(Pxy)
        lPxy[np.isinf(lPxy)] = -1000
        return np.sum(X * lPxy)

    def fit(self, X):
        self._initialize(X)

        prev_log_likelihood = 100000
        for i in range(self.max_iter):
            self._em_algorithm(X)
            log_likelihood = self._log_likelihood(X)
            if abs((log_likelihood - prev_log_likelihood) / prev_log_likelihood) < self.t:
                break
            prev_log_likelihood = log_likelihood

        return self

    def transform(self, X):
        Pz = np.einsum('ij,k,ki,kj->k', X, self.Pz, self.Px_z, self.Py_z)
        Px_z = np.einsum('ij,k,ki,kj->ki', X, self.Pz, self.Px_z, self.Py_z)
        Pz_x = Px_z.T * Pz[np.newaxis, :]
        return (Pz_x / np.sum(Pz_x, axis=1)[:, np.newaxis])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
