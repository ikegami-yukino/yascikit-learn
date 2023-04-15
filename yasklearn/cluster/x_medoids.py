import numpy as np
from scipy import stats

from .k_medoids import KMedoids


class XMedoids(KMedoids):

    def __init__(self,
                 n_clusters=2,
                 max_iter=1000,
                 n_init=1,
                 random_state=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self._n_threads = 1

    def _recursively_split(self, clusters):
        for cluster in clusters:
            if cluster.size <= 3:
                self.clusters_.append(cluster)
                continue

            k_medoids = KMedoids(2,
                                 max_iter=self.max_iter,
                                 n_init=self.n_init,
                                 random_state=self.random_state)
            k_medoids = k_medoids.fit(cluster.data)
            c1, c2 = self.Cluster.build(cluster.data, k_medoids, cluster.index)

            denominator = np.sqrt(
                np.linalg.det(c1.cov) + np.linalg.det(c2.cov))
            if denominator != 0:
                beta = np.linalg.norm(c1.center - c2.center) / denominator
            else:
                beta = 0
            alpha = 0.5 / stats.norm.cdf(beta)
            bic = -2 * (cluster.size * np.log(alpha) + c1.log_likelihood() +
                        c2.log_likelihood()) + 2 * cluster.df * np.log(
                            cluster.size)

            if bic < cluster.bic():
                self._recursively_split([c1, c2])
            else:
                self.clusters_.append(cluster)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def fit(self, X):
        """
        Args:
            X: array-like or sparse matrix, shape=(n_samples, n_features)
        Returns:
            XMedoids
        """
        self.clusters_ = []
        clusters = self.Cluster.build(
            X,
            KMedoids(n_clusters=self.n_clusters,
                     max_iter=self.max_iter,
                     n_init=self.n_init).fit(X))
        self._recursively_split(clusters)

        self.labels_ = np.empty(X.shape[0], dtype=np.intp)
        for i, c in enumerate(self.clusters_):
            self.labels_[c.index] = i

        self.cluster_centers_ = np.array([c.center for c in self.clusters_])
        self.cluster_log_likelihoods_ = np.array(
            [c.log_likelihood() for c in self.clusters_])
        self.cluster_sizes_ = np.array([c.size for c in self.clusters_])

        return self

    class Cluster(object):

        @classmethod
        def build(cls, X, cluster_model, index=None):
            if index is None:
                index = np.array(range(0, X.shape[0]))
            labels = range(0, len(np.unique(cluster_model.labels_)))

            return tuple(
                cls(X, index, cluster_model, label) for label in labels)

        def __init__(self, X, index, cluster_model, label):
            self.data = X[cluster_model.labels_ == label]
            self.index = index[cluster_model.labels_ == label]
            self.size = self.data.shape[0]
            self.df = self.data.shape[1] * (self.data.shape[1] + 3) / 2
            center_ = cluster_model.best_medoids[label]
            self.center = X[center_, :]
            self.cov = np.cov(self.data.T)

        def log_likelihood(self):
            try:
                return sum([
                    stats.multivariate_normal.logpdf(x, self.center, self.cov)
                    for x in self.data
                ])
            except Exception:
                return 0.0

        def bic(self):
            return -2 * self.log_likelihood() + self.df * np.log(self.size)
