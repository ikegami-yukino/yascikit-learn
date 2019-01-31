import numpy as np
from scipy import stats

from k_medoids import KMedoids
from scipy.spatial.distance import pdist, squareform


class XMedoids:
    def __init__(self, n_clusters=2, max_iter=9999, n_init=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init

    def fit(self, X):
        self.__clusters = []
        clusters = self.Cluster.build(X, KMedoids(n_clusters=self.n_clusters,
                                                  max_iter=self.max_iter,
                                                  n_init=self.n_init).fit(squareform(pdist(X))))
        self.__recursively_split(clusters)

        self.labels_ = np.empty(X.shape[0], dtype=np.intp)
        for i, c in enumerate(self.__clusters):
            self.labels_[c.index] = i

        self.cluster_centers_ = np.array([c.center for c in self.__clusters])
        self.cluster_log_likelihoods_ = np.array([c.log_likelihood() for c in self.__clusters])
        self.cluster_sizes_ = np.array([c.size for c in self.__clusters])

        return self

    def __recursively_split(self, clusters):
        for cluster in clusters:
            if cluster.size <= 3:
                self.__clusters.append(cluster)
                continue

            k_means = KMedoids(2, max_iter=self.max_iter, n_init=self.n_init).fit(squareform(pdist(cluster.data)))
            c1, c2 = self.Cluster.build(cluster.data, k_means, cluster.index)

            beta = np.linalg.norm(c1.center - c2.center) / np.sqrt(np.linalg.det(c1.cov) + np.linalg.det(c2.cov))
            alpha = 0.5 / stats.norm.cdf(beta)
            bic = -2 * (cluster.size * np.log(
                alpha) + c1.log_likelihood() + c2.log_likelihood()) + 2 * cluster.df * np.log(cluster.size)

            if bic < cluster.bic():
                self.__recursively_split([c1, c2])
            else:
                self.__clusters.append(cluster)

    class Cluster:
        @classmethod
        def build(cls, X, cluster_model, index=None):
            if index is None:
                index = np.array(range(0, X.shape[0]))
            labels = range(0, len(np.unique(cluster_model.labels_)))

            return tuple(cls(X, index, cluster_model, label) for label in labels)

        def __init__(self, X, index, cluster_model, label):

            self.data = X[cluster_model.labels_ == label]
            self.index = index[cluster_model.labels_ == label]
            self.size = self.data.shape[0]
            self.df = self.data.shape[1] * (self.data.shape[1] + 3) / 2
            center_ = cluster_model.cluster_centers_[label]
            self.center = X[center_, :]
            self.cov = np.cov(self.data.T)

        def log_likelihood(self):
            return sum([stats.multivariate_normal.logpdf(x, self.center, self.cov) for x in self.data])

        def bic(self):
            return -2 * self.log_likelihood() + self.df * np.log(self.size)
