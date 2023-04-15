import numpy as np
from scipy import stats
from sklearn.cluster import KMeans


class XMeans(KMeans):
    """
    Implementation of following method:

クラスター数を自動決定するk-meansアルゴリズムの拡張について
http://www.rd.dnc.ac.jp/~tunenori/doc/xmeans_euc.pdf
"""

    def __init__(self, n_clusters=2, **k_means_args):
        """
        Args:
            n_clusters (int): The initial number of clusters applied to KMeans
        """
        self.n_clusters = n_clusters
        self.k_means_args = k_means_args
        self._n_threads = 1

    def fit(self, X):
        """
        Args:
            X: array-like or sparse matrix, shape=(n_samples, n_features)
        Returns:
            XMeans
        """
        self.__clusters = []

        clusters = self.Cluster.build(
            X,
            KMeans(self.n_clusters, **self.k_means_args).fit(X))
        self.__recursively_split(clusters)

        self.labels_ = np.empty(X.shape[0], dtype=np.intp)
        for i, c in enumerate(self.__clusters):
            self.labels_[c.index] = i

        self.cluster_centers_ = np.array([c.center for c in self.__clusters])
        self.cluster_log_likelihoods_ = np.array(
            [c.log_likelihood() for c in self.__clusters])
        self.cluster_sizes_ = np.array([c.size for c in self.__clusters])

        return self

    def fit_predict(self, X):
        """
        Args:
            X: array-like or sparse matrix, shape=(n_samples, n_features)
        Returns:
            labels: ndarray of shape (n_samples,)
        """
        return self.fit(X).labels_

    def fit_transform(self, X):
        """
        Args:
            X: array-like or sparse matrix, shape=(n_samples, n_features)
        Returns:
            X_new: ndarray of shape (n_samples, n_clusters)
        """
        return self.fit(X)._transform(X)

    def __recursively_split(self, clusters):
        """
        Split clusters recursively.
        Args:
            clusters (list): contains instances of 'XMeans.Cluster'
        """
        for cluster in clusters:
            if cluster.size <= 3:
                self.__clusters.append(cluster)
                continue

            k_means = KMeans(2, **self.k_means_args).fit(cluster.data)
            c1, c2 = self.Cluster.build(cluster.data, k_means, cluster.index)

            det_c1 = np.linalg.det(c1.cov)
            det_c2 = np.linalg.det(c2.cov)
            if det_c1 == 0 and det_c2 == 0:
                beta = 0
            else:
                beta = np.linalg.norm(c1.center - c2.center) / np.sqrt(det_c1 +
                                                                       det_c2)
            alpha = 0.5 / stats.norm.cdf(beta)
            bic = -2 * (cluster.size * np.log(alpha) + c1.log_likelihood() +
                        c2.log_likelihood()) + 2 * cluster.df * np.log(
                            cluster.size)

            if bic < cluster.bic():
                self.__recursively_split([c1, c2])
            else:
                self.__clusters.append(cluster)

    class Cluster:
        """
        k-means法によって生成されたクラスタに関する情報を持ち、尤度やBICの計算を行うクラス
        """

        def __init__(self, X, index, k_means, label):
            # index: Xの各行におけるサンプルが元データの何行目のものかを示すベクトル
            self.data = X[k_means.labels_ == label]
            self.index = index[k_means.labels_ == label]
            self.size = self.data.shape[0]
            self.df = self.data.shape[1] * (self.data.shape[1] + 3) / 2
            self.center = k_means.cluster_centers_[label]
            self.cov = np.cov(self.data.T)

        @classmethod
        def build(cls, X, k_means, index=None):
            if index is None:
                index = np.arange(0, X.shape[0])
            labels = range(0, k_means.get_params()["n_clusters"])

            return tuple(cls(X, index, k_means, label) for label in labels)

        def log_likelihood(self):
            return sum(
                stats.multivariate_normal.logpdf(x, self.center, self.cov)
                for x in self.data)

        def bic(self):
            return -2 * self.log_likelihood() + self.df * np.log(self.size)
