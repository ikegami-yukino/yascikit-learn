import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state


def _make_initial_medoids(distances, n_clusters, random_state):
    random_state = check_random_state(random_state)

    m, n = distances.shape

    distances_df = pd.DataFrame({'id': range(m)})
    distances_df = pd.concat(
        [distances_df,
         pd.DataFrame(distances, columns=[i for i in range(n)])],
        axis=1)

    medoids = []
    for cluster_num in range(n_clusters):

        if cluster_num == 0:
            medoid = random_state.randint(0, m, size=1)
            medoids.extend(medoid)
        else:
            distance = distances_df.drop(medoids, axis=0)
            distance = distance.loc[:, ['id'] + medoids]
            distance['min_distance'] = distance.min(axis=1)
            distance['min_distance_squared'] = distance[
                'min_distance'] * distance['min_distance']
            ids = distance['id'].values
            distance_values = distance['min_distance_squared'] / np.sum(
                distance['min_distance_squared'])
            medoid = ids[random_state.choice(range(ids.size),
                                             1,
                                             p=distance_values)]
            medoids.extend(medoid)

    medoids = sorted(medoids)
    return medoids


def _sse(distances, predicted_values, medoids):
    unique_labels = sorted(np.unique(predicted_values))

    sse = []
    for label, medoid in zip(unique_labels, medoids):
        distance = distances[medoid, predicted_values == label]
        distance_squared = distance * distance
        sse.extend(distance_squared.tolist())
    return np.sum(sse)


class KMedoids(KMeans):

    def __init__(self,
                 n_clusters=8,
                 max_iter=300,
                 n_init=10,
                 batch_size=100,
                 random_state=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self._n_threads = 1

    def fit(self, X):
        """
        Args:
            X: array-like or sparse matrix, shape=(n_samples, n_features)
        Returns:
            KMedoids
        """
        random_state = check_random_state(self.random_state)

        new_X = squareform(pdist(X, metric='euclidean'))
        m, n = new_X.shape

        col_names = ['x_' + str(i + 1) for i in range(n)]

        best_results = None
        best_sse = np.Inf
        best_medoids = []
        for init_num in range(self.n_init):
            initial_medoids = random_state.choice(range(m), n, replace=False)
            tmp_X = new_X[initial_medoids, :]

            labels = np.argmin(tmp_X, axis=1)

            results = pd.DataFrame([range(m), labels]).T
            results.columns = ['id', 'label']

            results = pd.concat(
                [results, pd.DataFrame(new_X, columns=col_names)], axis=1)

            previous_medoids = initial_medoids
            new_medoids = []

            loop = 0
            while len(set(previous_medoids).intersection(set(
                    new_medoids))) != self.n_clusters and loop < self.max_iter:

                if loop > 0:
                    previous_medoids = new_medoids.copy()
                    new_medoids = []

                for i in range(self.n_clusters):
                    tmp = results.loc[results['label'] == i, :].copy()

                    tmp['distance'] = np.sum(tmp.loc[:, col_names].values,
                                             axis=1)
                    tmp.reset_index(drop=True, inplace=True)
                    new_medoids.append(tmp.loc[tmp['distance'].idxmin(), 'id'])

                new_medoids = sorted(new_medoids)
                tmp_X = new_X[:, new_medoids]

                results['label'] = np.argmin(tmp_X, axis=1)

                loop += 1

            results = results.loc[:, ['id', 'label']]
            results['flag_medoid'] = 0
            for medoid in new_medoids:
                results.loc[results['id'] == medoid, 'flag_medoid'] = 1
            tmp_X = pd.DataFrame(tmp_X,
                                 columns=[
                                     'medoid_distance' + str(i)
                                     for i in range(self.n_clusters)
                                 ])
            results = pd.concat([results, tmp_X], axis=1)

            sse = _sse(distances=new_X,
                       predicted_values=results['label'].values,
                       medoids=new_medoids)

            if sse < best_sse:
                best_sse = sse
                best_results = results.copy()
                best_medoids = new_medoids.copy()

        self.labels_ = best_results['label'].values
        self.best_medoids = best_medoids
        self.cluster_centers_ = X[best_medoids]
        self.inertia_ = best_sse

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
