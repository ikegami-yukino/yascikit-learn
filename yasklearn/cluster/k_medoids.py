import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


def _making_initial_medoids(distances, n_clusters):

    m, n = distances.shape

    distances_pd = pd.DataFrame({'id': range(m)})
    distances_pd = pd.concat([distances_pd, pd.DataFrame(distances,
                                                         columns=[i for i in range(n)])], axis=1)

    medoids = []
    for cluster_num in range(n_clusters):

        if cluster_num == 0:
            medoid = np.random.randint(0, m, size=1)
            medoids.extend(medoid)
        else:
            distance = distances_pd.drop(medoids, axis=0)
            distance = distance.ix[:, ['id'] + medoids]
            distance['min_distance'] = distance.min(axis=1)
            distance['min_distance_squared'] = distance['min_distance']*distance['min_distance']
            ids = distance['id'].values
            distance_values = distance['min_distance_squared'] / np.sum(distance['min_distance_squared'])
            medoid = ids[np.random.choice(range(ids.size), 1, p=distance_values)]
            medoids.extend(medoid)

    medoids = sorted(medoids)
    return medoids


class KMedoids(object):

    def __init__(self, n_clusters=8, max_iter=300, n_init=10, random_state=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def fit_predict(self, X):
        random_state = check_random_state(self.random_state )

        m, n = X.shape

        col_names = ['x_' + str(i + 1) for i in range(m)]

        best_results = None
        best_sse = np.Inf
        best_medoids = None
        for init_num in range(self.n_init):
            initial_medoids = random_state.choice(range(m), self.n_clusters, replace=False)
            tmp_X = X[:, initial_medoids]

            labels = np.argmin(tmp_X, axis=1)

            results = pd.DataFrame([range(m), labels]).T
            results.columns = ['id', 'label']


            results = pd.concat([results, pd.DataFrame(X, columns=col_names)], axis=1)

            before_medoids = initial_medoids
            new_medoids = []

            loop = 0
            while len(set(before_medoids).intersection(set(new_medoids))) != self.n_clusters and loop < self.max_iter:

                if loop > 0:
                    before_medoids = new_medoids.copy()
                    new_medoids = []

                for i in range(self.n_clusters):
                    tmp = results.ix[results['label'] == i, :].copy()

                    tmp['distance'] = np.sum(tmp.ix[:, ['x_' + str(id + 1) for id in tmp['id']]].values, axis=1)
                    tmp = tmp.reset_index(drop=True)
                    new_medoids.append(tmp.loc[tmp['distance'].idxmin(), 'id'])

                new_medoids = sorted(new_medoids)
                tmp_X = X[:, new_medoids]

                clustaling_labels = np.argmin(tmp_X, axis=1)
                results['label'] = clustaling_labels

                loop += 1

            results = results.ix[:, ['id', 'label']]
            results['flag_medoid'] = 0
            for medoid in new_medoids:
                results.ix[results['id'] == medoid, 'flag_medoid'] = 1
            tmp_X = pd.DataFrame(tmp_X, columns=['medoid_distance'+str(i) for i in range(self.n_clusters)])
            results = pd.concat([results, tmp_X], axis=1)

            sse = self._sse(distances=X, predicted_values=results['label'].values, medoids=new_medoids)

            if sse < best_sse:
                best_sse = sse
                best_results = results.copy()
                best_medoids = new_medoids.copy()

        self.labels_ = best_results['label'].values
        self.results = best_results
        self.cluster_centers_ = np.array(best_medoids)
        self.inertia_ = best_sse

        return self.labels_

    def fit(self, X):

        m, n = X.shape

        col_names = ['x_' + str(i + 1) for i in range(m)]

        best_results = None
        best_sse = np.Inf
        best_medoids = None
        for init_num in range(self.n_init):

            initial_medoids = _making_initial_medoids(X, n_clusters=self.n_clusters)
            tmp_X = X[:, initial_medoids]

            labels = np.argmin(tmp_X, axis=1)
            results = pd.DataFrame([range(m), labels]).T
            results.columns = ['id', 'label']

            results = pd.concat([results,
                                 pd.DataFrame(X, columns=col_names)], axis=1)

            before_medoids = initial_medoids
            new_medoids = []

            loop = 0
            while len(set(before_medoids).intersection(set(new_medoids))) != self.n_clusters and loop < self.max_iter:

                if loop > 0:
                    before_medoids = new_medoids.copy()
                    new_medoids = []

                for i in range(self.n_clusters):
                    tmp = results.ix[results['label'] == i, :].copy()
                    tmp['distance'] = np.sum(tmp.ix[:, ['x_' + str(id + 1) for id in tmp['id']]].values, axis=1)
                    tmp.reset_index(inplace=True)
                    new_medoids.append(tmp.loc[tmp['distance'].idxmin(), 'id'])

                new_medoids = sorted(new_medoids)
                tmp_X = X[:, new_medoids]
                labels = np.argmin(tmp_X, axis=1)
                results['label'] = labels

                loop += 1

            results = results.ix[:, ['id', 'label']]
            results['flag_medoid'] = 0
            for medoid in new_medoids:
                results.ix[results['id'] == medoid, 'flag_medoid'] = 1
            tmp_X = pd.DataFrame(tmp_X, columns=['label' + str(i) + '_distance' for i in range(self.n_clusters)])
            results = pd.concat([results, tmp_X], axis=1)

            sse = self._sse(distances=X, predicted_values=results['label'].values, medoids=new_medoids)

            if sse < best_sse:
                best_sse = sse
                best_results = results.copy()
                best_medoids = new_medoids.copy()

        self.results = best_results
        self.cluster_centers_ = np.array(best_medoids)
        self.labels_ = self.results['label'].values

        return self

    def _sse(self, distances, predicted_values, medoids):
        unique_labels = sorted(np.unique(predicted_values))

        sse = []
        for label, medoid in zip(unique_labels, medoids):
            distance = distances[medoid, predicted_values == label]
            distance_squared = distance * distance
            sse.extend(distance_squared.tolist())
        return np.sum(sse)
