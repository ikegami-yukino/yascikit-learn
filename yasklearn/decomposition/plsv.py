# -*- coding: utf-8 -*-
from collections import defaultdict
import itertools

import flati
import numpy as np
from sklearn.utils import check_random_state
"""Probabilistic Latent Semantic Visualization (PLSV)

IWATA, Tomoharu; YAMADA, Takeshi; UEDA, Naonori.
Probabilistic latent semantic visualization: topic model for visualizing documents.
In: Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining.
ACM, 2008. pp. 363-371.

岩田具治, 山田武士, 上田修功. トピックモデルに基づく文書群の可視化
情報処理学会論文誌, Vol. 50, No. 6, pp. 1234-1244 (June 2009)
http://www.kecl.ntt.co.jp/as/members/iwata/plsv.pdf
"""


def euclid(a, b):
    return np.linalg.norm(a - b)


class PLSV(object):

    def __init__(self, n_components=3, n_dimension=2, max_iter=1000, doc_topic_prior=0.01,
                 topic_word_prior=0.0001, gamma=0.0001, learning_rate=0.1, random_state=None):
        """
        Params:
            <int> n_components : num of topics
            <int> n_dimension : n_dimension for visualization
            <float> doc_topic_prior : hyper parameter of theta a.k.a alpha
            <float> topic_word_prior : hyper parameter of phi a.k.a beta
            <float> gamma : hyper parameter of xai
            <float> learning_rate
            <int> random_state
        """
        self.n_components = n_components
        self.n_dimension = n_dimension
        self.max_iter = max_iter
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.gamma = gamma * n_components
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _initialize(self, X):
        random_state = check_random_state(self.random_state)

        self.total_samples = len(X)
        self.vocab = self._extract_unique_words(X)
        self.num_vocas = len(self.vocab)

        self.doc_topic_prior *= self.total_samples

        self.doc_topic_distr = np.zeros([self.total_samples, self.n_components], dtype=np.float32)
        self.prob_zpnm = random_state.dirichlet(np.ones(self.n_components), (self.total_samples, self.num_vocas))
        self.prob_zpnm = self.prob_zpnm.astype(np.float32)

        self.phi = np.ones([self.n_components, self.n_dimension], dtype=np.float32)
        self.xai = np.zeros([self.total_samples, self.n_dimension], dtype=np.float32)
        self.theta = random_state.dirichlet(np.ones(self.num_vocas), (self.n_components))
        self.theta = self.theta.astype(np.float32)

    def _extract_unique_words(self, X):
        vocab = defaultdict(lambda: len(vocab))
        for word in flati.flatten(X, ignore=str):
            vocab[word]
        return vocab

    def _dist(self, doc_id, topic_id):
        half_euclid = lambda t_id: np.exp(-0.5 * euclid(self.xai[doc_id], self.phi[t_id]))
        distances = list(map(half_euclid, range(self.n_components)))
        denominator = sum(distances)
        numerator = distances[topic_id]
        return numerator / denominator

    def _posterior(self, d_id, topic_id, word_id):
        f = lambda t_id: self.doc_topic_distr[d_id, t_id] * self.theta[t_id, word_id]
        denominator = sum(map(f, range(self.n_components)))
        numerator = self.doc_topic_distr[d_id, topic_id] * self.theta[topic_id, word_id]
        return numerator / denominator

    def _expect(self, X):
        for (d_id, t_id) in itertools.product(range(self.total_samples), range(self.n_components)):
            self.doc_topic_distr[d_id, t_id] = self._dist(d_id, t_id)

        for (d_id, row) in enumerate(X):
            for (w_id, t_id) in itertools.product(range(len(row)), range(self.n_components)):
                self.prob_zpnm[d_id, w_id, t_id] = self._posterior(d_id, t_id, w_id)

    def _update_theta(self, X, topic_id, word_id):
        numerator = 0
        denominator = 0
        for (doc_id, _) in enumerate(X):
            for w_id in range(self.num_vocas):
                if w_id == word_id:
                    numerator = self.prob_zpnm[doc_id, word_id, topic_id]
                denominator += self.prob_zpnm[doc_id, word_id, topic_id]
        return (numerator + self.doc_topic_prior) / (denominator + self.doc_topic_prior * self.num_vocas)

    def _update_xai(self, doc_id, topic_id, grad):
        diff = grad * (self.xai[doc_id] - self.phi[topic_id]) - self.gamma * self.xai[doc_id]
        self.xai[doc_id] += self.learning_rate * diff

    def _update_phi(self, doc_id, topic_id, grad):
        diff = grad * (self.phi[topic_id] - self.xai[doc_id]) - self.topic_word_prior * self.phi[topic_id]
        self.phi[topic_id] += self.learning_rate * diff

    def _update(self, X):
        for (doc_id, doc) in enumerate(X):
            for (word, topic_id) in itertools.product(doc, range(self.n_components)):
                word_id = self.vocab[word]
                p_zpx = self.doc_topic_distr[doc_id, topic_id]
                p_z = self.prob_zpnm[doc_id, word_id, topic_id]
                grad = p_zpx - p_z
                self._update_xai(doc_id, topic_id, grad)
                self._update_phi(doc_id, topic_id, grad)

    def _maximize(self, X):
        for (t_id, w_id) in itertools.product(range(self.n_components), range(self.num_vocas)):
            self.theta[t_id, w_id] = self._update_theta(X, t_id, w_id)
        self._update(X)

    def fit(self, X):
        """Fitting the model by data X
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document word matrix.
        Returns
        -------
        self : PLSV object
        """
        self._initialize(X)

        for i in range(self.max_iter):
            self._expect(X)
            self._maximize(X)
        return self

    def fit_transform(self, X):
        """Fitting the model by data X and transform data X according to the fitted model.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document word matrix.
        Returns
        -------
        doc_topic_distr : shape=(n_samples, n_components)
            Document topic distribution for X.
        position : shape=(n_samples, n_dimension)
        """
        self.fit(X)
        return self.doc_topic_distr, self.xai
