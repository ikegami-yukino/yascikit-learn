import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state


class PLSA(BaseEstimator, TransformerMixin):
    """PLSA algorithm.

  Args:
    K: The number of latent topics.
    max_iter: The maximum number of iterations.
    tol: The tolerance for convergence.

  Returns:
    A tuple of (theta, phi), where theta is the topic distribution for each document and phi is the word distribution for each topic.
  """

    def __init__(self,
                 n_components=10,
                 max_iter=10,
                 tol=1.0e-7,
                 random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        check_random_state(self.random_state)

    def fit(self, X):
        """Fit the model to the data.

    Args:
      X: A NumPy array of documents, where each document is a row.

    Returns:
      The fitted model.
    """

        # Initialize the parameters.
        D = X.shape[0]
        V = X.shape[1]
        self.theta = np.random.rand(D, self.n_components)
        self.phi = np.random.rand(self.n_components, V)

        # Iterate until convergence.
        for i in range(self.max_iter):
            # E-step: Compute the expected topic distribution for each word.
            e_theta = np.dot(X, self.phi) / np.sum(
                self.phi, axis=1, keepdims=True)

            # M-step: Update the topic distribution and word distribution.
            self.theta = np.dot(X.T, e_theta) / np.sum(
                e_theta, axis=0, keepdims=True)
            self.phi = np.dot(e_theta.T, X) / np.sum(
                e_theta, axis=1, keepdims=True)

            # Check for convergence.
            diff = np.linalg.norm(self.theta - self.theta_old)
            if diff < self.tol:
                break

            self.theta_old = self.theta

        return self

    def transform(self, X):
        """Transform the data.

    Args:
      X: A NumPy array of documents, where each document is a row.

    Returns:
      A NumPy array of topic distributions, where each row corresponds to a document.
    """

        return np.dot(X, self.phi)
