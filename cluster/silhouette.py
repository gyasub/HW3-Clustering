import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        #number of rows
        n_samples = X.shape[0]

        #initializing score
        silhouette_scores = np.zeros(n_samples)

        # loop through rows
        for i in range(n_samples):

            #average intra cluster distance
            a = np.mean(cdist(X[y == y[i]], [X[i]]))

            #average distance to all points in the nearest neighboring cluster 
            b_values = [np.mean(cdist(X[y == j], [X[i]])) for j in set(y) - {y[i]}]
            b = np.min(b_values) if len(b_values) > 0 else 0

            #calculate silhouette score for the current observation
            silhouette_scores[i] = (b - a) / max(a, b)

        return silhouette_scores