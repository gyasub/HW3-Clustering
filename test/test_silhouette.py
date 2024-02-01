# write your silhouette score unit tests here

import pytest
from cluster.silhouette import Silhouette
from cluster.utils import make_clusters
from sklearn.metrics import silhouette_samples
import numpy as np


def test_silhouette_score():        # To test silhouette scoring against Sklearn

    # Get a random matrix
    mat, labels = make_clusters()

    my_silhouette = Silhouette()

    my_score = my_silhouette.score(mat, labels)
    my_score_avg = np.mean(my_score)

    sk_silhouette = silhouette_samples(mat, labels)
    sk_score_avg = np.mean(sk_silhouette)

    # Asserting that the silhouette scores match within 1 unit
    assert np.allclose(my_score_avg, sk_score_avg, atol=1)



