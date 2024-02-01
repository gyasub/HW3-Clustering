# Write your k-means unit tests here

import pytest
from cluster.kmeans import KMeans
from cluster.utils import make_clusters
from sklearn.cluster import KMeans as skkm
import numpy as np


def test_kmeans_input_shape():    # To test input data shape

    # Generate random input
    mat, _ = make_clusters()

    # Asserting that the generated matrix is 2D
    assert len(mat.shape) == 2
    

def test_kmeans_pred():     # To test prediction output

    # Generate random input
    mat, labels = make_clusters()    

    k_value = len(np.unique(labels))

    k_means = KMeans(k_value)
    k_means.fit(mat)

    # My prediction of cluster labels
    predictions = k_means.predict(mat)


    sk_cluster = skkm(k_value)
    sk_cluster.fit(mat)

    # Sklearn predictions
    sk_predictions = sk_cluster.predict(mat)

    # Asserting that number of output clusters is same 
    assert set(predictions) == set(sk_predictions)



def test_kmeans_error():    # To test error

    # Generate random input
    mat, labels = make_clusters()

    k_value = len(np.unique(labels))

    k_means = KMeans(k_value)
    k_means.fit(mat)

    # My prediction of cluster labels
    my_error = k_means.get_error()


    sk_cluster = skkm(k_value)
    sk_cluster.fit(mat)
    # Sklearn predictions
    sk_error = sk_cluster.inertia_/len(mat)

    # Asserting that the MSE matches within a reasonable range
    assert np.allclose(my_error, sk_error, atol=5)


