import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """



        # Error Handling Statements

        if not isinstance(k, int) or k <= 0:
            raise ValueError("k should be a positive integer.")

        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError("tol should be a positive numeric value.")

        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter should be a positive integer.")

        # Initializing variables

        self.k = k
        self.tol = tol
        self.max_iter = max_iter


        self.centroids = None
        self.data_matrix = None


    # Private methods for use within the class
        
    def _calculate_distances(self, mat: np.ndarray) -> np.ndarray:
        return cdist(mat, self.centroids, 'euclidean')

    def _assign_labels(self, distances: np.ndarray) -> np.ndarray:
        return np.argmin(distances, axis=1)
    

    
    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        #checking if matrix is 2D 
        if len(mat.shape) != 2:
            raise ValueError("Input matrix 'mat' should be a 2D matrix.")

        #making copy of matrix for use in later methods
        self.data_matrix = mat.copy()
        
        #randomly assigning centroids
        self.centroids = mat[np.random.choice(mat.shape[0], self.k, replace=False)]


        for _ in range(self.max_iter):
            
            #finding euclidean distance between points and centroids
            distances = self._calculate_distances(mat)

            #assigning labels bassed on shortest distance
            labels = self._assign_labels(distances)

            #initializing an empty list for cluster means
            cluster_means = []

            for i in range(self.k):
                
                #points belonging to each cluster
                cluster_points = mat[labels == i]

                #finding mean of data points
                cluster_mean = np.mean(cluster_points, axis=0)

                #appending mean of each cluster
                cluster_means.append(cluster_mean)

            #setting the cluster means as new centroids 
            new_centroids = np.array(cluster_means)

            #computing magnitude change in centroids
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
                
            self.centroids = new_centroids
        


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        #Edge Case Handling

        if not isinstance(mat, np.ndarray):
            raise ValueError("Input matrix 'mat' should be a numpy array.") 
        
        #checking if matrix is 2D 
        if len(mat.shape) != 2:
            raise ValueError("Input matrix 'mat' should be a 2D matrix.")
        
        

        #finding distances between datapoints and the fit centroid
        distances = self._calculate_distances(mat)
        
        #array of labels to centroids
        labels = self._assign_labels(distances)
        
        
        return labels

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

        #finding distances between datapoints and the fit centroid
        distances = self._calculate_distances(self.data_matrix)
        

        #array of labels to centroids
        labels = self._assign_labels(distances)
        

        error = np.sum((self.data_matrix - self.centroids[labels])**2) / len(self.data_matrix)

        return error




    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        if self.centroids is None:
            raise ValueError("The model has not been fit yet")

        if not isinstance(self.centroids, np.ndarray):
            raise ValueError("Centroids are not a NumPy Array")
        

        return self.centroids


