import numpy as np
import pandas as pd
class MyKNeighborsRegressor:
    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        return self

    def predict(self, x):
        x = np.array(x)
        euclidean_distances = self._calc_euclidean_distances(self.X, x)
        indices = np.argsort(euclidean_distances, axis=0)[:self.n_neighbors, :]

        neighbors = self.y[indices]
        distances = np.take_along_axis(euclidean_distances, indices, axis=0)

        if self.weights == 'uniform':
            y = np.mean(neighbors, axis=0)
        elif self.weights == 'distance':
            weights = 1 / (distances + 1e-9)
            y = np.sum(neighbors * weights, axis=0) / np.sum(weights, axis=0)
        else:
            raise ValueError("weights must be 'uniform' or 'distance'")
        
        return y

    @staticmethod
    def _calc_euclidean_distances(data1, data2):
        diff = data1[:, np.newaxis, :] - data2[np.newaxis, :, :]
        squared_diff = diff ** 2
        euclidean_distances = np.sqrt(squared_diff.sum(axis=2))
        return euclidean_distances



class MyKNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        return self

    def predict(self, x):
        x = np.array(x)
        euclidean_distances = self._calc_euclidean_distances(self.X, x)
        indices = np.argsort(euclidean_distances, axis=0)[:self.n_neighbors, :]
        neighbors = self.y[indices]
        neighbors_df = pd.DataFrame(neighbors)
        most_frequent = neighbors_df.mode(axis=0).iloc[0]
        return most_frequent.to_numpy()

    @staticmethod
    def _calc_euclidean_distances(data1, data2):
        diff = data1[:, np.newaxis, :] - data2[np.newaxis, :, :]
        squared_diff = diff ** 2
        euclidean_distances = np.sqrt(squared_diff.sum(axis=2))
        return euclidean_distances
