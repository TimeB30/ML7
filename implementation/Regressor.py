import numpy as np
import pandas as pd
class MyLinearRegressor:
    def __init__(self, method="direct"):
        self.beta = None
        self.method = method

    def fit(self, X, y, lr=3e-3, max_iter=100):
        X, y = self._prepare_inputs(X, y)
        X = self._add_bias(X)
        self.beta = np.random.randn(X.shape[1])
        if self.method == "direct":
            self._fit_direct(X, y)
        elif self.method == "iteract":
            self._fit_iteract(X, y, lr, max_iter)
        else:
            raise AttributeError("Нет такого метода")

    def _fit_direct(self, X: np.ndarray, y: np.ndarray):
        self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y

    def _fit_iteract(self, X: np.ndarray, y: np.ndarray, lr, max_iter):
        for _ in range(max_iter):
            gradient = -2 / X.shape[0] * X.T @ (y - X @ self.beta)
            self.beta -= lr * gradient

    def predict(self, X):
        X = self._prepare_inputs(X)
        X = self._add_bias(X)
        return X @ self.beta

    @staticmethod
    def _add_bias(X: np.ndarray):
        return np.hstack((np.ones((X.shape[0], 1)), X))

    @staticmethod
    def _prepare_inputs(X, y=None):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(X, list):
            X = np.array(X)
        X = X.astype(float)

        if y is not None:
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.values
            if isinstance(y, list):
                y = np.array(y)
            if pd.api.types.is_categorical_dtype(y):
                y = y.astype(int)
            y = y.astype(float)
            return X, y

        return X

class MyLogisticRegressor:
    def __init__(self):
        self.beta = None

    def fit(self, X, y, lr=3e-4, max_iter=100):
        X, y = self._prepare_inputs(X, y)
        X = self._add_bias(X)
        self.beta = np.random.randn(X.shape[1])
        for _ in range(max_iter):
            gradient = X.T @ (self.sigmoid(X @ self.beta) - y) / X.shape[0]
            self.beta -= lr * gradient

    def predict(self, X, treshold=0.5):
        probabilities = self.predict_probabilities(X)
        return (probabilities >= treshold).astype(int)

    def predict_probabilities(self, X):
        X = self._prepare_inputs(X)
        X = self._add_bias(X)
        return self.sigmoid(X @ self.beta)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _add_bias(X: np.ndarray):
        return np.hstack((np.ones((X.shape[0], 1)), X))

    @staticmethod
    def _prepare_inputs(X, y=None):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(X, list):
            X = np.array(X)
        X = X.astype(float)

        if y is not None:
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.values
            if isinstance(y, list):
                y = np.array(y)
            if pd.api.types.is_categorical_dtype(y):
                y = y.astype(int)
            y = y.astype(float)
            return X, y

        return X