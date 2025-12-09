import numpy as np
import pandas as pd
from .DecTree import MyDecisionTreeRegressor

class MyBaseModelClassifier:
    def fit(self, _, y):
        eps = 1e-10
        p = np.clip(np.mean(y), eps, 1 - eps)
        self.y_mean = np.log(p / (1 - p))

    def predict(self, X):
        return self.y_mean * np.ones(X.shape[0])

class BaseModelRegressor:
    def fit(self, _, y):
        self.y_mean = np.mean(y)

    def predict(self, X):
        return self.y_mean * np.ones(X.shape[0])

class QueueModel:
    def __init__(self):
        self.models = []
        self.alphas = []

    def add_model(self, model, alpha):
        self.models.append(model)
        self.alphas.append(alpha)

    def predict(self, X):
        F = np.zeros(X.shape[0])
        for model, alpha in zip(self.models, self.alphas):
            F += alpha * model.predict(X)
        return F

class MyGradientBoosting:
    def __init__(self, max_depth=5, min_leaf_split=10, n_estimators=10):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_leaf_split = min_leaf_split

    def fit(self, X, y):
        X, y = self._prepare_inputs(X, y)
        self.model_queue = QueueModel()
        self.base_model.fit(X, y)
        self.model_queue.add_model(self.base_model, 1)

        for i in range(self.n_estimators):
            F = self.model_queue.predict(X)
            gradient = self.calculate_gradient(F, y)

            if np.allclose(gradient, 0):
                print("Gradient is zero, stopping early.")
                break

            new_model = self.model_class(self.min_leaf_split, self.max_depth)
            new_model.fit(X, gradient)
            alpha = self.calculate_alpha(y, F, new_model.predict(X))
            self.model_queue.add_model(new_model, alpha)

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

class MyGradientBoostingClassifier(MyGradientBoosting):
    def __init__(self, max_depth=5, min_leaf_split=10, n_estimators=10):
        super().__init__(max_depth, min_leaf_split, n_estimators)
        self.model_class = MyDecisionTreeRegressor
        self.base_model = MyBaseModelClassifier()

    def calculate_gradient(self, F, y):
        p = self.sigmoid(F)
        gradient = p - y
        return gradient

    def predict_proba(self, X):
        X = self._prepare_inputs(X)
        return self.sigmoid(self.model_queue.predict(X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def calculate_alpha(self, y, F, h):
        p = self.sigmoid(F)
        g = y - p
        numerator = np.sum(g * h)
        denominator = np.sum(h ** 2)
        
        if denominator == 0:
            return 0.0

        alpha = numerator / denominator
        return alpha

    @staticmethod
    def sigmoid(y):
        return 1 / (1 + np.exp(-y))

class MyGradientBoostingRegressor(MyGradientBoosting):
    def __init__(self, max_depth=5, min_leaf_split=10, n_estimators=10):
        super().__init__(max_depth, min_leaf_split, n_estimators)
        self.model_class = MyDecisionTreeRegressor
        self.base_model = BaseModelRegressor()

    def calculate_gradient(self, F, y):
        gradient = -2 * (y - F)
        return gradient

    def predict(self, X):
        X = self._prepare_inputs(X)
        return self.model_queue.predict(X)

    def calculate_alpha(self, y, F, h):
        numerator = np.sum(h * (y - F))
        denominator = np.sum(h ** 2)

        if denominator == 0:
            return 0.0

        alpha = numerator / denominator
        return alpha