from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from abc import ABC, abstractmethod
import numpy as np


class Embedder(ABC):
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray):
        self.embedder.fit(X)

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)


class TfidfEmbedder(Embedder):
    def __init__(self) -> None:
        self.embedder = TfidfTransformer()

    def transform(self, X: np.array) -> np.array:
        return self.embedder.transform(X).toarray()


class NormalizingEmbedder(Embedder):
    def __init__(self, normalization_type: str = "std") -> None:
        if normalization_type == "std":
            self.embedder = StandardScaler()
        elif normalization_type == "minmax":
            self.embedder = MinMaxScaler()

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.embedder.transform(X)


class Clusterizer:
    def __init__(self, embedder: Embedder, clustering_model) -> None:
        self.embedder = embedder
        self.clustering_model = clustering_model

    def fit(self, X: np.array):
        self.embedder.fit(X)
        x_embedded = self.embedder.transform(X)
        self.clustering_model.fit(x_embedded)

    def predict(self, X: np.array) -> np.array:
        x_embedded = self.embedder.transform(X)
        return self.clustering_model.predict(x_embedded)

    def fit_predict(self, X: np.array) -> np.array:
        self.fit(X)
        return self.predict(X)
