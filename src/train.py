import numpy as np
from sklearn.cluster import KMeans


def train_model(
    features: np.ndarray, n_clusters: int = 5, random_state: int = 42
) -> KMeans:
    """Treina o agrupamento KMeans com a quantidade de grupos e a semente informadas."""
    LModelo = KMeans(n_clusters=n_clusters, random_state=random_state)
    LModelo.fit(features)
    return LModelo
