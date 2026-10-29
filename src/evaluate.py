import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_model(model: KMeans, features: np.ndarray) -> dict:
    """Calcula a inércia, o índice de silhueta e a quantidade de grupos do modelo."""
    LSilhueta = silhouette_score(features, model.labels_)
    return {
        "inertia": round(float(model.inertia_), 4),
        "silhouette_score": round(float(LSilhueta), 4),
        "n_clusters": int(model.n_clusters),
    }
