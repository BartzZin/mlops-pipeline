import math

from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def predict(model: KMeans, scaler: StandardScaler, lat: float, lng: float) -> dict:
    """Classifica o ponto e calcula a distância geodésica até o centro do grupo, em km."""
    if not math.isfinite(lat) or not math.isfinite(lng):
        raise ValueError("Latitude e longitude devem ser valores finitos.")
    if not -90 <= lat <= 90 or not -180 <= lng <= 180:
        raise ValueError("Latitude ou longitude fora dos limites geográficos.")

    LPontoNormalizado = scaler.transform([[lat, lng]])
    LGrupo = int(model.predict(LPontoNormalizado)[0])
    LCentroNormalizado = model.cluster_centers_[LGrupo]
    LCentro = scaler.inverse_transform([LCentroNormalizado])[0]
    LDistancia = geodesic((lat, lng), (LCentro[0], LCentro[1])).km

    return {"cluster": LGrupo, "distance_km": round(float(LDistancia), 4)}
