from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str | Path) -> np.ndarray:
    """Lê as colunas lat e lng do CSV, nessa ordem, como uma matriz numérica."""
    LTabela = pd.read_csv(filepath)
    if not {"lat", "lng"}.issubset(LTabela.columns):
        raise ValueError("O CSV deve conter as colunas 'lat' e 'lng'.")

    LPontos = LTabela.loc[:, ["lat", "lng"]].to_numpy(dtype=float)
    if LPontos.shape[0] == 0 or not np.isfinite(LPontos).all():
        raise ValueError("O CSV deve conter pontos válidos, sem valores ausentes ou infinitos.")
    return LPontos


def prepare_features(points: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Normaliza os pontos e devolve as características e o normalizador ajustado."""
    LPontos = np.asarray(points, dtype=float)
    if LPontos.ndim != 2 or LPontos.shape[1] != 2 or LPontos.shape[0] == 0:
        raise ValueError("Informe uma matriz de pontos com duas colunas: latitude e longitude.")
    if not np.isfinite(LPontos).all():
        raise ValueError("Os pontos não podem conter valores ausentes ou infinitos.")

    LNormalizador = StandardScaler()
    LCaracteristicas = LNormalizador.fit_transform(LPontos)
    return LCaracteristicas, LNormalizador
