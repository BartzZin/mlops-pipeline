from pathlib import Path

import numpy as np
import pandas as pd


def GerarDados(ADiretorioBase: str | Path | None = None) -> Path:
    """Gera os 300 pontos do exercício sem alterar a semente global do NumPy."""
    LDiretorioBase = (
        Path(ADiretorioBase).resolve()
        if ADiretorioBase is not None
        else Path(__file__).resolve().parent
    )
    LGerador = np.random.RandomState(42)
    LQuantidade = 300
    LLatitudes = LGerador.normal(-23.55, 0.15, LQuantidade)
    LLongitudes = LGerador.normal(-46.63, 0.15, LQuantidade)
    LTabela = pd.DataFrame({"lat": LLatitudes, "lng": LLongitudes})

    LCaminhoDados = LDiretorioBase / "data" / "raw" / "points.csv"
    LCaminhoDados.parent.mkdir(parents=True, exist_ok=True)
    LTabela.to_csv(LCaminhoDados, index=False)
    return LCaminhoDados


if __name__ == "__main__":
    LCaminho = GerarDados()
    print(f"Dados gerados: 300 pontos em {LCaminho}")
