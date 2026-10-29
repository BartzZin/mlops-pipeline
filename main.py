import json
from pathlib import Path

from src.data import load_data, prepare_features
from src.evaluate import evaluate_model
from src.predict import predict
from src.train import train_model


class TPipelineEntrega:
    """Coordena as etapas do pipeline de regiões de entrega."""

    def __init__(self, ADiretorioBase: str | Path | None = None) -> None:
        self.FDiretorioBase = (
            Path(ADiretorioBase).resolve()
            if ADiretorioBase is not None
            else Path(__file__).resolve().parent
        )

    def Executar(self) -> dict:
        LCaminhoDados = self.FDiretorioBase / "data" / "raw" / "points.csv"
        LPontos = load_data(LCaminhoDados)
        LCaracteristicas, LNormalizador = prepare_features(LPontos)
        LModelo = train_model(LCaracteristicas, n_clusters=5, random_state=42)
        LMetricas = evaluate_model(LModelo, LCaracteristicas)
        LPredicao = predict(LModelo, LNormalizador, lat=-23.55, lng=-46.63)
        LResultado = {"metrics": LMetricas, "sample_prediction": LPredicao}
        self.SalvarResultado(LResultado)
        return LResultado

    def SalvarResultado(self, AResultado: dict) -> None:
        LCaminhoSaida = self.FDiretorioBase / "outputs" / "metrics.json"
        LCaminhoSaida.parent.mkdir(parents=True, exist_ok=True)
        LCaminhoSaida.write_text(
            json.dumps(AResultado, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    LResultado = TPipelineEntrega().Executar()
    print("Pipeline executado com sucesso!")
    print(json.dumps(LResultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
