import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate_data import GerarDados
from main import TPipelineEntrega
from src.data import load_data, prepare_features
from src.evaluate import evaluate_model
from src.predict import predict
from src.train import train_model


class TTesteDados(unittest.TestCase):
    def test_carrega_coordenadas_na_ordem_esperada(self):
        with tempfile.TemporaryDirectory() as LDiretorio:
            LCaminho = Path(LDiretorio) / "pontos.csv"
            LCaminho.write_text(
                "identificador,lng,lat\n1,-46.63,-23.55\n2,-46.7,-23.6\n",
                encoding="utf-8",
            )
            LPontos = load_data(LCaminho)

        self.assertEqual(LPontos.shape, (2, 2))
        np.testing.assert_allclose(LPontos, [[-23.55, -46.63], [-23.6, -46.7]])

    def test_rejeita_csv_sem_colunas_ou_com_dados_invalidos(self):
        with tempfile.TemporaryDirectory() as LDiretorio:
            LCaminho = Path(LDiretorio) / "pontos.csv"
            for LConteudo in (
                "latitude,longitude\n-23,-46\n",
                "lat,lng\n",
                "lat,lng\n,-46\n",
                "lat,lng\ninf,-46\n",
            ):
                with self.subTest(conteudo=LConteudo):
                    LCaminho.write_text(LConteudo, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_data(LCaminho)

    def test_normalizacao_permite_recuperar_coordenadas_originais(self):
        LPontos = np.array([[-23.5, -46.6], [-23.7, -46.8], [-23.9, -46.5]])
        LCaracteristicas, LNormalizador = prepare_features(LPontos)

        np.testing.assert_allclose(LCaracteristicas.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(LCaracteristicas.std(axis=0), 1)
        np.testing.assert_allclose(LNormalizador.inverse_transform(LCaracteristicas), LPontos)

    def test_rejeita_caracteristicas_fora_do_formato(self):
        for LPontos in ([], [1, 2], [[1, 2, 3]], [[np.nan, 2]], [[1, np.inf]]):
            with self.subTest(pontos=LPontos):
                with self.assertRaises(ValueError):
                    prepare_features(LPontos)


class TTesteModelo(unittest.TestCase):
    def test_treino_respeita_parametros_e_avalia_grupos_conhecidos(self):
        LCaracteristicas = np.array([[-2.0, 0], [-1.0, 0], [1.0, 0], [2.0, 0]])
        LModelo = train_model(LCaracteristicas, n_clusters=2, random_state=7)

        self.assertEqual(LModelo.random_state, 7)
        self.assertEqual(len(set(LModelo.labels_)), 2)
        self.assertEqual(
            evaluate_model(LModelo, LCaracteristicas),
            {"inertia": 1.0, "silhouette_score": 0.6571, "n_clusters": 2},
        )

    def test_predicao_mede_distancia_em_coordenadas_originais(self):
        LPontos = np.array([[0.0, 1.0], [0.0, 1.0], [20.0, 20.0], [20.0, 20.0]])
        LCaracteristicas, LNormalizador = prepare_features(LPontos)
        LModelo = train_model(LCaracteristicas, n_clusters=2)

        LCentro = predict(LModelo, LNormalizador, lat=0.0, lng=1.0)
        LPredicao = predict(LModelo, LNormalizador, lat=0.0, lng=0.0)

        self.assertEqual(LCentro["distance_km"], 0.0)
        self.assertEqual(LPredicao["cluster"], LCentro["cluster"])
        self.assertEqual(LPredicao["distance_km"], 111.3195)

        for LLatitude, LLongitude in ((91, 0), (0, 181), (np.nan, 0), (0, np.inf)):
            with self.subTest(latitude=LLatitude, longitude=LLongitude):
                with self.assertRaises(ValueError):
                    predict(LModelo, LNormalizador, lat=LLatitude, lng=LLongitude)


class TTestePipeline(unittest.TestCase):
    def test_gera_dados_reproduziveis_e_salva_resultado_completo(self):
        with tempfile.TemporaryDirectory() as LDiretorio:
            LCaminhoDados = GerarDados(LDiretorio)
            LConteudoOriginal = LCaminhoDados.read_bytes()
            self.assertEqual(load_data(LCaminhoDados).shape, (300, 2))

            GerarDados(LDiretorio)
            self.assertEqual(LCaminhoDados.read_bytes(), LConteudoOriginal)

            LPipeline = TPipelineEntrega(LDiretorio)
            LResultado = LPipeline.Executar()
            LCaminhoSaida = Path(LDiretorio) / "outputs" / "metrics.json"
            LResultadoSalvo = json.loads(LCaminhoSaida.read_text(encoding="utf-8"))

            self.assertEqual(LResultadoSalvo, LResultado)
            self.assertEqual(LPipeline.Executar(), LResultado)
            self.assertEqual(set(LResultado), {"metrics", "sample_prediction"})
            self.assertEqual(
                set(LResultado["metrics"]), {"inertia", "silhouette_score", "n_clusters"}
            )
            self.assertEqual(set(LResultado["sample_prediction"]), {"cluster", "distance_km"})
            self.assertEqual(LResultado["metrics"]["n_clusters"], 5)
            self.assertGreater(LResultado["metrics"]["inertia"], 0)
            self.assertGreaterEqual(LResultado["metrics"]["silhouette_score"], -1)
            self.assertLessEqual(LResultado["metrics"]["silhouette_score"], 1)
            self.assertIn(LResultado["sample_prediction"]["cluster"], range(5))
            self.assertGreaterEqual(LResultado["sample_prediction"]["distance_km"], 0)


if __name__ == "__main__":
    unittest.main()
