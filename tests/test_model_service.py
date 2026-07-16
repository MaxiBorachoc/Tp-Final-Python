import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.exceptions import ModelNotTrainedError
from src.model_service import load_model, predict_price, train_model


def build_training_dataset() -> pd.DataFrame:
    rows = []
    brands = [
        ("Toyota", "Corolla", 22000),
        ("Ford", "Focus", 18500),
        ("Fiat", "Cronos", 16000),
        ("Volkswagen", "Polo", 17500),
    ]
    for index in range(24):
        marca, modelo, base_price = brands[index % len(brands)]
        anio = 2014 + (index % 10)
        kilometraje = 30000 + index * 4500
        precio = base_price + (anio - 2014) * 700 - kilometraje * 0.04
        rows.append(
            {
                "marca": marca,
                "modelo": modelo,
                "anio": anio,
                "kilometraje": kilometraje,
                "combustible": "Nafta" if index % 3 else "Diesel",
                "transmision": "Manual" if index % 2 else "Automatica",
                "precio_usd": round(precio, 2),
            }
        )
    return pd.DataFrame(rows)


class ModelServiceTest(unittest.TestCase):
    def test_train_model_returns_metrics_and_saves_pickle(self) -> None:
        df = build_training_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "modelo.pkl"

            model, metrics = train_model(df, artifact_path=artifact_path)

            self.assertTrue(artifact_path.exists())
            self.assertIn("r2", metrics)
            self.assertIn("mae", metrics)
            self.assertEqual(metrics["train_rows"], 19)
            self.assertEqual(metrics["test_rows"], 5)
            self.assertIsNotNone(model)

    def test_predict_price_accepts_unknown_categories(self) -> None:
        df = build_training_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "modelo.pkl"
            model, _ = train_model(df, artifact_path=artifact_path)

            prediction = predict_price(
                model,
                {
                    "marca": "Marca Nueva",
                    "modelo": "Modelo Nuevo",
                    "anio": 2022,
                    "kilometraje": 45000,
                    "combustible": "Hibrido",
                    "transmision": "Automatica",
                },
            )

            self.assertIsInstance(prediction, float)

    def test_load_model_reads_saved_pipeline(self) -> None:
        df = build_training_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "modelo.pkl"
            train_model(df, artifact_path=artifact_path)

            model = load_model(artifact_path)
            prediction = predict_price(
                model,
                {
                    "marca": "Toyota",
                    "modelo": "Corolla",
                    "anio": 2021,
                    "kilometraje": 60000,
                    "combustible": "Nafta",
                    "transmision": "Manual",
                },
            )

            self.assertGreater(prediction, 0)

    def test_predict_without_model_raises_error(self) -> None:
        with self.assertRaises(ModelNotTrainedError):
            predict_price(
                None,
                {
                    "marca": "Toyota",
                    "modelo": "Corolla",
                    "anio": 2021,
                    "kilometraje": 60000,
                    "combustible": "Nafta",
                    "transmision": "Manual",
                },
            )


if __name__ == "__main__":
    unittest.main()
