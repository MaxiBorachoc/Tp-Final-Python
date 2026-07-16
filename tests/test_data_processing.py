import unittest

import pandas as pd

from src.data_processing import (
    clean_dataset,
    get_average_price_by_brand,
    normalize_column_names,
    validate_schema,
)
from src.exceptions import DatasetValidationError


class DataProcessingTest(unittest.TestCase):
    def test_normalize_column_names(self) -> None:
        df = pd.DataFrame(columns=[" Marca ", "Precio USD"])

        normalized = normalize_column_names(df)

        self.assertEqual(list(normalized.columns), ["marca", "precio_usd"])

    def test_validate_schema_detects_missing_columns(self) -> None:
        df = pd.DataFrame({"marca": ["Toyota"]})

        with self.assertRaises(DatasetValidationError):
            validate_schema(df)

    def test_clean_dataset_removes_invalid_rows_and_duplicates(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "marca": " toyota ",
                    "modelo": "corolla",
                    "anio": "2020",
                    "kilometraje": "50000",
                    "combustible": "nafta",
                    "transmision": "manual",
                    "precio_usd": "18000",
                },
                {
                    "marca": " toyota ",
                    "modelo": "corolla",
                    "anio": "2020",
                    "kilometraje": "50000",
                    "combustible": "nafta",
                    "transmision": "manual",
                    "precio_usd": "18000",
                },
                {
                    "marca": "Ford",
                    "modelo": "Focus",
                    "anio": "1970",
                    "kilometraje": "80000",
                    "combustible": None,
                    "transmision": "manual",
                    "precio_usd": "9000",
                },
                {
                    "marca": "Fiat",
                    "modelo": "Cronos",
                    "anio": "2022",
                    "kilometraje": "-10",
                    "combustible": "nafta",
                    "transmision": "manual",
                    "precio_usd": "12000",
                },
            ]
        )

        clean_df, report = clean_dataset(df)

        self.assertEqual(len(clean_df), 1)
        self.assertEqual(clean_df.loc[0, "marca"], "Toyota")
        self.assertEqual(report["duplicados_eliminados"], 1)
        self.assertEqual(report["filas_invalidas_eliminadas"], 2)

    def test_average_price_by_brand_uses_groupby(self) -> None:
        df = pd.DataFrame(
            {
                "marca": ["Toyota", "Toyota", "Ford"],
                "precio_usd": [10000, 20000, 12000],
            }
        )

        result = get_average_price_by_brand(df)

        toyota_price = result.loc[
            result["marca"] == "Toyota", "precio_promedio_usd"
        ].iloc[0]
        self.assertEqual(toyota_price, 15000)


if __name__ == "__main__":
    unittest.main()
