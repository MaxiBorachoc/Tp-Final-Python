"""Procesamiento de datos con Pandas para AutoValor."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from src.exceptions import DatasetValidationError


REQUIRED_COLUMNS = [
    "marca",
    "modelo",
    "anio",
    "kilometraje",
    "combustible",
    "transmision",
    "precio_usd",
]

CATEGORICAL_COLUMNS = ["marca", "modelo", "combustible", "transmision"]
NUMERIC_COLUMNS = ["anio", "kilometraje", "precio_usd"]
FEATURE_COLUMNS = [
    "marca",
    "modelo",
    "anio",
    "kilometraje",
    "combustible",
    "transmision",
]


def read_csv(source: str | Path | Any) -> pd.DataFrame:
    """Lee un CSV desde una ruta o archivo cargado por Streamlit."""
    try:
        return pd.read_csv(source)
    except EmptyDataError as exc:
        raise DatasetValidationError("El archivo CSV esta vacio.") from exc
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(
            "No se pudo leer la codificacion del archivo. Usa un CSV en UTF-8."
        ) from exc
    except ParserError as exc:
        raise DatasetValidationError(
            "No se pudo interpretar el CSV. Revisa el separador y el formato."
        ) from exc
    except Exception as exc:
        raise DatasetValidationError(f"No se pudo leer el CSV: {exc}") from exc


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas para trabajar con el esquema interno."""
    normalized = df.copy()
    normalized.columns = [
        str(column).strip().lower().replace(" ", "_") for column in normalized.columns
    ]
    return normalized


def validate_schema(df: pd.DataFrame) -> None:
    """Valida que el dataset tenga todas las columnas obligatorias."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        expected = ", ".join(REQUIRED_COLUMNS)
        missing = ", ".join(missing_columns)
        raise DatasetValidationError(
            "El CSV no tiene el esquema esperado. "
            f"Faltan columnas: {missing}. Columnas requeridas: {expected}."
        )


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Limpia el dataset y devuelve metricas visibles del proceso."""
    data = normalize_column_names(df)
    validate_schema(data)

    data = data[REQUIRED_COLUMNS].copy()
    filas_originales = len(data)
    nulos_originales = data.isna().sum().to_dict()
    duplicados_originales = int(data.duplicated().sum())

    data = data.dropna(how="all")
    filas_completamente_vacias = filas_originales - len(data)

    for column in CATEGORICAL_COLUMNS:
        data[column] = (
            data[column]
            .fillna("Desconocido")
            .astype(str)
            .str.strip()
            .replace("", "Desconocido")
            .str.title()
        )

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    filas_sin_precio = int(data["precio_usd"].isna().sum())
    data = data.dropna(subset=["precio_usd"])

    median_values = {}
    for column in ["anio", "kilometraje"]:
        median = data[column].median()
        if pd.isna(median):
            raise DatasetValidationError(
                "No hay suficientes datos numericos validos para limpiar el dataset."
            )
        median_values[column] = float(median)
        data[column] = data[column].fillna(median)

    data["anio"] = data["anio"].round().astype(int)
    data["kilometraje"] = data["kilometraje"].round(0)
    data["precio_usd"] = data["precio_usd"].round(2)

    max_year = datetime.now().year + 1
    valid_mask = (
        (data["precio_usd"] > 0)
        & (data["kilometraje"] >= 0)
        & (data["anio"].between(1980, max_year))
    )
    filas_invalidas_eliminadas = int((~valid_mask).sum())
    data = data.loc[valid_mask].copy()

    duplicados_eliminados = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)

    if data.empty:
        raise DatasetValidationError(
            "No quedaron datos validos despues de aplicar la limpieza."
        )

    metrics = {
        "filas_originales": filas_originales,
        "filas_finales": len(data),
        "filas_eliminadas_o_corregidas": filas_originales - len(data),
        "filas_completamente_vacias": filas_completamente_vacias,
        "filas_sin_precio_eliminadas": filas_sin_precio,
        "filas_invalidas_eliminadas": filas_invalidas_eliminadas,
        "duplicados_encontrados": duplicados_originales,
        "duplicados_eliminados": duplicados_eliminados,
        "nulos_originales": nulos_originales,
        "medianas_usadas": median_values,
        "estrategia_categoricos": "Completar faltantes con Desconocido.",
    }
    return data, metrics


def get_basic_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    """Calcula metricas descriptivas principales."""
    return {
        "registros": int(len(df)),
        "precio_promedio": float(df["precio_usd"].mean()),
        "kilometraje_promedio": float(df["kilometraje"].mean()),
        "anio_minimo": int(df["anio"].min()),
        "anio_maximo": int(df["anio"].max()),
    }


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve estadisticas descriptivas de variables numericas."""
    return df[["anio", "kilometraje", "precio_usd"]].describe().round(2)


def get_average_price_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por marca y calcula precio promedio."""
    return (
        df.groupby("marca", as_index=False)["precio_usd"]
        .mean()
        .rename(columns={"precio_usd": "precio_promedio_usd"})
        .sort_values("precio_promedio_usd", ascending=False)
        .round({"precio_promedio_usd": 2})
    )


def get_average_price_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por anio y calcula precio promedio."""
    return (
        df.groupby("anio", as_index=False)["precio_usd"]
        .mean()
        .rename(columns={"precio_usd": "precio_promedio_usd"})
        .sort_values("anio")
        .round({"precio_promedio_usd": 2})
    )


def apply_filters(
    df: pd.DataFrame,
    marcas: list[str],
    combustibles: list[str],
    transmisiones: list[str],
    year_range: tuple[int, int],
) -> pd.DataFrame:
    """Aplica filtros exploratorios sin alterar el dataset de entrenamiento."""
    filtered = df.copy()
    if marcas:
        filtered = filtered[filtered["marca"].isin(marcas)]
    if combustibles:
        filtered = filtered[filtered["combustible"].isin(combustibles)]
    if transmisiones:
        filtered = filtered[filtered["transmision"].isin(transmisiones)]
    return filtered[
        filtered["anio"].between(int(year_range[0]), int(year_range[1]))
    ].copy()
