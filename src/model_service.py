"""Servicio de modelo para regresion lineal multivariable."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_processing import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from src.exceptions import ModelNotTrainedError, ModelTrainingError


TARGET_COLUMN = "precio_usd"
NUMERIC_FEATURES = ["anio", "kilometraje"]
DEFAULT_MODEL_PATH = Path("artifacts/modelo_autos.pkl")


def build_pipeline() -> Pipeline:
    """Construye el pipeline de preprocesamiento y regresion lineal."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
            ("numericas", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocesamiento", preprocessor),
            ("modelo", LinearRegression()),
        ]
    )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variables predictoras y variable objetivo."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ModelTrainingError(f"Faltan columnas para entrenar el modelo: {missing}.")
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def train_model(
    df: pd.DataFrame,
    artifact_path: str | Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Entrena el pipeline y devuelve el modelo junto con sus metricas."""
    if len(df) < 10:
        raise ModelTrainingError("Se necesitan al menos 10 registros para entrenar.")

    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "artifact_path": str(artifact_path),
    }
    save_model(pipeline, artifact_path)
    return pipeline, metrics


def predict_price(model: Pipeline | None, car_data: dict[str, Any] | pd.DataFrame) -> float:
    """Predice el precio para un auto usando un pipeline entrenado."""
    if model is None:
        raise ModelNotTrainedError("Primero se debe entrenar o cargar un modelo.")

    if isinstance(car_data, pd.DataFrame):
        input_df = car_data.copy()
    else:
        input_df = pd.DataFrame([car_data])

    missing_columns = [column for column in FEATURE_COLUMNS if column not in input_df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ModelTrainingError(f"Faltan datos para predecir: {missing}.")

    prediction = model.predict(input_df[FEATURE_COLUMNS])
    return float(prediction[0])


def save_model(model: Pipeline, artifact_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Guarda el pipeline entrenado usando pickle."""
    path = Path(artifact_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(model, file)
    except Exception as exc:
        raise ModelTrainingError(f"No se pudo guardar el modelo: {exc}") from exc
    return path


def load_model(artifact_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Carga un pipeline generado por esta aplicacion."""
    path = Path(artifact_path)
    if not path.exists():
        raise ModelNotTrainedError("No se encontro un modelo guardado para cargar.")

    try:
        with path.open("rb") as file:
            model = pickle.load(file)
    except Exception as exc:
        raise ModelTrainingError(f"No se pudo cargar el modelo: {exc}") from exc

    if not isinstance(model, Pipeline):
        raise ModelTrainingError("El archivo cargado no contiene un pipeline valido.")
    return model
