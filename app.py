from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from src.data_processing import (
    apply_filters,
    clean_dataset,
    get_average_price_by_brand,
    get_average_price_by_year,
    get_basic_metrics,
    get_descriptive_stats,
    read_csv,
)
from src.exceptions import DatasetValidationError, ModelNotTrainedError, ModelTrainingError
from src.model_service import DEFAULT_MODEL_PATH, load_model, predict_price, train_model


DATASET_PATH = Path("data/autos_usados.csv")


def load_and_clean_dataset(uploaded_file: object | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Carga el CSV activo y ejecuta la limpieza de Pandas."""
    if uploaded_file is not None:
        raw_df = read_csv(uploaded_file)
    else:
        raw_df = read_csv(DATASET_PATH)
    clean_df, cleaning_report = clean_dataset(raw_df)
    return raw_df, clean_df, cleaning_report


def get_dataset_signature(clean_df: pd.DataFrame) -> tuple[int, int, int]:
    """Genera una firma simple para detectar cambios del dataset activo."""
    content_hash = int(pd.util.hash_pandas_object(clean_df, index=True).sum())
    return (
        int(len(clean_df)),
        int(clean_df.shape[1]),
        content_hash,
    )


def reset_model_if_dataset_changed(clean_df: pd.DataFrame) -> None:
    """Evita usar un modelo entrenado con otro dataset activo."""
    signature = get_dataset_signature(clean_df)
    previous_signature = st.session_state.get("dataset_signature")
    if previous_signature is not None and previous_signature != signature:
        for key in ["model", "model_metrics", "last_prediction"]:
            st.session_state.pop(key, None)
    st.session_state["dataset_signature"] = signature


def clear_runtime_state() -> None:
    """Limpia datos y modelo activos cuando no hay dataset valido."""
    for key in [
        "raw_df",
        "clean_df",
        "cleaning_report",
        "dataset_signature",
        "model",
        "model_metrics",
        "last_prediction",
    ]:
        st.session_state.pop(key, None)


def format_usd(value: float) -> str:
    """Formatea un importe en dolares estadounidenses."""
    return f"USD {value:,.2f}"


def render_cleaning_summary(report: dict) -> None:
    """Muestra el resumen de transformaciones realizadas con Pandas."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros originales", report["filas_originales"])
    col2.metric("Registros limpios", report["filas_finales"])
    col3.metric("Eliminados o corregidos", report["filas_eliminadas_o_corregidas"])
    col4.metric("Duplicados encontrados", report["duplicados_encontrados"])

    detail_rows = [
        ("Filas completamente vacias", report["filas_completamente_vacias"]),
        ("Filas sin precio eliminadas", report["filas_sin_precio_eliminadas"]),
        ("Filas invalidas eliminadas", report["filas_invalidas_eliminadas"]),
        ("Duplicados eliminados", report["duplicados_eliminados"]),
        ("Estrategia categoricos", report["estrategia_categoricos"]),
    ]
    st.dataframe(
        pd.DataFrame(detail_rows, columns=["Transformacion", "Resultado"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Valores faltantes originales")
    missing_df = pd.DataFrame(
        report["nulos_originales"].items(), columns=["Columna", "Valores faltantes"]
    )
    st.dataframe(missing_df, use_container_width=True, hide_index=True)


def render_exploration(clean_df: pd.DataFrame) -> None:
    """Muestra metricas, filtros y visualizaciones exploratorias."""
    metrics = get_basic_metrics(clean_df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros disponibles", metrics["registros"])
    col2.metric("Precio promedio", f"USD {metrics['precio_promedio']:,.2f}")
    col3.metric("Km promedio", f"{metrics['kilometraje_promedio']:,.0f}")
    col4.metric("Anios", f"{metrics['anio_minimo']} - {metrics['anio_maximo']}")

    st.subheader("Filtros exploratorios")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    marcas = sorted(clean_df["marca"].unique().tolist())
    combustibles = sorted(clean_df["combustible"].unique().tolist())
    transmisiones = sorted(clean_df["transmision"].unique().tolist())
    min_year = int(clean_df["anio"].min())
    max_year = int(clean_df["anio"].max())

    selected_marcas = filter_col1.multiselect("Marca", marcas, default=marcas)
    selected_combustibles = filter_col2.multiselect(
        "Combustible", combustibles, default=combustibles
    )
    selected_transmisiones = filter_col3.multiselect(
        "Transmision", transmisiones, default=transmisiones
    )
    if min_year == max_year:
        selected_year_range = (min_year, max_year)
        filter_col4.info(f"Anio disponible: {min_year}")
    else:
        selected_year_range = filter_col4.slider(
            "Rango de anios",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
        )

    filtered_df = apply_filters(
        clean_df,
        selected_marcas,
        selected_combustibles,
        selected_transmisiones,
        selected_year_range,
    )
    st.caption(
        "Los filtros son solo para exploracion. El modelo se entrenara con el "
        "dataset limpio completo en una fase posterior."
    )
    st.dataframe(filtered_df.head(20), use_container_width=True, hide_index=True)

    st.subheader("Estadisticas descriptivas")
    st.dataframe(get_descriptive_stats(clean_df), use_container_width=True)

    st.subheader("Precio promedio por marca")
    average_by_brand = get_average_price_by_brand(clean_df)
    st.dataframe(average_by_brand, use_container_width=True, hide_index=True)
    st.bar_chart(average_by_brand.set_index("marca"))

    st.subheader("Relacion entre kilometraje y precio")
    if filtered_df.empty:
        st.warning("No hay registros para mostrar con los filtros seleccionados.")
    else:
        st.scatter_chart(filtered_df, x="kilometraje", y="precio_usd")

    st.subheader("Precio promedio por anio")
    average_by_year = get_average_price_by_year(clean_df)
    st.line_chart(average_by_year.set_index("anio"))


def render_training(clean_df: pd.DataFrame) -> None:
    """Muestra entrenamiento, evaluacion y persistencia del modelo."""
    st.write(
        "El modelo usa como variables predictoras `marca`, `modelo`, `anio`, "
        "`kilometraje`, `combustible` y `transmision`. La variable objetivo es "
        "`precio_usd`."
    )
    st.metric("Registros disponibles", len(clean_df))
    st.caption(
        "Se entrena con el dataset limpio completo, no con los filtros visuales "
        "de la pestania de analisis."
    )

    col1, col2 = st.columns(2)
    train_clicked = col1.button("Entrenar modelo", type="primary")
    load_clicked = col2.button("Cargar modelo guardado")

    if train_clicked:
        try:
            model, metrics = train_model(clean_df)
            st.session_state["model"] = model
            st.session_state["model_metrics"] = metrics
            st.session_state.pop("last_prediction", None)
            st.success(f"Modelo entrenado y guardado en {metrics['artifact_path']}.")
        except ModelTrainingError as exc:
            st.error(str(exc))

    if load_clicked:
        try:
            st.session_state["model"] = load_model(DEFAULT_MODEL_PATH)
            st.success(f"Modelo cargado desde {DEFAULT_MODEL_PATH.as_posix()}.")
        except (ModelNotTrainedError, ModelTrainingError) as exc:
            st.error(str(exc))

    metrics = st.session_state.get("model_metrics")
    if metrics is not None:
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("R2", f"{metrics['r2']:.3f}")
        metric_col2.metric("MAE", format_usd(metrics["mae"]))
        metric_col3.metric("Entrenamiento", metrics["train_rows"])
        metric_col4.metric("Prueba", metrics["test_rows"])

        st.subheader("Variables utilizadas")
        variables_df = pd.DataFrame(
            {
                "Tipo": [
                    "Predictora",
                    "Predictora",
                    "Predictora",
                    "Predictora",
                    "Predictora",
                    "Predictora",
                    "Objetivo",
                ],
                "Variable": [
                    "marca",
                    "modelo",
                    "anio",
                    "kilometraje",
                    "combustible",
                    "transmision",
                    "precio_usd",
                ],
            }
        )
        st.dataframe(variables_df, use_container_width=True, hide_index=True)
    else:
        st.info("Todavia no hay metricas. Entrena el modelo para calcular R2 y MAE.")


def render_prediction(clean_df: pd.DataFrame) -> None:
    """Muestra formulario de prediccion con categorias del dataset activo."""
    model = st.session_state.get("model")
    if model is None:
        st.warning("Primero entrena o carga un modelo en la pestania Entrenamiento.")

    marcas = sorted(clean_df["marca"].unique().tolist())
    combustibles = sorted(clean_df["combustible"].unique().tolist())
    transmisiones = sorted(clean_df["transmision"].unique().tolist())
    max_year = datetime.now().year + 1
    min_year = 1980

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        marca = col1.selectbox("Marca", marcas)
        modelos = sorted(clean_df.loc[clean_df["marca"] == marca, "modelo"].unique())
        if len(modelos) == 0:
            modelos = sorted(clean_df["modelo"].unique().tolist())
        modelo = col2.selectbox("Modelo", modelos)

        col3, col4 = st.columns(2)
        anio = col3.number_input(
            "Anio",
            min_value=min_year,
            max_value=max_year,
            value=min(max_year, int(clean_df["anio"].median())),
            step=1,
        )
        kilometraje = col4.number_input(
            "Kilometraje",
            min_value=0,
            value=max(0, int(clean_df["kilometraje"].median())),
            step=1000,
        )

        col5, col6 = st.columns(2)
        combustible = col5.selectbox("Combustible", combustibles)
        transmision = col6.selectbox("Transmision", transmisiones)

        submitted = st.form_submit_button("Estimar precio", type="primary")

    if submitted:
        if model is None:
            st.error("No se puede predecir porque el modelo todavia no fue entrenado.")
            return

        car_data = {
            "marca": marca,
            "modelo": modelo,
            "anio": int(anio),
            "kilometraje": float(kilometraje),
            "combustible": combustible,
            "transmision": transmision,
        }
        try:
            prediction = predict_price(model, car_data)
            st.session_state["last_prediction"] = prediction
            st.success(f"Precio estimado: {format_usd(prediction)}")
        except (ModelNotTrainedError, ModelTrainingError, ValueError) as exc:
            st.error(str(exc))

    last_prediction = st.session_state.get("last_prediction")
    if last_prediction is not None:
        st.metric("Ultima estimacion", format_usd(last_prediction))


def main() -> None:
    """Aplicacion Streamlit inicial para AutoValor."""
    st.set_page_config(page_title="AutoValor", page_icon="A", layout="wide")

    st.title("AutoValor")
   

    tab_inicio, tab_datos, tab_entrenamiento, tab_prediccion = st.tabs(
        ["Inicio", "Datos y analisis", "Entrenamiento", "Prediccion"]
    )

    with tab_inicio:
        st.header("Proyecto academico")
        st.write(
            "AutoValor integrara analisis de datos con Pandas y regresion lineal "
            "multivariable para estimar precios orientativos de autos usados."
        )
        st.subheader("Clases aplicadas")
        st.markdown("- Analisis de datos con Pandas\n- Regresion lineal multivariable")

    with tab_datos:
        st.header("Datos y analisis")
        st.write(
            "Carga un CSV propio o usa el dataset sintetico incluido para ver "
            "la limpieza, estadisticas, filtros y visualizaciones con Pandas."
        )

        uploaded_file = st.file_uploader("Cargar CSV propio", type=["csv"])
        if DATASET_PATH.exists():
            st.download_button(
                "Descargar CSV de ejemplo",
                data=DATASET_PATH.read_bytes(),
                file_name="autos_usados.csv",
                mime="text/csv",
            )

        try:
            if not DATASET_PATH.exists() and uploaded_file is None:
                st.warning(
                    "No se encontro el dataset de ejemplo. Ejecuta "
                    "`python scripts/generar_dataset.py` o carga un CSV propio."
                )
                clear_runtime_state()
                raw_df = None
                clean_df = None
                cleaning_report = None
            else:
                raw_df, clean_df, cleaning_report = load_and_clean_dataset(uploaded_file)
                st.session_state["raw_df"] = raw_df
                st.session_state["clean_df"] = clean_df
                st.session_state["cleaning_report"] = cleaning_report
                reset_model_if_dataset_changed(clean_df)
        except DatasetValidationError as exc:
            st.error(str(exc))
            clear_runtime_state()
            raw_df = None
            clean_df = None
            cleaning_report = None

        if raw_df is not None and clean_df is not None and cleaning_report is not None:
            source_label = "CSV cargado" if uploaded_file is not None else "Dataset de ejemplo"
            st.success(f"Dataset activo: {source_label}")

            st.subheader("Vista previa original")
            st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)

            st.subheader("Resumen de limpieza con Pandas")
            render_cleaning_summary(cleaning_report)

            render_exploration(clean_df)

    with tab_entrenamiento:
        st.header("Entrenamiento")
        clean_df = st.session_state.get("clean_df")
        if clean_df is None:
            st.info("Primero carga o valida un dataset en la pestania Datos y analisis.")
        else:
            render_training(clean_df)

    with tab_prediccion:
        st.header("Prediccion")
        clean_df = st.session_state.get("clean_df")
        if clean_df is None:
            st.info("Primero carga o valida un dataset en la pestania Datos y analisis.")
        else:
            render_prediction(clean_df)


if __name__ == "__main__":
    main()
