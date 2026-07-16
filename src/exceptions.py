class DatasetValidationError(Exception):
    """Error de validacion del dataset."""


class ModelNotTrainedError(Exception):
    """Error al intentar usar un modelo no entrenado."""


class ModelTrainingError(Exception):
    """Error durante el entrenamiento o la persistencia del modelo."""
