import os
import logging
import mlflow
from utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopBestModelsLoaders:
    def __init__(self):
        # Config MinIO pour les artefacts
        os.environ["AWS_ACCESS_KEY_ID"] = settings.minio_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = settings.minio_secret_key
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.mlflow_s3_endpoint_uri

        # Config MLflow Tracking
        mlflow.set_tracking_uri(settings.mlflow_server)

        # chemins des modèles (ex : runs:/...)
        self.models_top = {
            "random_forest": settings.rf_second,
            "logreg": settings.logreg_third
        }
        self.toplevel_models = self.load_best_models()

    def load_best_models(self):
        try:
            models = {}
            for name, path in self.models_top.items():
                models[name] = mlflow.sklearn.load_model(path)  # va chercher sur MinIO
            return models
        except Exception as e:
            logger.error(f"Error detected: {str(e)}", exc_info=True)
            return {}