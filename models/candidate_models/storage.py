# other_models/storage.py
import logging
import mlflow
import mlflow.sklearn
import bentoml
from utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OtherModelsFullStorage:
    def __init__(self, experiment_name=settings.other_models_experiment):
        # Import local pour éviter import circulaire
        from core.handler_storage import MinIOStorageHandler

        # ----- MLflow -----
        self.mlflow_handler = MinIOStorageHandler("mlflow")
        self.mlflow_handler.setup_backend()
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(
                name=experiment_name,
                artifact_location=self.mlflow_handler.artifact_uri
            )
            logger.info(f"✅ Expérience MLflow '{experiment_name}' créée sur MinIO")
        mlflow.set_experiment(experiment_name)

        # ----- Joblib -----
        self.joblib_handler = MinIOStorageHandler("joblib")
        self.joblib_handler.setup_backend()

        # ----- Bentoml -----
        self.bento_handler = MinIOStorageHandler("bentoml")
        self.bento_handler.setup_backend()

    def store_models(self, models_dict: dict):
        for model_name, model in models_dict.items():
            logger.info(f"🔹 Traitement du modèle '{model_name}'")

            # ----- 1️⃣ MLflow -----
            try:
                with mlflow.start_run(run_name=model_name):
                    mlflow.sklearn.log_model(model, artifact_path="model")
                    logger.info(f"📦 '{model_name}' loggé dans MLflow/MinIO")
            except Exception as e:
                logger.error(f"❌ MLflow log failed for '{model_name}': {e}", exc_info=True)

            # ----- 2️⃣ Joblib -----
            try:
                joblib_filename = f"{model_name}.pkl"
                self.joblib_handler.save_joblib(model, joblib_filename)
                logger.info(f"💾 '{model_name}' sauvegardé Joblib sur MinIO")
            except Exception as e:
                logger.error(f"❌ Joblib save failed for '{model_name}': {e}", exc_info=True)

            # ----- 3️⃣ Bentoml -----
            try:
                bento_model = bentoml.sklearn.save_model(model_name, model)
                self.bento_handler.upload_bentoml_model(bento_model.path, f"bentoml_storage/{model_name}")
                logger.info(f"📦 '{model_name}' sauvegardé Bentoml sur MinIO")
            except Exception as e:
                logger.error(f"❌ Bentoml save failed for '{model_name}': {e}", exc_info=True)