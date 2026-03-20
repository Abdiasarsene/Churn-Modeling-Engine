# baseline/storage_baseline.py
import logging
import mlflow
import mlflow.sklearn
import bentoml
from utils.config import settings

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== STORE MODELS TRAINED INTO MINIO ======
class BaselineFullStorageModels:
    def __init__(self, experiment_name=settings.baseline_experiment):
        # Local import to avoid circular import
        from core.handler_storage import MinIOStorageHandler

        # MLflow 
        self.mlflow_handler = MinIOStorageHandler("mlflow")
        self.mlflow_handler.setup_backend()
        existing_exp = mlflow.get_experiment_by_name(experiment_name)
        if existing_exp is None:
            mlflow.create_experiment(
                name=experiment_name,
                artifact_location= self.mlflow_handler.artifact_uri
            )
            logger.info(f"✅ MLflow Experiment '{experiment_name}' create on MinIO")
        mlflow.set_experiment(experiment_name)

        # Joblib
        self.joblib_handler = MinIOStorageHandler("joblib")
        self.joblib_handler.setup_backend()

        # BentoML
        self.bento_handler = MinIOStorageHandler("bentoml")
        self.bento_handler.setup_backend()

    # Start storing models trained on MinIO based on each prefix
    def store_models(self, trained_models: dict):
        for model_name, model in trained_models.items():
            logger.info(f"🔹 Back up of '{model_name}'")

            # MLflow
            try:
                with mlflow.start_run(run_name=model_name):
                    mlflow.sklearn.log_model(model, artifact_path="model")
                    logger.info(f"📦 '{model_name}' logged MLflow/MinIO")
            except Exception as e:
                logger.error(f"❌ MLflow log failed for '{model_name}': {e}", exc_info=True)

            # Joblib
            try:
                joblib_filename = f"{model_name}.pkl"
                self.joblib_handler.save_joblib(model, joblib_filename)
                logger.info(f"💾 '{model_name}' sauvegardé Joblib sur MinIO")
            except Exception as e:
                logger.error(f"❌ Joblib save failed for '{model_name}': {e}", exc_info=True)

            # BentoML
            try:
                bento_model = bentoml.sklearn.save_model(model_name, model)
                self.bento_handler.upload_bentoml_model(bento_model.path, f"bentoml_storage/{model_name}")
                logger.info(f"📦 '{model_name}' sauvegardé complet Bentoml sur MinIO")
            except Exception as e:
                logger.error(f"❌ Bentoml save failed for '{model_name}': {e}", exc_info=True)