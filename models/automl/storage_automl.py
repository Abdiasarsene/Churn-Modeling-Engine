# automl/storage_autom.py
import mlflow
import joblib
import logging
from utils.config import settings
from core.handler_storage import MinIOStorageHandler

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== STORE AUTOML MODEL
class StorageAutoML:
    # Set up
    def __init__(self, experiment_name=settings.automl_experiment):
        # MinIO setting up
        handler = MinIOStorageHandler("mlflow")
        handler.setup_backend()

        # Get experiment
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(
                name=experiment_name,
                artifact_location=handler.artifact_uri
            )
            logger.info(f"✅ Experiment '{experiment_name}' create for MinIO")
        else:
            logger.info(f"ℹ️ Experiment '{experiment_name}' already created")

        mlflow.set_experiment(experiment_name)
        logger.info(f"✅ MLflow configured for MinIO ({handler.artifact_uri})")

    def log_model(self, model, preprocessor):
        with mlflow.start_run(run_name="AutoML_TrainingOnly"):
            # Log model
            mlflow.sklearn.log_model(model, artifact_path="automl_model")

            # Log preprocessor
            joblib.dump(preprocessor, settings.joblib_preprocessor)
            mlflow.log_artifact(settings.joblib_preprocessor)

            logger.info("✅ AutoML and preprocesssor logged on MinIO via MLflow")