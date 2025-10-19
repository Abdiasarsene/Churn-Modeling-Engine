# trainer/baseline/storage_baseline.py
import mlflow
import mlflow.sklearn
from utils.config import settings

# ====== STORAGE MODEL INTO MLFLOW======
class BaselineStorageMLflow:
    # Set up
    def __init__(self, experiment_name=settings.baseline_experiment):
        mlflow.set_experiment(experiment_name)

    # MLflow storage
    def log_baseline(self, model_name, model):
        with mlflow.start_run(run_name=model_name):
            mlflow.sklearn.log_model(model, artifact_path="model")