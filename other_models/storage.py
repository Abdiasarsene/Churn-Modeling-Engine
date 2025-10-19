# other_models/storage.py
import mlflow
import mlflow.sklearn
from utils.config import settings

# ====== OTHER MODELS STORAGE MLFLOW ======
class OtherModelsStorageMLflow:
    def __init__(self, experiment_name=settings.other_models_experiment):
        mlflow.set_experiment(experiment_name)

    def log_model(self, model_name, model):
        with mlflow.start_run(run_name=model_name):
            mlflow.sklearn.log_model(model, artifact_path="model")