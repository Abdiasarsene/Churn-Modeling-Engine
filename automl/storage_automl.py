# automl/storage_autom.py 
import mlflow
import joblib
from utils.config import settings

# ====== STORAGE AUTOML ======
class StorageAutoML:
    # Setting MLflow
    def __init__(self, experiment_name=settings.automl_experiment):
        mlflow.set_experiment(experiment_name)

    # Log and store Preprocessor
    def log_model(self, model, preprocessor):
        with mlflow.start_run(run_name="AutoML_TrainingOnly"):
            mlflow.sklearn.log_model(model, "automl_model")
            joblib.dump(preprocessor, settings.joblib_preprocessor)
            mlflow.log_artifact(settings.joblib_preprocessor)