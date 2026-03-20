# top_three_models/store_models.py
import mlflow
import mlflow.sklearn
import json
import os
from utils.config import settings

# ===== TOP MODELS STORE ======
class TopModelsStorage:
    # Set up
    def __init__(self, experiment_name=settings.top_models_experiment):
        mlflow.set_tracking_uri(settings.mlflow_server)
        mlflow.set_experiment(experiment_name)
        self.local_path = settings.top_models_comparison_json 

    # Log models
    def log_models(self, models_dict, cv_scores):
        for name, model in models_dict.items():
            metrics = cv_scores.get(name, {})
            with mlflow.start_run(run_name=name):
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, artifact_path="model")

    # Log comparison
    def log_comparison(self, comparison_dict):
        # Local save
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        with open(self.local_path, "w") as f:
            json.dump(comparison_dict, f, indent=2)

        # Log MKLflow
        with mlflow.start_run(run_name="TopModels_Comparison"):
            mlflow.log_artifact(self.local_path)