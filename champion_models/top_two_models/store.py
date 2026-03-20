# top_two_models/store.py
import os
import json
import mlflow
import logging
from utils.config import settings

# ======= LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== STORAGE MODELS ======
class ModelLogger:
    # Set up
    def __init__(self, local_path: str = settings.top_two_comparison):
        self.local_path = local_path
        mlflow.set_experiment(settings.deploy_models_experiment)

    # Log comparison
    def log_comparison(self, comparison_dict: dict):
        try:
            # Locally saved
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
            with open(self.local_path, "w") as f:
                json.dump(comparison_dict, f, indent=2)
            logger.info(f"✅ Comparison JSON saved locally at {self.local_path}")

            # Log to MLflow
            with mlflow.start_run(run_name="TopModels_Comparison"):
                mlflow.log_artifact(self.local_path)
            logger.info("✅ Comparison JSON logged in MLflow")
        except Exception as e: 
            logger.error(f"❌ Error Detected : {str(e)}", exc_info=True)
    
    # Log top models
    def log_top_models(self, models: dict, ranked_models: list, top_n=2):
        for i, (name, scores) in enumerate(ranked_models[:top_n]):
            if name not in models:
                logger.warning(f"Top model '{name}' missing from models dict")
                continue

            model = models[name]
            model_to_log = model.model if hasattr(model, "model") else model

            try:
                run_name = f"log_top_model_{name}"
                with mlflow.start_run(run_name=run_name):
                    # Log model
                    mlflow.sklearn.log_model(
                        sk_model=model_to_log,
                        artifact_path=name
                    )
                    # Log metrics
                    mlflow.log_metric("recall_score", scores["recall_score"])
                    mlflow.log_metric("latency_seconds", scores["latency_seconds"])
                logger.info(f"Top-{i+1} model '{name}' logged with metrics")
            except Exception as e:
                logger.error(f"Failed to log model '{name}': {e}", exc_info=True)