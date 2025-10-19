# top_two_models/top_models_loader.py
import logging
import mlflow
from utils.config import settings

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======= LOAD MODELS ======
class BestModelsLoader:
    # Set up
    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_server)
        self.models_top = {
            "automl": settings.automl_first,
            "random_forest": settings.rf_second,
            "logreg": settings.logreg_third
        }
        self.toplevel_models = self.load_models_mlflow_native()

    # Load native models
    def load_models_mlflow_native(self):
        try:
            models = {}
            for name, path in self.models_top.items():
                models[name] = mlflow.sklearn.load_model(path)
            logger.info("All models are loaded")
            return models
        except Exception as e:
            logger.error(f"Error detected: {str(e)}", exc_info=True)
            return {}