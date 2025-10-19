# top_one_model/best_models_loaders.py
import logging
import mlflow
from utils.config import settings

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== TOP BEST MODEL ======
class TopBestModelsLoaders:
    # Set up
    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_server)
        self.models_top = {
            "random_forest": settings.rf_second,
            "logreg": settings.logreg_third
        }
        self.toplevel_models = self.load_best_models()

    def load_best_models(self):
        # Load best models
        try:
            models = {}
            for name, path in self.models_top.items():
                models[name] = mlflow.sklearn.load_model(path)
            return models
        except Exception as e:
            logger.error(f"Error detected: {str(e)}", exc_info=True)
            return {}