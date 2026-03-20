# top_three_models/loders_models.py 
import logging
import mlflow
from utils.config import settings

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== LOAD MODELS ======
def load_all_models_mlflow():
    # Models
    models_path = {
    "dummy": settings.dummy,
    "logreg": settings.logreg,
    "automl": settings.automl,
    "random_forest": settings.random_forest,
    "decision_tree": settings.decision_tree
    }

    # List dic to encase models loaded via mlflow
    models = {}
    
    for name, path in models_path.items():
        models[name] = mlflow.sklearn.load_model(path)
        logger.info(f"🔍 Type du modèle {name} : {type(models[name])}")
    return models