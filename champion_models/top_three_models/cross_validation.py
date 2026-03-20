# top_three_models/cross_validation.py
import logging
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from top_three_models.automl_pipeline import wrap_automl_pipeline

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== CROSS VALIDATION ======
class CrossValidationModels:
    # Set up
    def __init__(self, cv=5):
        self.cv = cv
        self.cv_results = {}

    # Apply CV
    def train(self, x_train, y_train, models):
        for name, model in models.items():
            # Encase AutoML
            if name == "automl" and not isinstance(model, Pipeline):
                model = wrap_automl_pipeline(model, x_train)

            try:
                logger.info(f"🔍 CV starts'{name}'...")
                cv_score = cross_validate(
                    model,
                    x_train,
                    y_train,
                    cv=self.cv,
                    return_train_score=True,
                    scoring="roc_auc"
                )
                
                # Store output for CV
                self.cv_results[name] = {
                    "train_primary": float(np.mean(cv_score["train_score"])),
                    "test_primary": float(np.mean(cv_score["test_score"])),
                    "fit_time": float(np.mean(cv_score["fit_time"])),
                    "score_time": float(np.mean(cv_score["score_time"]))
                }
            except Exception as e:
                logger.error(f"❌ Échec de la CV pour '{name}' : {e}")

        return self.cv_results