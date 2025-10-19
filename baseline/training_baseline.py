# trainer/baseline/training_baseline.py
import logging
from sklearn.pipeline import Pipeline
from utils.preprocessing import get_preprocessing

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== TRAIN BASELINE MODELS ======
class BaselineTrainer:
    # Set up 
    def __init__(self, task_type):
        self.task_type = task_type

    # Train models
    def train(self, model_dict, x_train, y_train):
        try:
            trained_models = {}

            for name, model in model_dict.items():
                if "dummy" in name.lower():
                    pipeline = model
                else:
                    pipeline = Pipeline([
                        ("preprocessor", get_preprocessing(x_train)),
                        ("model", model)
                    ])

                pipeline.fit(x_train, y_train)
                trained_models[name] = pipeline

            return trained_models
        except Exception as e:
            logger.error(f"❌ Error Detected : {str(e)}", exc_info=True)
            raise e