# automl/train_automl.py
import logging
from flaml import AutoML
from utils.preprocessing import get_preprocessing

# ====== LOGGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLoger(__name__)

# ====== TRAIN AUTOML MODELS ======
class TrainAutoML:
    def __init__(self, time_budget=300):
        self.time_budget = time_budget
        self.automl = None
        self.preprocessor = None

    def run(self, x_train, y_train):
        try: 
            # Preprocessing
            self.preprocessor = get_preprocessing(x_train)
            X_train_proc = self.preprocessor.fit_transform(x_train, y_train)

            # Train AutoML
            self.automl = AutoML()
            settings = {
                "time_budget": self.time_budget,
                "task": "classification",
                "metric": "roc_auc",
            }

            self.automl.fit(X_train_proc, y_train, **settings)
            return self.automl, self.preprocessor
        except Exception as e:
            logger.error(f"❌ Error Detected : {str(e)}", exc_info=True)
            raise e