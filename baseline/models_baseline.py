# trainer/baseline/models_baseline.py 
import logging
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== QUOTE MODELS TO TRAIN ======
class BaselineBuilder:
    # Set up
    def __init__(self, task_type):
        self.task_type = task_type
        self.models = self._select_models()
    
    # List of baseline models based on task type
    def _select_models(self):
        try: 
            if self.task_type == "classification":
                return {
                    "dummy": DummyClassifier(strategy="most_frequent"),
                    "logreg": LogisticRegression(max_iter=500)
                }
            
            elif self.task_type == "regression":
                return {
                    "dummy": DummyRegressor(strategy="mean"),
                    "linreg": LinearRegression()
                }
            
            else:
                raise ValueError(f"Unknown Task : {self.task_type}")
        except Exception as e:
            logger.error(f"❌ Error Detected : {str(e)}", exc_info=True)
            raise e
    
    def get_models(self):
        return self.models