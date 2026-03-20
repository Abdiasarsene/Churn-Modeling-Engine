# other_models/training.py
import logging
from utils.preprocessing import get_preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== OTHER MODELS ======
class OthersModelsTrainers:
    # Set up
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = self._select_models()

    # Select models
    def _select_models(self):
        return {
            "random_forest": RandomForestClassifier(random_state=self.random_state),
            "decision_tree": DecisionTreeClassifier(random_state=self.random_state)
        }

    # Train models
    def train(self, x_train, y_train):
        try:
            models_trained = {}

            for name, model in self.models.items():
                preprocessor = get_preprocessing(x_train)
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", model)
                ])
                pipeline.fit(x_train, y_train)
                models_trained[name] = pipeline

            return models_trained
        except Exception as e:
            logger.error(f"❌ Error Detected : {str(e)}", exc_info=True)