# top_two_models/evaluate_final_metric.py
import time
import logging
from sklearn.metrics import recall_score
from utils.preprocessing import get_preprocessing

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== MODELS EVALUATOR ======
class ModelEvaluator:
    # Set up
    def __init__(self, models: dict, x_train, x_test, y_train, y_test):
        self.models = models
        self.x_test = x_test
        self.y_test = y_test
        self.results = {}

        # Check if estimators contain pipeline
        try:
            if self._models_have_pipeline():
                logger.info("No preprocessing : models already include their pipeline.")
                self.preprocessor = None
                self.x_test_proc = x_test
            else:
                logger.info("Preprocessing enabled: no pipelines detected in models.")
                self.preprocessor = get_preprocessing(x_train, y_train)
                self.preprocessor.fit(x_train, y_train)
                self.x_test_proc = self.preprocessor.transform(x_test)
        except Exception as e:
            logger.error(f"❌ Error Detected : {str(e)}", exc_info=True)

    # Models have pipeline
    def _models_have_pipeline(self):
        for name, model in self.models.items():
            if hasattr(model, "steps"):
                return True
            if hasattr(model, "_model_impl") and "pipeline" in str(type(model._model_impl)).lower():
                return True
        return False

    # Evaluate metrics
    def evaluate(self):
        for name, model in self.models.items():
            try:
                # Evaluate metric & apply latency
                start = time.perf_counter()
                y_pred = model.predict(self.x_test_proc)
                latency = time.perf_counter() - start
                recall = recall_score(self.y_test, y_pred, average="binary", pos_label=1)

                # Log metrics performance into dico
                self.results[name] = {
                    "recall_score": round(recall, 4),
                    "latency_seconds": round(latency, 4)
                }
                logger.info(f"{name} → Recall: {recall:.4f}, Latence: {latency:.4f}s")
            except Exception as e:
                logger.error(f"Échec pour '{name}' : {e}", exc_info=True)
        return self.results

    def rank_models(self):
        # Valid recall score and latency score
        valid = {
            name: scores for name, scores in self.results.items()
            if scores["recall_score"] is not None and scores["latency_seconds"] is not None
        }

        # Rank models based on test score & latency
        ranked = sorted(
            valid.items(),
            key=lambda item: (-item[1]["recall_score"], item[1]["latency_seconds"])
        )
        logger.info("Classement des modèles :")
        for i, (name, scores) in enumerate(ranked, 1):
            logger.info(f"{i}. {name} → Recall: {scores['recall_score']}, Latence: {scores['latency_seconds']}s")
        return ranked

    # Select champion
    def select_champion(self):
        ranked = self.rank_models()
        return ranked[0][0] if ranked else None