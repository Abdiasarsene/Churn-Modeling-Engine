# runners/run_top_three_models.py
import logging
from scripts.utils.config import settings
from scripts.utils.data_loader import load_and_split
from champion_models.top_three_models.models_loaders import load_all_models_mlflow
from champion_models.top_three_models.local_backup_of_models import save_models_to_storage
from champion_models.top_three_models.cross_validation import CrossValidationModels
from champion_models.top_three_models.compare_models import StrategicComparator
from champion_models.top_three_models.store_models import TopModelsStorage
from champion_models.top_three_models.top_models import TopModelsSelector

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== MAIN FUNCTION ======
def main():
    logger.info("🚀 Starting Top-3 Models Selection Pipeline")

    # 1️⃣ Load data
    x_train, x_test, y_train, y_test, _ = load_and_split()
    logger.info(f"📦 Data loaded: {len(x_train)} train, {len(x_test)} test samples")

    # 2️⃣ Load models from MLflow
    models = load_all_models_mlflow()
    logger.info(f"✅ Models loaded: {list(models.keys())}")

    # 3️⃣ Local backup before cross-validation
    save_models_to_storage(models, storage_dir=settings.storage_models)
    logger.info("✅ Local backup completed successfully")

    # 4️⃣ Cross-validation on training data
    cv_runner = CrossValidationModels(cv=5)
    cv_scores = cv_runner.train(x_train, y_train, models)
    logger.info("✅ Cross-validation completed successfully")

    # 5️⃣ Compare models strategically
    comparator = StrategicComparator(
        min_gain=0.02,
        max_overfit=0.05
    )
    comparison_df, comparison_json = comparator.compare(cv_scores)
    logger.info("✅ Comparison completed successfully")

    # 6️⃣ Store all CV results and comparison in MLflow
    storage = TopModelsStorage()
    storage.log_models(models_dict=models, cv_scores=cv_scores)
    storage.log_comparison(comparison_dict=comparison_json)
    logger.info("✅ Models and comparison logged in MLflow and local storage")

    # 7️⃣ Select Top-3 Models
    selector = TopModelsSelector()
    top_models = selector.select(cv_scores, comparison_df, top_n=3)
    logger.info(f"🏆 Top-3 models selected: {list(top_models.keys())}")

    # 8️⃣ Display scores
    for name, score in top_models.items():
        logger.info(f"   • {name} — ROC AUC: {score:.4f}")

    logger.info("🎯 Top-3 Models Selection Pipeline executed successfully")


if __name__ == "__main__":
    main()