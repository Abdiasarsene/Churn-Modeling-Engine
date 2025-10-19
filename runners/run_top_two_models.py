# runners/run_two_models.py
import logging
from utils.data_loader import load_and_split
from top_two_models.top_models_loader import BestModelsLoader
from top_two_models.evaluate_final_metric import ModelEvaluator
from top_two_models.compare_final_metrics import compare_models
from top_two_models.dataframe_safe_predictor import DataFrameSafePredictor
from top_two_models.store import ModelLogger

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== MAIN FUNCTION ======
def main():
    logger.info("🚀 Starting two top models selection")

    # Load data
    x_train, x_test, y_train, y_test, _ = load_and_split()
    logger.info(f"Data loaded: {len(x_train)} train, {len(x_test)} test samples")

    # Load top models
    loader = BestModelsLoader()
    models = loader.toplevel_models
    logger.info(f"✅ Models loaded: {list(models.keys())}")

    # Wrap models for DataFrame compatibility
    feature_names = x_train.columns.tolist()
    models = {
        name: DataFrameSafePredictor(model, feature_names)
        for name, model in models.items()
    }

    # Evaluate models
    evaluator = ModelEvaluator(models, x_train, x_test, y_train, y_test)
    results = evaluator.evaluate()
    ranked_models = evaluator.rank_models()
    champion = evaluator.select_champion()
    logger.info(f"✅ Champion selected: {champion}")

    # Compare models and generate JSON
    comparison_df, comparison_json = compare_models(evaluation_results=results)
    logger.info(f"✅ Comparison JSON ready at {comparison_json}")

    # 6️⃣ Store models
    logger_store = ModelLogger()
    logger_store.log_comparison(comparison_dict=results)
    logger_store.log_top_models(models=models, ranked_models=ranked_models, top_n=2)
    logger.info("✅ Top-2 models logged in MLflow")

if __name__ == "__main__":
    main()