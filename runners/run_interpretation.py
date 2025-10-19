# runners/run_interpretation.py
import logging
from top_one_model.best_models_loader import TopBestModelsLoaders
from explainers.shap_explainer import ShapExplainer
from explainers.lime_explainer import LimeExplainer
from utils.data_loader import load_and_split

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__main__")

# ====== M  IN FUNCTION ======
def main():
    logger.info("🚀 Starting interpretability run")

    # Load models
    loader = TopBestModelsLoaders()
    models = loader.load_best_models()
    logger.info(f"✅ {len(models)} models loaded: {list(models.keys())}")

    # Prepare dataset
    x_train, x_test, y_train, y_test, _ = load_and_split()
    logger.info("✅ Dataset ready for interpretability")

    # Explainers
    shap_explainer = ShapExplainer()
    lime_explainer = LimeExplainer()

    for model_name, model in models.items():
        # SHAP global
        logger.info(f"🔍 SHAP global interpretation for model: {model_name}")
        shap_explainer.shap_global(model, x_train, model_name)

        # LIME local
        logger.info(f"🔍 LIME local interpretation for model: {model_name}")
        lime_explainer.lime_local(
            model,
            X_train=x_train,
            X_batch=x_test,
            feature_names=x_train.columns,
            idx=42,
            model_name=model_name
        )

    logger.info("🏁 Interpretability run complete!")

if __name__ == "__main__":
    main()