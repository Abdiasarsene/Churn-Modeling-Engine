# explainers/shap_explainer.py
import shap
import matplotlib.pyplot as plt
import os
import logging
from sklearn.pipeline import Pipeline

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== SHAP EXPALINER ======
class ShapExplainer:
    # Set up
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # Extract estimator & features for SHAP
    def _extract_pipeline(self, model, X):
        try:
            if isinstance(model, Pipeline):
                preprocessor = model[:-1]
                estimator = model.named_steps['model']
                X_transformed = preprocessor.transform(X)
                feature_names = X.columns
                logger.info("✅ Model and features extraction for SHAP global done")
                return estimator, X_transformed, feature_names
            else:
                return model, X.values, X.columns
        except Exception as e:
            logger.error(f"❌ Error Detected : {str(e)}", exc_info=True)
            raise e

    def shap_global(self, model, X, model_name):
        try:
            estimator, X_ready, feature_names = self._extract_pipeline(model, X)
            explainer = shap.Explainer(estimator.predict, X_ready)
            shap_values = explainer(X_ready)

            path = os.path.join(self.output_dir, model_name, "shap_summary_bar.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            shap.summary_plot(shap_values, features=X_ready, feature_names=feature_names, plot_type="bar", show=False)
            plt.savefig(path)
            plt.close()
            logger.info(f"✅ SHAP global plot saved: {path}")
            return shap_values
        except Exception as e:
            logger.error(f"❌ Error Detected : {str(e)}", exc_info=True)
            raise e