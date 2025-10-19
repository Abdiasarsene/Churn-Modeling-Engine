# explainer/lime_explainer.py  
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LIME")

# ====== LIME EXPLAINER ======
class LimeExplainer:
    # Set up 
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # Prepare data for LIME
    def _prepare_for_lime(self, model, X_train, X_batch):
        try:
            if isinstance(model, Pipeline):
                preprocessor = model[:-1]
                estimator = model.named_steps['model']
                X_train_transformed = preprocessor.transform(X_train).astype(float)
                X_batch_transformed = preprocessor.transform(X_batch).astype(float)
                logger.info("✅ Estimator extracted and transformed data for LIME done")
                return estimator, X_train_transformed, X_batch_transformed
            else:
                return model, X_train.values.astype(float), X_batch.values.astype(float)
        except Exception as e:
            logger.error(f"❌ Error Detected : {str(e)}", exc_info=True)

    # LIME application
    def lime_local(self, model, X_train, X_batch, feature_names, idx, model_name, num_features=5):
        try:
            estimator, X_train_ready, X_batch_ready = self._prepare_for_lime(model, X_train, X_batch)
            
            explainer = LimeTabularExplainer(
                training_data=X_train_ready,
                feature_names=feature_names,
                class_names=["0", "1"],
                discretize_continuous=True
            )
            logger.info(f"🔹 Generating LIME explanation for {model_name}, index {idx}")
            exp = explainer.explain_instance(
                data_row=X_batch_ready[idx],
                predict_fn=estimator.predict_proba,
                num_features=num_features
            )

            lime_df = pd.DataFrame(exp.as_list(), columns=["Feature", "Contribution"])
            path = os.path.join(self.output_dir, model_name, f"lime_case_{idx}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig = exp.as_pyplot_figure()
            fig.savefig(path)
            plt.close()
            logger.info(f"✅ LIME local plot saved: {path}")
            return lime_df  
        except Exception as e: 
            logger.error(f"❌ Error Detected : {str(e)}", exc_info=True)