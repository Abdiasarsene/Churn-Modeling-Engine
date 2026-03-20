# top_two_models/dataframe_safe_predictor.py
import pandas as pd
import numpy as np

# ====== DATAFRAME SAFE OF PREDICTION ======
class DataFrameSafePredictor:
    # Set up
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names

    # Ensure safety of dataframe
    def _ensure_dataframe(self, X):
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=self.feature_names)
        return X

    # Predict
    def predict(self, X):
        X_df = self._ensure_dataframe(X)
        return self.model.predict(X_df)

    def predict_proba(self, X):
        X_df = self._ensure_dataframe(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_df)
        raise AttributeError(f" {type(self.model).__name__} model doesn't support `predict_proba`.")