# top_three_models/automl_pipeline.py
from sklearn.pipeline import Pipeline
from utils.preprocessing import get_preprocessing

# ====== WRAP AUTOML PIPELINE ======
def wrap_automl_pipeline(model, x_train):
    preprocessor = get_preprocessing(x_train)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline