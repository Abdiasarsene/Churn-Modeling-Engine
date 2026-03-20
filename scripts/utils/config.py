# utils/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

# ====== SETTING ======
class Settings(BaseSettings):
    # Path
    train_dataset: str
    joblib_preprocessor: str
    top_models_comparison_json: str
    top_two_comparison: str
    storage_models: str
    
    # Target
    target:str
    
    # Tracking uri
    mlflow_server: str
    mlflow_s3_endpoint: str
    mlflow_s3_endpoint_uri: str
    mlflow_artifacts_uri: str
    
    # Experiments
    baseline_experiment: str
    automl_experiment: str
    other_models_experiment: str
    top_models_experiment: str
    deploy_models_experiment: str
    
    # MinIO
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    
    # All models
    dummy: str
    logreg: str
    automl: str
    random_forest: str
    decision_tree: str
    
    # Top 3 model
    automl_first: str
    rf_second: str
    logreg_third: str
    
    model_config= SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

settings = Settings()