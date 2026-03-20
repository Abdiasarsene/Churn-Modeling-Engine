# top_three_models/local_backup_of_models.py
import os
import joblib

# ====== LOCAL BACKUP OF MODELS ======
def save_models_to_storage(models: dict, storage_dir: str = "storage_models/") -> None:
    os.makedirs(storage_dir, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(storage_dir, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"💾 '{name}' model saved into {path}")