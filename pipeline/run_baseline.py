# runners/run_baseline.py
import logging
from scripts.utils.data_loader import load_and_split
from models.baseline.task_type import detect_task_type
from models.baseline.models_baseline import BaselineBuilder
from models.baseline.training_baseline import BaselineTrainer
from models.baseline.storage_baseline import BaselineFullStorageModels
from scripts.core.minio_config import setup_minio_structure

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("🚀 Starting Baseline Training")

        # ====== CREATE BUCKET AND PREFIXES ======
        setup_minio_structure()
        logger.info("✅ MinIO bucket and prefixes are ready")

        # ====== LOAD DATA ======
        x_train, _, y_train, _, _ = load_and_split()
        logger.info("✅ Data split done")

        # ====== DETECT TASK TYPE ======
        task_type = detect_task_type(y_train)
        logger.info(f"✅ Detected task type: {task_type}")

        # ====== BUILD MODELS ======
        builder = BaselineBuilder(task_type)
        model_dict = builder.get_models()
        logger.info(f"✅ Models ready: {list(model_dict.keys())}")

        # ====== TRAIN MODELS ======
        trainer = BaselineTrainer(task_type=task_type)
        trained_models = trainer.train(model_dict, x_train, y_train)
        logger.info("✅ Training completed")

        # ====== STORE MODELS (MLflow, Joblib, Bentoml) ======
        storage = BaselineFullStorageModels()
        storage.store_models(trained_models)
        logger.info("✅ All baseline models stored to MLflow, Joblib, and Bentoml on MinIO")

    except Exception as e:
        logger.error(f"❌ Runner failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()