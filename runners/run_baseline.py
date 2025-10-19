# runners/run_baseline.py
import logging
from utils.data_loader import load_and_split
from baseline.task_type import detect_task_type
from baseline.models_baseline import BaselineBuilder
from baseline.training_baseline import BaselineTrainer
from baseline.storage_baseline import BaselineStorageMLflow

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== MAIN FUNCTION ======
def main():
    try:
        logger.info("🚀 Starting Baseline Training")
        
        # Load data
        x_train, _, y_train, _, _ = load_and_split()

        # Detect task type
        task_type = detect_task_type(y_train)
        logger.info(f"✅ Detected task type: {task_type}")

        # Build models based on task type
        builder = BaselineBuilder(task_type)
        model_dict = builder.get_models()
        logger.info(f"✅ Models ready: {list(model_dict.keys())}")

        # Train models
        trainer = BaselineTrainer(task_type=task_type)
        trained_models = trainer.train(model_dict, x_train, y_train)
        logger.info("✅ Training completed")

        # Log to MLflow
        storage = BaselineStorageMLflow()
        for model_name, model in trained_models.items():
            storage.log_baseline(model_name, model)
        logger.info("✅ All baseline models logged to MLflow")

    except Exception as e:
        logger.error(f"❌ Runner failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()