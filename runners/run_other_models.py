# runners/run_others_models
import logging
from utils.data_loader import load_and_split
from other_models.training import OthersModelsTrainers
from other_models.storage import OtherModelsStorageMLflow

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== MAIN FUNCTION ======
def main():
    try:
        logger.info("🚀 Starting training for other models")

        # Load data
        x_train, _, y_train, _, _ = load_and_split()

        # Train models
        trainer = OthersModelsTrainers()
        trained_models = trainer.train(x_train, y_train)
        logger.info(f"✅ Training completed: {list(trained_models.keys())}")

        # Log to Mlflow
        storage = OtherModelsStorageMLflow()
        for model_name, model in trained_models.items():
            storage.log_model(model_name, model)
        logger.info("✅ All models logged to MLflow")

    except Exception as e:
        logger.error(f"❌ Runner failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()