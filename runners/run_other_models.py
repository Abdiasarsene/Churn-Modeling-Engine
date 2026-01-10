# runners/run_others_models
import logging
from core.minio_config import setup_minio_structure
from utils.data_loader import load_and_split
from other_models.training import OthersModelsTrainers
from other_models.storage import OtherModelsFullStorage

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== MAIN FUNCTION ======
def main():
    try:
        logger.info("🚀 Starting training for other models")
        
        # Set up
        setup_minio_structure()

        # Load data
        x_train, _, y_train, _, _ = load_and_split()

        # Train models
        trainer = OthersModelsTrainers()
        trained_models = trainer.train(x_train, y_train)
        logger.info(f"✅ Training completed: {list(trained_models.keys())}")

        # Log to Mlflow
        storage = OtherModelsFullStorage()
        storage.store_models(trained_models)
        logger.info("✅ All models logged to MLflow")

    except Exception as e:
        logger.error(f"❌ Runner failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()