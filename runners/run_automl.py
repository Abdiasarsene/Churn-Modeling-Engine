# runners/run_automl.py
import logging
from utils.data_loader import load_and_split
from automl.train_automl import TrainAutoML
from automl.extract_best_models import ExtractBestModels
from automl.storage_automl import StorageAutoML

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== MAIN ======
def main():
    try:
        logger.info("🚀 Starting AutoML Training")

        # Load data
        x_train, _, y_train, _, _ = load_and_split()

        # Train AutoML
        trainer = TrainAutoML()
        automl, preprocessor = trainer.run(x_train, y_train)
        logger.info("✅ AutoML training done")

        # Extract best model
        extractor = ExtractBestModels()
        top_model = extractor.run(automl)
        logger.info(f"✅ Top model extracted : {list(top_model.keys())}")

        # Log into MLflow
        storage = StorageAutoML()
        storage.log_model(top_model["automl_best_model"], preprocessor)
        logger.info("✅ Estimator & preprocessor logged into MLflow")

    except Exception as e:
        logger.error(f"❌ Erreur détectée : {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()