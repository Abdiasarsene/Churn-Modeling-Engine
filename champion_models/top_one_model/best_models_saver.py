# top_one_model/save_models_minio.py
import os
import joblib
import logging
from core.handler_storage import MinIOStorageHandler
from utils.config import settings
from top_one_model.best_models_loader import TopBestModelsLoaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinIOSaver:
    """
    Sauvegarde les modèles déjà chargés par TopBestModelsLoaders
    vers les préfixes bentoml_storage et joblib_storage sur MinIO
    et en local.
    """
    def __init__(self):
        # Handler MinIO pour bentoml_storage
        self.bento_handler = MinIOStorageHandler(prefix="bentoml_storage")
        self.bento_handler.setup_backend()

        # Handler MinIO pour joblib_storage
        self.joblib_handler = MinIOStorageHandler(prefix="joblib_storage")
        self.joblib_handler.setup_backend()

        # Répertoires locaux
        self.local_bento_dir = settings.bentoml_dir
        self.local_joblib_dir = settings.joblib_dir
        os.makedirs(self.local_bento_dir, exist_ok=True)
        os.makedirs(self.local_joblib_dir, exist_ok=True)

    def save_models(self, models_dict: dict):
        """
        Prend un dictionnaire de modèles {name: model} et les sauvegarde.
        """
        for name, model in models_dict.items():
            try:
                # 1️⃣ Sauvegarde locale Bentoml
                local_bento_path = os.path.join(self.local_bento_dir, f"{name}_best.pkl")
                joblib.dump(model, local_bento_path)
                logger.info(f"💾 Modèle '{name}' sauvegardé localement (bentoml) : {local_bento_path}")

                # Upload vers MinIO bentoml_storage
                self.bento_handler.upload_file(local_bento_path, f"bentoml_storage/{name}_best.pkl")
                logger.info(f"☁️ Modèle '{name}' envoyé sur MinIO (bentoml_storage/)")

                # 2️⃣ Sauvegarde locale Joblib
                local_joblib_path = os.path.join(self.local_joblib_dir, f"{name}_best.pkl")
                joblib.dump(model, local_joblib_path)
                logger.info(f"💾 Modèle '{name}' sauvegardé localement (joblib) : {local_joblib_path}")

                # Upload vers MinIO joblib_storage
                self.joblib_handler.upload_file(local_joblib_path, f"joblib_storage/{name}_best.pkl")
                logger.info(f"☁️ Modèle '{name}' envoyé sur MinIO (joblib_storage/)")

            except Exception as e:
                logger.error(f"❌ Erreur lors de la sauvegarde du modèle '{name}': {str(e)}", exc_info=True)


if __name__ == "__main__":
    # Exemple d'utilisation
    loader = TopBestModelsLoaders()
    saver = MinIOSaver()
    saver.save_models(loader.toplevel_models)