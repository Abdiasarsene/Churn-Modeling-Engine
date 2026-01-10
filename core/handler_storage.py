# core/handler_storage.py
import os
import logging
import mlflow
import joblib
import s3fs
from core.minio_config import PROJECT_BUCKET
from utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinIOStorageHandler:

    def __init__(self, backend: str):
        self.backend = backend.lower()
        self.prefix_map = {
            "mlflow": "mlflow_storage/",
            "bentoml": "bentoml_storage/",
            "joblib": "joblib_storage/"
        }

        if self.backend not in self.prefix_map:
            raise ValueError(f"Unrecognized backend: {self.backend}")

        self.prefix = self.prefix_map[self.backend]
        self.artifact_uri = f"s3://{PROJECT_BUCKET}/{self.prefix}"

        # s3fs filesystem (only for Joblib)
        if self.backend == "joblib":
            self.fs = s3fs.S3FileSystem(
                key=settings.minio_access_key,
                secret=settings.minio_secret_key,
                client_kwargs={"endpoint_url": settings.mlflow_s3_endpoint_uri}
            )

    # ====== Setup backend ======
    def setup_backend(self):
        if self.backend == "mlflow":
            self._setup_mlflow()
        elif self.backend == "bentoml":
            self._setup_bentoml()
        elif self.backend == "joblib":
            self._setup_joblib()
        else:
            raise ValueError(f"Backend '{self.backend}' not supported.")

    # ====== MLflow ======
    def _setup_mlflow(self):
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.mlflow_s3_endpoint_uri
        os.environ["AWS_ACCESS_KEY_ID"] = settings.minio_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = settings.minio_secret_key
        mlflow.set_tracking_uri(settings.mlflow_server)
        logger.info(f"✅ MLflow configured for MinIO ({self.artifact_uri})")

    # ====== BentoML ======
    def _setup_bentoml(self):
        os.environ["BENTOML_S3_ENDPOINT_URL"] = settings.mlflow_s3_endpoint_uri
        os.environ["AWS_ACCESS_KEY_ID"] = settings.minio_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = settings.minio_secret_key
        os.environ["BENTOML_S3_BUCKET"] = PROJECT_BUCKET
        os.environ["BENTOML_S3_PREFIX"] = self.prefix
        logger.info(f"✅ BentoML configured for MinIO ({self.artifact_uri})")

    # ====== Joblib ======
    def _setup_joblib(self):
        logger.info(f"✅ Joblib target path: {self.artifact_uri}")

    # ====== Joblib Save/Load directement sur MinIO ======
    def save_joblib(self, obj, filename: str):
        path = f"{PROJECT_BUCKET}/{self.prefix}{filename}"
        with self.fs.open(f"s3://{path}", 'wb') as f:
            joblib.dump(obj, f)
        logger.info(f"💾 Joblib object saved to {path}")

    def load_joblib(self, filename: str):
        path = f"{PROJECT_BUCKET}/{self.prefix}{filename}"
        with self.fs.open(f"s3://{path}", 'rb') as f:
            obj = joblib.load(f)
        logger.info(f"📂 Joblib object loaded from {path}")
        return obj

    # ====== Upload simple file ======
    def upload_file(self, local_path: str, remote_path: str):
        if self.backend not in ["joblib", "bentoml"]:
            raise ValueError("upload_file only supported for joblib or bentoml backends")
        fs = s3fs.S3FileSystem(
            key=settings.minio_access_key,
            secret=settings.minio_secret_key,
            client_kwargs={"endpoint_url": settings.mlflow_s3_endpoint_uri}
        )
        fs.put(local_path, f"s3://{PROJECT_BUCKET}/{remote_path}")
        logger.info(f"✅ {local_path} uploaded to s3://{PROJECT_BUCKET}/{remote_path}")

    # ====== Upload d'un dossier BentoML complet ======
    def upload_bentoml_model(self, local_model_dir: str, remote_prefix: str):
        """
        Upload a full BentoML model folder to MinIO, preserving folder structure.
        """
        if not os.path.isdir(local_model_dir):
            raise ValueError(f"{local_model_dir} is not a valid directory.")

        fs = s3fs.S3FileSystem(
            key=settings.minio_access_key,
            secret=settings.minio_secret_key,
            client_kwargs={"endpoint_url": settings.mlflow_s3_endpoint_uri}
        )

        for root, dirs, files in os.walk(local_model_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                # relative path inside the model folder
                rel_path = os.path.relpath(local_file_path, local_model_dir)
                remote_path = f"{remote_prefix}/{rel_path.replace(os.sep, '/')}"
                fs.put(local_file_path, f"s3://{PROJECT_BUCKET}/{remote_path}")
                logger.info(f"✅ {local_file_path} uploaded to s3://{PROJECT_BUCKET}/{remote_path}")