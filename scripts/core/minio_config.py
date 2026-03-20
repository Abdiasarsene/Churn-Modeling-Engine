# core/minio_config.py
import io
from minio import Minio
from utils.config import settings 
import logging

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== INITIALIZE MINIO CLIENT ======
client = Minio(
    settings.mlflow_s3_endpoint, 
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    secure=False
)

# ====== DEFINE BUCKET & PREFIXES ======
PROJECT_BUCKET = settings.minio_bucket
PREFIXES = [
    "mlflow_storage/",
    "bentoml_storage/",
    "joblib_storage/"
]

# ====== CREATE BUCKET + PREFIXES ======
def setup_minio_structure():
    try:
        # Check and create the main bucket
        if not client.bucket_exists(PROJECT_BUCKET):
            client.make_bucket(PROJECT_BUCKET)
            logger.info(f"✅ Bucket '{PROJECT_BUCKET}' created in MinIO.")
        else:
            logger.info(f"ℹ️ Bucket '{PROJECT_BUCKET}' already exists.")

        for prefix in PREFIXES:
            marker_object = f"{prefix}.keep"
            if not object_exists(marker_object):
                client.put_object(
                    PROJECT_BUCKET,
                    marker_object,
                    data=io.BytesIO(b""),
                    length=0,
                    content_type="text/plain"
                )
                logger.info(f"📂 Prefix '{prefix}' initialized in '{PROJECT_BUCKET}'.")
    except Exception as e:
        logger.error(f"❌ Error during MinIO setup: {str(e)}", exc_info=True)

def object_exists(object_name: str) -> bool:
    try:
        client.stat_object(PROJECT_BUCKET, object_name)
        return True
    except Exception:
        return False