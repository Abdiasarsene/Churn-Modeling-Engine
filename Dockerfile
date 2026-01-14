FROM python:3.10-slim

WORKDIR /app

# Installer MLflow et psycopg2 pour Postgres
RUN pip install --no-cache-dir mlflow psycopg2-binary boto3

EXPOSE 5000

CMD ["mlflow", "server", \
        "--backend-store-uri", "postgresql://mlflow:mlflow@postgres:5432/mlflow", \
        "--default-artifact-root", "s3://churn-model-engine/mlflow_storage", \
        "--host", "0.0.0.0", \
        "--port", "5000"]