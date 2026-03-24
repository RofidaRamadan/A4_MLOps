FROM python:3.10-slim

# Receive the arguments from GitHub Actions
ARG RUN_ID
ARG MLFLOW_TRACKING_URI

# Set them as ENV so MLflow can "see" them during the build
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV RUN_ID=${RUN_ID}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# This will now work because MLFLOW_TRACKING_URI is set in the environment
RUN python -c "import mlflow; mlflow.artifacts.download_artifacts(run_id='$RUN_ID', artifact_path='model', dst_path='.')"

COPY . .
CMD ["python", "serve.py"]