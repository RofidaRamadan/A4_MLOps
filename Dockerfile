FROM python:3.10-slim

WORKDIR /app

# Accept the Run ID from GitHub Actions
ARG RUN_ID
ENV MODEL_ID=$RUN_ID

# Simulate downloading the specific model version
RUN echo "Downloading model weights for Run ID: $MODEL_ID" > /app/log.txt

# Just to show it worked in the logs
RUN cat /app/log.txt

CMD ["python", "-c", "print('Container started for Model ID: ' + os.environ['MODEL_ID'])"]