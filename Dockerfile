# #  Base Image 
# FROM python:3.9-slim

# #  Set the WORKDIR 
# # Ensures paths are relative to /app, solving the "Path Issues"
# WORKDIR /app

# #  Efficient Layering Strategy 
# # Copy ONLY requirements first so Docker can cache the installation
# COPY requirements.txt .


# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# # One-command execution: the definition of MLOps reproducibility.
# CMD ["python", "train_model.py"]

FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Accept the Run ID from the pipeline
ARG RUN_ID
ENV MODEL_RUN_ID=${RUN_ID}

# Mocking the model download
RUN echo "Ready to serve model from MLflow Run: ${MODEL_RUN_ID}" > /app/status.txt


# Copy the model

CMD ["python", "-c", "print('Container running for model: ' + os.environ['MODEL_RUN_ID'])"]