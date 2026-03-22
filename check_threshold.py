import mlflow
import os
import sys

# Get the Run ID we saved during training
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0)

print(f"Model ID: {run_id} | Accuracy: {accuracy}")

# THRESHOLD LOGIC
# Set to 0.1 for SUCCESS screenshot
# Set to 1.0 for FAILURE screenshot
threshold = 0.1 

if accuracy >= threshold:
    print(" Threshold passed! Proceeding to Docker build.")
    sys.exit(0)
else:
    print(" Threshold failed! Stopping deployment.")
    sys.exit(1)