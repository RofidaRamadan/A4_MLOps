import mlflow
import sys
import os

# Ensure URI is set from GitHub Secret
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0)

    print(f"Model ID: {run_id} | Accuracy: {accuracy:.4f}")

    if accuracy < 0.85:
        print(" FAILED: Accuracy below threshold.")
        sys.exit(1)
    else:
        print(" PASSED: Threshold met.")
        sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)