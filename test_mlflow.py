import mlflow

# Set the experiment name
mlflow.set_experiment("Assignment3_Rofida")

with mlflow.start_run():
    mlflow.log_param("framework", "tensorflow")
    mlflow.log_metric("accuracy", 0.95)
    print("Run logged! Check the UI at http://localhost:5000")