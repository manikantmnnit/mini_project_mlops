import dagshub

import mlflow
mlflow.set_tracking_uri('https://dagshub.com/manikantmnnit/mini_project_mlops.mlflow') # Set the MLflow tracking URI to the DagsHub project's MLflow server
dagshub.init(repo_owner='manikantmnnit', repo_name='mini_project_mlops', mlflow=True) # Initialize DagsHub to log MLflow runs to the DagsHub project's MLflow server


with mlflow.start_run():
  # dummy parameters and metrics to show how to log them
  mlflow.log_param('parameter name', 'value')    # Log a parameter (key-value pair)
  mlflow.log_metric('metric name', 1)   # Log a metric; metrics can be updated throughout the run