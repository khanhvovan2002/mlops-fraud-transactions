import pandas as pd
import json
import os
from joblib import dump
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import mlflow

# Set MLflow tracking URI
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/micolp20022@gmail.com/fraud-model")

# Set path to inputs
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_data_file = 'train.csv'
train_data_path = os.path.join(PROCESSED_DATA_DIR, train_data_file)

# Read data
df = pd.read_csv(train_data_path)

# Split data into dependent and independent variables
X_train = df.drop('fraud', axis=1)
y_train = df['fraud']

# Model 
logit_model = LogisticRegression(max_iter=10000)
logit_model = logit_model.fit(X_train, y_train)

# Cross validation
cv = StratifiedKFold(n_splits=3) 
val_logit = cross_val_score(logit_model, X_train, y_train, cv=cv).mean()

# Validation accuracy to JSON
train_metadata = {
    'validation_acc': val_logit
}

# Start an MLflow run
with mlflow.start_run():

    # Log parameters
    mlflow.log_param("max_iter", 10000)

    # Log the model
    mlflow.sklearn.log_model(logit_model, "logit_model")

    # Log validation accuracy
    mlflow.log_metric("validation_acc", val_logit)

    # Log metadata
    mlflow.log_params(train_metadata)

# Set path to output (model)
MODEL_DIR = os.environ["MODEL_DIR"]
model_name = 'logit_model.joblib'
model_path = os.path.join(MODEL_DIR, model_name)

# Serialize and save model
dump(logit_model, model_path)

# Set path to output (metadata)
RESULTS_DIR = os.environ["RESULTS_DIR"]
train_results_file = 'train_metadata.json'
results_path = os.path.join(RESULTS_DIR, train_results_file)

# Serialize and save metadata
with open(results_path, 'w') as outfile:
    json.dump(train_metadata, outfile)

print(4)
