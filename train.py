import pandas as pd
import json
import os
from joblib import dump
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import mlflow
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
# Set MLflow tracking URI
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/micolp20022@gmail.com/fraud-model")

# Set path to inputs
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_data_file = 'train.csv'
train_data_path = os.path.join(PROCESSED_DATA_DIR, train_data_file)

# Read data
transactions_data = pd.read_csv(train_data_path)
# Split data into dependent and independent variables
# X_train_ = transactions_data.drop('fraud', axis=1)
# df_cleaned = X_train.dropna()
# df_drop = df_cleaned.drop(columns = ['source','target','device','zipcodeOri','zipMerchant'])
# category_columns = df_drop.select_dtypes(include=['object']).columns
# df_encoded = pd.get_dummies(df_drop, columns=category_columns)
# C = 2*np.pi/12
# C_ = 2*np.pi/24
# # Map month to the unit circle.
# df_encoded["month_sin"] = np.sin(df_encoded['month']*C)
# df_encoded["month_cos"] = np.cos(df_encoded['month']*C)
# df_encoded.timestamp = df_encoded.timestamp.values.astype(np.int64) // 10 ** 6
# df_encoded['hour_sin']=np.sin(df_encoded['hour']*C)
# df_encoded['hour_cos']=np.cos(df_encoded['hour']*C)
# df = df_encoded.drop(columns = ['month', 'hour'])


scaler = StandardScaler()

y_train = transactions_data['fraud']
X_train = scaler.fit_transform(transactions_data.drop('fraud', axis=1))
# Model 
logit_model = LogisticRegression(max_iter=10)
logit_model = logit_model.fit(X_train, y_train)

# Cross validation
cv = StratifiedKFold(n_splits=3) 
val_logit = cross_val_score(logit_model, X_train, y_train, cv=cv).mean()
val_f1 = cross_val_score(logit_model, X_train, y_train, cv=cv, scoring = 'f1').mean()

# Validation accuracy to JSON
train_metadata = {
    'validation_acc': val_logit,
    'validation_f1':val_f1
}

# Start an MLflow run
with mlflow.start_run():

    # Log parameters
    mlflow.log_param("max_iter", 10)

    # Log the model
    # Log the model with a timestamp in the model name
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Generate timestamp
    model_name_with_timestamp = f"logit_model_{timestamp}.joblib"
    mlflow.sklearn.log_model(logit_model, model_name_with_timestamp)
    # mlflow.sklearn.log_model(logit_model, "logit_model")

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

