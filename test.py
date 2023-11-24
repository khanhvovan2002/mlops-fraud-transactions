import pandas as pd
from joblib import load
import json
import os
import numpy as np 
import numpy
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump







# Set path for the input (model)
MODEL_DIR = os.environ["MODEL_DIR"]
model_file = 'logit_model.joblib'
model_path = os.path.join(MODEL_DIR, model_file)

# Set path for the input (test data)
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
test_data_file = 'test.csv'
test_data_path = os.path.join(PROCESSED_DATA_DIR, test_data_file)



# Load model
logit_model = load(model_path)

# Load data
df = pd.read_csv(test_data_path)
X_train = df.drop('fraud', axis=1)
df_cleaned = X_train.dropna()
df_drop = df_cleaned.drop(columns = ['source','target','device','zipcodeOri','zipMerchant'])
category_columns = df_drop.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df_drop, columns=category_columns)
C = 2*np.pi/12
C_ = 2*np.pi/24
# Map month to the unit circle.scscs
df_encoded["month_sin"] = np.sin(df_encoded['month']*C)
df_encoded["month_cos"] = np.cos(df_encoded['month']*C)
df_encoded.timestamp = df_encoded.timestamp.values.astype(np.int64) // 10 ** 6
df_encoded['hour_sin']=np.sin(df_encoded['hour']*C)
df_encoded['hour_cos']=np.cos(df_encoded['hour']*C)
df = df_encoded.drop(columns = ['month', 'hour'])


scaler = StandardScaler()

y_test = df['fraud']
X_test = scaler.fit_transform(df.drop('fraud', axis=1))
# Predict
logit_predictions = logit_model.predict(X_test)

# Compute test accuracy
test_logit = accuracy_score(y_test,logit_predictions)

# Test accuracy to JSON
test_metadata = {
    'test_acc': test_logit
}


# Set output path
RESULTS_DIR = os.environ["RESULTS_DIR"]
test_results_file = 'test_metadata.json'
results_path = os.path.join(RESULTS_DIR, test_results_file)

# Serialize and save metadata
with open(results_path, 'w') as outfile:
    json.dump(test_metadata, outfile)





