import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
# from sklearn.preprocessing import StandardScaler
import numpy as np

# Set path for the input
RAW_DATA_DIR = os.environ["RAW_DATA_DIR_1"]
# RAW_DATA_DIR_2 = os.environ["RAW_DATA_DIR_1"]s

# RAW_DATA_FILE = os.environ["RAW_DATA_FILE"]
# raw_data_path = os.path.join(RAW_DATA_DIR, RAW_DATA_FILE)

# Read dataset
transactions_data = pd.read_csv(RAW_DATA_DIR,  parse_dates=['timestamp'])
# user_events_data = pd.read_csv(RAW_DATA_DIR_2,index_col=0, quotechar="\'", parse_dates=['timestamp'])
#preproccessing
transactions_data['month']= transactions_data.timestamp.dt.month
transactions_data['hour']= transactions_data.timestamp.dt.hour
# fraud_data = transactions_data.loc[transactions_data['fraud'] == 1]
# non_fraud_data = transactions_data.loc[transactions_data['fraud'] == 0]
# df_cleaned = transactions_data.dropna()
# df_drop = df_cleaned.drop(columns = ['source','target','device','zipcodeOri','zipMerchant'])
# category_columns = df_drop.select_dtypes(include=['object']).columns
# df_encoded = pd.get_dummies(df_drop, columns=category_columns)
# import numpy as np
# C = 2*np.pi/12
# C_ = 2*np.pi/24
# # Map month to the unit circle.
# df_encoded["month_sin"] = np.sin(df_encoded['month']*C)
# df_encoded["month_cos"] = np.cos(df_encoded['month']*C)
# df_encoded.timestamp = df_encoded.timestamp.values.astype(np.int64) // 10 ** 6
# df_encoded['hour_sin']=np.sin(df_encoded['hour']*C)
# df_encoded['hour_cos']=np.cos(df_encoded['hour']*C)
# df = df_encoded.drop(columns = ['month', 'hour'])


# Split into train and test
# Standardize the input features
# scaler = StandardScaler()
# X, y = df.drop(columns=['fraud']), df['fraud']

# X = scaler.fit_transform(df)

# Split the dataset into training and testing sets
train, test = train_test_split(transactions_data, test_size=0.3, stratify=df['fraud'])

# Set path to the outputs
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
test_path = os.path.join(PROCESSED_DATA_DIR, 'test.csv')

# Save csv
train.to_csv(train_path, index=False)
test.to_csv(test_path,  index=False)