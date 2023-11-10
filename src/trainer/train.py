from google.cloud import storage
from datetime import datetime
import pytz
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import logging
import gcsfs


# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Navigate to the main folder of the repository
main_folder = os.path.abspath(os.path.join(current_directory, "../../"))

def load_data():
    # Initialize a client using the obtained credentials
    client = storage.Client.from_service_account_json(os.path.join(main_folder, 'timeseries-end-to-end-404019-570bf9fb54eb.json'))
    
    bucket_name = 'timeseries-lab'
    blob_path = 'files/md5/6a/9f3a785adb5187e91a9fc3450e2a74'
    bucket = client.get_bucket(bucket_name)
    # List objects in the bucket
    blobs = bucket.list_blobs()
    for blob in blobs:
        print(blob.name)
        
    # Download the specific file
    blob = bucket.blob(blob_path)
    data = blob.download_as_text()
    df = pd.read_csv(StringIO(data))
    column_names = [
        'Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 
        'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'
    ]

    df.columns = column_names

    return df

def normalize_data(data, stats):
    normalized_data = {}
    for column in data.columns:
        mean = stats["mean"][column]
        std = stats["std"][column]
        
        normalized_data[column] = [(value - mean) / std for value in data[column]]
    
    # Convert normalized_data dictionary back to a DataFrame
    normalized_df = pd.DataFrame(normalized_data, index=data.index)
    return normalized_df

def data_transform(df):
    
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)

    # Splitting the data into training and testing sets (80% training, 20% testing)
    train, test = train_test_split(df, test_size=0.2, shuffle=False)

    # Separating features and target variable
    X_train = train.drop(columns=['CO(GT)'])
    y_train = train['CO(GT)']

    X_test = test.drop(columns=['CO(GT)'])
    y_test = test['CO(GT)']

    # Get the json from GCS
    client = storage.Client()
    bucket_name = 'mlops-data-ie7374' # Change this to your bucket name
    blob_path = 'scaler/normalization_stats.json' # Change this to your blob path where the data is stored
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Download the json as a string
    data = blob.download_as_string()
    stats = json.loads(data)

    # Normalize the data using the statistics from the training set
    X_train_scaled = normalize_data(X_train, stats)
    y_train_scaled = (y_train - stats["mean"]['CO(GT)']) / stats["std"]['CO(GT)']
    
    return X_train_scaled, X_test, y_train_scaled, y_test



if __name__ == '__main__':
    load_data()
    
