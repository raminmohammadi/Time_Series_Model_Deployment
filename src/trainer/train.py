import logging
from google.cloud import storage
from datetime import datetime
import pytz
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import os
import gcsfs
from dotenv import load_dotenv

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Navigate to the main folder of the repository
main_folder = os.path.abspath(os.path.join(current_directory, "../../"))

# load env variables
dotenv_path = os.path.join(main_folder, 'google_cloud.env')
load_dotenv(dotenv_path)

fs = gcsfs.GCSFileSystem()
storage_client = storage.Client()
bucket_name = os.getenv("BUCKET_NAME")
MODEL_DIR = os.environ['AIP_STORAGE_URI']

# Configure logging
gs_log_file_path = 'gs://timeseries-end-to-end-mlops/logging/log_file.log'

def configure_logging():
    # Configure logging to console and a custom StringIO handler
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log_stream = StringIO()
    file_handler = logging.StreamHandler(log_stream)
    logging.getLogger().addHandler(file_handler)

configure_logging()

def store_model_in_gcs(gcs_model_path, model_version, model):
    """
    Store a machine learning model in Google Cloud Storage.

    Args:
    blob_path (str): The path to the blob where you want to store the model.
    model_version (str): The version of the model.
    model: The machine learning model to be stored.

    Returns:
    None
    """
    try:
        # Save the model to a binary file using joblib
        local_model_file_path = "model.pkl"
        joblib.dump(model, local_model_file_path)

        # Use gcsfs to open a GCS file for writing
        with fs.open(gcs_model_path, 'wb') as f:
            joblib.dump(model, f)
        logging.info(f"Model successfully uploaded to {MODEL_DIR}")
    except Exception as e:
        logging.error(f"Error uploading model: {e}")
        raise
    finally:
        # Clean up the temporary model file
        if os.path.exists(local_model_file_path):
            os.remove(local_model_file_path)

def upload_log_to_gcs(log_messages, gcs_log_path):
    """
    Upload log messages to Google Cloud Storage.

    Args:
    log_messages (str): The log messages to be uploaded.
    gcs_log_path (str): The GCS path to upload the log messages.

    Returns:
    None
    """
    try:
        with fs.open(gcs_log_path, 'wb') as f:
            f.write(log_messages.encode())
        logging.info(f"Log messages successfully uploaded to {gcs_log_path}")
    except Exception as e:
        logging.error(f"Error uploading log messages to GCS: {e}")
        raise

def store_data_in_gcs(blob_path, data):
    """
    Store data in Google Cloud Storage.

    Args:
    blob_path (str): The path to the blob where you want to store the data.
    data (str): The data to be stored.

    Returns:
    None
    """
    try:
        # Initialize a client using the obtained credentials
        client = storage.Client()
        # Convert the dictionary to a JSON-formatted string
        data_json = json.dumps(data)
        # Get the bucket
        bucket = client.get_bucket(bucket_name)

        # Create a new blob and upload the data
        blob = bucket.blob(blob_path)
        blob.upload_from_string(data_json)

        logging.info(f"Data successfully uploaded to {blob_path}")
    except Exception as e:
        logging.error(f"Error uploading data: {e}")
        raise


def model_to_json(model):
    """
    Convert a scikit-learn model to a JSON-formatted string.

    Args:
    model: The machine learning model.

    Returns:
    str: The JSON-formatted string.
    """
    # Customize this function based on your model serialization needs
    # For demonstration purposes, converting the model to a dictionary
    model_dict = {"model_type": type(model).__name__, "parameters": model.get_params()}
    return json.dumps(model_dict)

def load_data():
    """
    Load time series data from Google Cloud Storage bucket.

    Returns:
    pandas.DataFrame: Loaded time series data.
    """
    try:
        # Initialize a client using the obtained credentials
        client = storage.Client()

        bucket = client.get_bucket(bucket_name)

        # List objects in the bucket
        blobs = bucket.list_blobs()
        train_blobs = [blob.name for blob in blobs if 'train' in blob.name and '.csv' in blob.name]
        # logging.info(f"List of training blobs: {train_blobs}")

        # Download the specific file
        blobs_data = [bucket.blob(blob_path).download_as_text() for blob_path in train_blobs]
        data = [pd.read_csv(StringIO(data)) for data in blobs_data]
        df = pd.concat(data)

        column_names = [
            'Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
            'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'
        ]

        df.columns = column_names

        logging.info("Time series data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise



def calculate_stats(df):
    """
    Calculate mean and standard deviation for each column in a DataFrame.

    Args:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    dict: A dictionary containing mean and standard deviation for each column.
    """
    try:
        # Calculate mean and standard deviation
        stats = {
            "mean": df.mean().to_dict(),
            "std": df.std().to_dict()
        }

        logging.info("Statistics calculated successfully.")
        return stats
    except Exception as e:
        logging.error(f"Error calculating statistics: {e}")
        raise

def normalize_data(data, stats):
    """
    Normalize the given data using provided statistics.

    Args:
    data (pandas.DataFrame): The data to be normalized.
    stats (dict): A dictionary containing mean and standard deviation for each column.

    Returns:
    pandas.DataFrame: Normalized data.
    """
    try:
        normalized_data = {}
        for column in data.columns:
            mean = stats["mean"][column]
            std = stats["std"][column]

            normalized_data[column] = [(value - mean) / std for value in data[column]]

        # Convert normalized_data dictionary back to a DataFrame
        normalized_df = pd.DataFrame(normalized_data, index=data.index)

        logging.info("Data normalized successfully.")
        return normalized_df
    except Exception as e:
        logging.error(f"Error normalizing data: {e}")
        raise

def data_transform(df):
    """
    Transform time series data for training and testing.

    Args:
    df (pandas.DataFrame): The input DataFrame containing time series data.

    Returns:
    tuple: A tuple containing the following transformed data:
        - X_train_scaled (pandas.DataFrame): Scaled features for the training set.
        - X_test (pandas.DataFrame): Features for the testing set.
        - y_train_scaled (pandas.Series): Scaled target variable for the training set.
        - y_test (pandas.Series): Target variable for the testing set.
    """
    try:
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

        try:
            # Get the json from GCS
            client = storage.Client()
            bucket_name = os.getenv("BUCKET_NAME")
            blob_path = 'scaler/normalization_stats.json'  # Change this to your blob path where the data is stored
            bucket = client.get_bucket("timeseries-end-to-end-mlops")
            blob = bucket.blob(blob_path)
            # Download the json as a string
            data = blob.download_as_string()
            stats = json.loads(data)
        except ValueError as e:
            logging.warning(f"Error: {e}")
            blob_path = 'scaler/normalization_stats.json'  # Change this to your blob path where the data is stored
            stats = calculate_stats(X_train)
            store_data_in_gcs(blob_path, stats)

        # Normalize the data using the statistics from the training set
        X_train_scaled = normalize_data(X_train, stats)

        logging.info("Data transformation completed successfully.")
        return X_train_scaled, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in data transformation: {e}")
        raise

def train_model(X_train, y_train):
    """
    Train a random forest regression model.

    Args:
    X_train (pandas.DataFrame): Features for the training set.
    y_train (pandas.Series): Target variable for the training set.

    Returns:
    RandomForestRegressor: Trained random forest regression model.
    """
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        logging.info("Model training completed successfully.")
        return model
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise

if __name__ == '__main__':
    try:
        # Load and transform data
        gcs_train_data_path = "gs://timeseries-end-to-end-mlops/data/train/train_data.csv"
        data = load_data()
        X_train, X_test, y_train, y_test = data_transform(data)
        model = train_model(X_train, y_train)

        edt = pytz.timezone('US/Eastern')
        # Get the current time in EDT
        current_time_edt = datetime.now(edt)
        version = current_time_edt.strftime('%d-%m-%Y-%H%M%S')

        gcs_model_path = f"{MODEL_DIR}/model_{version}.pkl"
        print(gcs_model_path)

        store_model_in_gcs(gcs_model_path, version, model)
        logging.info("Script execution completed successfully.")
    except Exception as e:
        logging.error(f"Error during script execution: {e}")
        raise

    # Capture all log messages
    log_messages = logging.getLogger().handlers[1].stream.getvalue()

    # Upload all log messages to GCS
    upload_log_to_gcs(log_messages, gs_log_file_path)
