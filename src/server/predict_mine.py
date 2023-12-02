from flask import Flask, jsonify, request
from google.cloud import storage
import joblib
import os
import json
import logging
from dotenv import load_dotenv

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Navigate to the main folder of the repository
main_folder = os.path.abspath(os.path.join(current_directory, "../../"))

# load env variables
dotenv_path = os.path.join(main_folder, 'google_cloud.env')
load_dotenv(dotenv_path)


# Configure logging
# gs_log_file_path = 'gs://timeseries-mlops/logging/log_file.log'

# log_file_path = os.path.join(main_folder, 'logging/log_file.log')

#
# def configure_logging():
#     # Configure logging to console
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
#     # Create a handler for writing logs to a local file
#     file_handler = logging.FileHandler(log_file_path)
#
#     # Set the logging level for the file handler
#     file_handler.setLevel(logging.INFO)
#
#     # Create a formatter and add it to the file handler
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)
#
#     # Add the file handler to the root logger
#     logging.getLogger().addHandler(file_handler)
#
# configure_logging()

app = Flask(__name__)

def initialize_variables():
    """
    Initialize environment variables.
    Returns:
        tuple: The project id and bucket name.
    """
    # Initialize environment variables.
    # logging.info("Initializing environment variables.")
    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET_NAME")
    return project_id, bucket_name

def initialize_client_and_bucket(bucket_name):
    """
    Initialize a storage client and get a bucket object.
    Args:
        bucket_name (str): The name of the bucket.
    Returns:
        tuple: The storage client and bucket object.
    """
    # logging.info("Initializing storage client and bucket.")
    # storage_client = client = storage.Client.from_service_account_json(os.path.join(main_folder,
    #                                                                    'timeseries-end-to-end-406818-f9feca5f6f00.json'))
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    return storage_client, bucket

def load_stats(bucket, SCALER_BLOB_NAME='scaler/normalization_stats.json'):
    """
    Load normalization stats from a blob in the bucket.
    Args:
        bucket (Bucket): The bucket object.
        SCALER_BLOB_NAME (str): The name of the blob containing the stats.
    Returns:
        dict: The loaded stats.
    """
    # logging.info(f"Loading normalization stats from blob: {SCALER_BLOB_NAME}")
    scaler_blob = bucket.blob(SCALER_BLOB_NAME)
    stats_str = scaler_blob.download_as_text()
    stats = json.loads(stats_str)
    return stats

def load_model(bucket, bucket_name):
    """
    Fetch and load the latest model from the bucket.
    Args:
        bucket (Bucket): The bucket object.
        bucket_name (str): The name of the bucket.
    Returns:
        _BaseEstimator: The loaded model.
    """
    # logging.info("Fetching and loading the latest model.")
    latest_model_blob_name = fetch_latest_model(bucket_name)
    local_model_file_name = os.path.basename(latest_model_blob_name)
    model_blob = bucket.blob(latest_model_blob_name)
    model_blob.download_to_filename(local_model_file_name)
    model = joblib.load(local_model_file_name)
    return model

def fetch_latest_model(bucket_name, prefix="models/model_"):
    """Fetches the latest model file from the specified GCS bucket.
    Args:
        bucket_name (str): The name of the GCS bucket.
        prefix (str): The prefix of the model files in the bucket.
    Returns:
        str: The name of the latest model file.
    """
    # logging.info(f"Fetching the latest model from bucket: {bucket_name}")
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    blob_names = [blob.name for blob in blobs]
    if not blob_names:
        # logging.error("Error: No model files found in the GCS bucket.")
        raise ValueError("No model files found in the GCS bucket.")

    latest_blob_name = sorted(blob_names, key=lambda x: x.split('_')[-1],
                              reverse=True)[0]
    return latest_blob_name

def normalize_data(instance, stats):
    """
    Normalizes a data instance using provided statistics.
    Args:
        instance (dict): A dictionary representing the data instance.
        stats (dict): A dictionary with 'mean' and 'std' keys for normalization.
    Returns:
        dict: A dictionary representing the normalized instance.
    """
    logging.info("Normalizing data instance.")
    normalized_instance = {}
    for feature, value in instance.items():
        mean = stats["mean"].get(feature, 0)
        std = stats["std"].get(feature, 1)
        normalized_instance[feature] = (value - mean) / std
    return normalized_instance

@app.route('/health', methods=['GET'])
def health_check():
    # logging.info("Health check endpoint called.")
    return {"status": "healthy"}

@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict():
    """
    Prediction route that normalizes input data, and returns model predictions.
    Returns:
        Response: A Flask response containing JSON-formatted predictions.
    """
    # logging.info("Prediction route called.")
    request_json = request.get_json()
    request_instances = request_json['instances']

    formatted_instances = []
    for instance in request_instances:
        normalized_instance = normalize_data(instance, stats)
        formatted_instance = [
            normalized_instance['PT08.S1(CO)'],
            normalized_instance['NMHC(GT)'],
            normalized_instance['C6H6(GT)'],
            normalized_instance['PT08.S2(NMHC)'],
            normalized_instance['NOx(GT)'],
            normalized_instance['PT08.S3(NOx)'],
            normalized_instance['NO2(GT)'],
            normalized_instance['PT08.S4(NO2)'],
            normalized_instance['PT08.S5(O3)'],
            normalized_instance['T'],
            normalized_instance['RH'],
            normalized_instance['AH']
        ]
        formatted_instances.append(formatted_instance)

    prediction = model.predict(formatted_instances)
    prediction = prediction.tolist()
    output = {'predictions': [{'result': pred} for pred in prediction]}
    # logging.info(f"Predictions: {output}")
    return jsonify(output)

project_id, bucket_name = initialize_variables()
storage_client, bucket = initialize_client_and_bucket(bucket_name)
stats = load_stats(bucket)
model = load_model(bucket, bucket_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8089)))
