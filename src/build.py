from google.cloud import aiplatform
import os, logging
from dotenv import load_dotenv

# Get the current directory
current_directory = os.path.abspath(__file__)

# Navigate to the main folder of the repository
main_folder = os.path.abspath(os.path.join(current_directory, "../../"))
log_file_path = os.path.join(main_folder, 'logging/log_file.log')


def configure_logging():
    # Configure logging to console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %('
                                                   'levelname)s - %(message)s')

    # Create a handler for writing logs to a local file
    file_handler = logging.FileHandler(log_file_path)

    # Set the logging level for the file handler
    file_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the root logger
    logging.getLogger().addHandler(file_handler)

configure_logging()


# load env variables
dotenv_path = os.path.join(main_folder, 'google_cloud.env')
load_dotenv(dotenv_path)

# Configuration parameters
REGION = os.getenv("REGION")
PROJECT_ID = os.getenv("PROJECT_ID")
BASE_OUTPUT_DIR = os.getenv("BASE_OUTPUT_DIR")
BUCKET = os.getenv("AIP_MODEL_DIR")  # Should be same as AIP_STORAGE_URI specified in the
# docker file
CONTAINER_URI = os.getenv("CONTAINER_URI")
MODEL_SERVING_CONTAINER_IMAGE_URI = os.getenv("MODEL_SERVING_CONTAINER_IMAGE_URI")
DISPLAY_NAME = 'timeseries-mlops'
SERVICE_ACCOUNT_EMAIL = os.getenv("SERVICE_ACCOUNT_EMAIL")


from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(os.path.join(main_folder,
                                                                                 'timeseries-end-to-end-406317-08b77e4c7f05.json'))

def initialize_aiplatform(project_id, region, bucket):
    """Initializes the AI platform with the given parameters.
    :param project_id: GCP project ID
    :param region: GCP region
    :param bucket: GCS bucket
    """

    aiplatform.init(project=project_id,
                    location=region,
                    staging_bucket=bucket)


def create_training_job(display_name, container_uri,
                        model_serving_container_image_uri, bucket):
    """Creates a custom container training job.
    :param display_name: Display name of the training job
    :param container_uri: URI of the training container
    :param model_serving_container_image_uri: URI of the model serving container
    :param bucket: GCS bucket

    :return: Custom container training job
    """
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_uri,
        model_serving_container_image_uri=model_serving_container_image_uri,
        staging_bucket=bucket,
    )
    return job


def run_training_job(job, display_name, base_output_dir, service_account_email):
    """Runs the custom container training job.
    :param job: Custom container training job
    :param display_name: Display name of the training job
    :param base_output_dir: Base output directory
    :param service_account_email: Service account email

    :return: Trained model
    """
    model = job.run(
        model_display_name=display_name,
        base_output_dir=base_output_dir,
        service_account=service_account_email
    )
    return model


def deploy_model_to_endpoint(model, display_name, service_account_email):
    """Deploys the trained model to an endpoint.
    :param model: Trained model
    :param display_name: Display name of the endpoint

    :return: Endpoint
    """
    endpoint = model.deploy(
        deployed_model_display_name=display_name,
        sync=True,
        service_account=service_account_email,
    )
    return endpoint

def main():
    # logging.basicConfig(level=logging.INFO)
    try:
        # Initialize AI platform
        initialize_aiplatform(PROJECT_ID, REGION, BUCKET)

        # Create and run the training job
        training_job = create_training_job(DISPLAY_NAME, CONTAINER_URI,
                                           MODEL_SERVING_CONTAINER_IMAGE_URI,
                                           BUCKET)


        model = run_training_job(training_job, DISPLAY_NAME, BASE_OUTPUT_DIR,
                                 SERVICE_ACCOUNT_EMAIL)

        # Deploy the model to the endpoint
        endpoint = deploy_model_to_endpoint(model, DISPLAY_NAME, SERVICE_ACCOUNT_EMAIL)

        # logging.info("Model deployment completed successfully.")
        return endpoint

    except Exception as e:
        # logging.error(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    endpoint = main()