import datetime as dt
from airflow import DAG
from airflow.operators.bash_operator import BashOperator

LOCAL_PREPROCESS_FILE_PATH = '/tmp/preprocess.py'
GITHUB_PREPROCESS_RAW_URL = 'https://raw.githubusercontent.com/raminmohammadi/Time_Series_Model_Deployment/main/src/data_preprocess.py'

LOCAL_TRAIN_FILE_PATH = '/tmp/train_TA.py'
GITHUB_TRAIN_RAW_URL = 'https://raw.githubusercontent.com/raminmohammadi/Time_Series_Model_Deployment/main/src/trainer/train.py'

default_args = {
    'owner': 'Time_Series_IE7374',
    'start_date': dt.datetime(2023, 10, 24),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=5),
}

dag = DAG(
    'model_retraining',
    default_args=default_args,
    description='Model retraining at 9 PM everyday',
    schedule_interval='0 21 * * *',  # Every day at 9 pm
    catchup=False,
)

# Tasks for pulling scripts from GitHub
pull_preprocess_script = BashOperator(
    task_id='pull_preprocess_script',
    bash_command=f'curl -o {LOCAL_PREPROCESS_FILE_PATH} {GITHUB_PREPROCESS_RAW_URL}',
    dag=dag,
)

pull_train_script = BashOperator(
    task_id='pull_train_script',
    bash_command=f'curl -o {LOCAL_TRAIN_FILE_PATH} {GITHUB_TRAIN_RAW_URL}',
    dag=dag,
)

env = {
    'AIP_STORAGE_URI': 'gs://timeseries-end-to-end-mlops/model',
}

# Tasks for running scripts
run_preprocess_script = BashOperator(
    task_id='run_preprocess_script',
    bash_command=f'python {LOCAL_PREPROCESS_FILE_PATH}',
    env=env,
    dag=dag,
)

run_train_script = BashOperator(
    task_id='run_train_script',
    bash_command=f'python {LOCAL_TRAIN_FILE_PATH}',
    env=env,
    dag=dag,
)

# Setting up dependencies
pull_preprocess_script >> pull_train_script >> run_preprocess_script >> run_train_script
