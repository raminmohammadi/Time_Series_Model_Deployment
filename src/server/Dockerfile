FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY server/predict.py /app/

# Install Flask and google-cloud-storage
RUN pip install Flask google-cloud-storage joblib scikit-learn grpcio gcsfs  \
    python-dotenv

ENV AIP_STORAGE_URI=gs://timeseries-end-to-end-mlops/model
ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080
ENV BUCKET_NAME=timeseries-end-to-end-mlops

ENV PROJECT_ID=timeseries-end-to-end-406317

# Run predict.py when the container launches
ENTRYPOINT ["python", "predict.py"]