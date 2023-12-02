FROM python:3.9-slim

WORKDIR /

ENV AIP_STORAGE_URI=gs://timeseries-mlops/model
ENV BUCKET_NAME=timeseries-mlops

COPY trainer /trainer

RUN pip install --no-cache-dir pandas google-cloud-storage scikit-learn joblib\
    grpcio gcsfs python-dotenv


ENTRYPOINT ["python", "-m", "trainer.train"]
