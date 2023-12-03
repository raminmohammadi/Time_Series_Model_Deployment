FROM python:3.9-slim

WORKDIR /

ENV AIP_STORAGE_URI=gs://timeseries-end-to-end-mlops/model
ENV BUCKET_NAME=timeseries-end-to-end-mlops

COPY trainer /trainer

RUN pip install --no-cache-dir pandas google-cloud-storage scikit-learn joblib\
    grpcio gcsfs python-dotenv


ENTRYPOINT ["python", "-m", "trainer.train"]
