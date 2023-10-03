FROM jupyter/scipy-notebook

RUN pip install joblib
RUN pip install mlflow 
RUN pip install --upgrade databricks-cli
USER root
RUN apt-get update && apt-get install -y jq

RUN mkdir model raw_data processed_data results


ENV RAW_DATA_DIR=/home/jovyan/raw_data
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV MODEL_DIR=/home/jovyan/model
ENV RESULTS_DIR=/home/jovyan/results
ENV RAW_DATA_FILE=adult.csv
ENV DATABRICKS_HOST =https://community.cloud.databricks.com
ENV DATABRICKS_USERNAME=micolp20022@gmail.com
ENV DATABRICKS_PASSWORD=Khanhvovan2002@

COPY adult.csv ./raw_data/adult.csv
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py
COPY test.py ./test.py
