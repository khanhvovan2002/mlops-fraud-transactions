FROM jupyter/scipy-notebook
RUN pip install neptune
RUN pip install joblib
RUN pip install mlflow 
RUN pip install xgboost
RUN pip install databricks-cli==0.8.7
RUN echo "[DEFAULT]" >> ~/.databrickscfg 
RUN echo "host= https://community.cloud.databricks.com/">> ~/.databrickscfg 
RUN echo "username=micolp20022@gmail.com">> ~/.databrickscfg
RUN echo "password=Khanhvovan2002@" >> ~/.databrickscfg
RUN cat ~/.databrickscfg
USER root
RUN apt-get update && apt-get install -y jq

RUN mkdir model raw_data processed_data results


ENV RAW_DATA_DIR_1='https://s3.wasabisys.com/iguazio/data/fraud-demo-mlrun-fs-docs/data.csv'
# ENV RAW_DATA_DIR_2='https://s3.wasabisys.com/iguazio/data/fraud-demo-mlrun-fs-docs/events.csv'

ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV MODEL_DIR=/home/jovyan/model
ENV RESULTS_DIR=/home/jovyan/results
# ENV RAW_DATA_FILE=adult.csv


COPY adult.csv ./raw_data/adult.csv
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py
COPY test.py ./test.py
