#!/bin/bash

#Setting mlflow uri environment variables
export MLFLOW_ARTIFACT_URI=$(python -c "from config import config; print(config.MLFLOW_ARTIFACT_URI)")
export DATASET1_S3LOC="s3://deepfm-model/input_data/ratings.csv"
export DATASET2_S3LOC="s3://deepfm-model/input_data/movies.csv"


#Run the MLflow server
mlflow server \
    --backend-store-uri "$MLFLOW_TRACKING_URI" \
    --default-artifact-root "$MLFLOW_ARTIFACT_URI" \
    --host 0.0.0.0 \
    --port 8080 &

#Wait for server to start
sleep 10

#Run the model training code
python -m train \
        --dataset1-s3loc $DATASET1_S3LOC \
        --dataset2-s3loc $DATASET2_S3LOC \
        --embed-dim 10 \
        --lr .005 \
        --epochs 4

#Run the pytest codes marked as code
PYTHONPATH=$(pwd) pytest -m code

#Run the pytest codes in the data folder
pytest --dataset1-loc=$DATASET1_S3LOC tests/data --verbose --disable-warnings

#Run sample prediction
python -m predict \
        --input "(1,1,\"Adventure|Animation|Children|Comedy|Fantasy\")"



