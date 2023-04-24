import os
from src.constant.s3_bucket import TRAINING_BUCKET_NAME

PRED_SCHEMA_FILE_PATH = os.path.join('config', 'prediction_schema.yaml')
PREDICTION_ARTIFACT_DIR = "prediction_artifacts"
PRED_FILE_INPUR_DIR_NAME = "prediction_raw_files"
PRED_FILE_OUTPUT_DIR_NAME = "prediction_outputs"
PREDICTION_OUTPUT_FILE_NAME = "visibility_prediction.csv"
MODEL_BUCKET_NAME = TRAINING_BUCKET_NAME