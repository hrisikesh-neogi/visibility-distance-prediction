# pipeline name and root directory constant
import os
from src.constant.s3_bucket import TRAINING_BUCKET_NAME



PIPELINE_NAME: str = "src"
ARTIFACT_DIR: str = "training_artifacts"
LOG_DIR = "logs"
LOG_FILE = "visibility.log"

# common file name

FILE_NAME: str = "visibility.csv"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = ""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_RAW_BATCH_FILES_STORE_DIR: str = "raw_batch_files"
DATA_INGESTION_INGESTED_DIR: str = "ingested"

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
DATA_TRANSFORMATION_TRAIN_FILE_NAME: str = "train.csv"
DATA_TRANSFORMATION_TEST_FILE_NAME: str = "test.csv"


"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
DATA_CLUSTERING_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_CLUSTERING_DIR_NAME: str = "data_clustering"
DATA_CLUSTERING_CLUSTERED_DATA_DIR:str = "clustered_data"
DATA_CLUSTERING_CLUSTERED_TRAIN_FILE_NAME:str = "clustered_train.csv"
DATA_CLUSTERING_CLUSTERED_TEST_FILE_NAME:str = "clustered_test.csv"

DATA_CLUSTERING_TRAINED_MODEL_DIR: str = "trained_clustering_model"
DATA_CLUSTERING_TRAINED_MODEL_NAME: str = "clustering_model.pkl"


"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.45
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
"""
MODEL Evauation related constant start with MODEL_EVALUATION var name
"""

MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_PUSHER_BUCKET_NAME = TRAINING_BUCKET_NAME

