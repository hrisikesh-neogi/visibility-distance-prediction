import os
from src.utils.main_utils import MainUtils
from src.constant.prediction_pipeline import *

from src.constant.training_pipeline import *
from pymongo import MongoClient
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    raw_batch_files_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_RAW_BATCH_FILES_STORE_DIR)
    
    
    ingested_data_dir: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR)
    
    
    collection_name:str = DATA_INGESTION_COLLECTION_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    valid_data_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_VALID_DIR)
    invalid_data_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_INVALID_DIR)


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                    DATA_TRANSFORMATION_TRAIN_FILE_NAME)
   
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   DATA_TRANSFORMATION_TEST_FILE_NAME)
    transformed_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     PREPROCSSING_OBJECT_FILE_NAME)


@dataclass
class DataClusteringConfig:
    bucket_name: str = TRAINING_BUCKET_NAME
    s3_model_key_path: str = DATA_CLUSTERING_TRAINED_MODEL_NAME
    clustering_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_CLUSTERING_DIR_NAME)
    clustered_train_data_dir: str = os.path.join(clustering_dir,DATA_CLUSTERING_CLUSTERED_DATA_DIR, DATA_CLUSTERING_CLUSTERED_TRAIN_FILE_NAME)
    clustered_test_data_dir: str = os.path.join(clustering_dir,DATA_CLUSTERING_CLUSTERED_DATA_DIR, DATA_CLUSTERING_CLUSTERED_TEST_FILE_NAME)

    trained_clustering_model_file_path: str = os.path.join(clustering_dir, DATA_CLUSTERING_TRAINED_MODEL_DIR, DATA_CLUSTERING_TRAINED_MODEL_NAME)
    train_test_split_ratio: float = DATA_CLUSTERING_TRAIN_TEST_SPLIT_RATIO


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_dir: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR)
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH



@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_PUSHER_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME


@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_PUSHER_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME





@dataclass
class PredictionPipelineConfig:
    pred_config_file_path = PRED_SCHEMA_FILE_PATH
    clustering_model_name = DATA_CLUSTERING_TRAINED_MODEL_NAME
    model_bucket_name = MODEL_BUCKET_NAME
    prediction_file_name = PREDICTION_OUTPUT_FILE_NAME.split('.')[0] + '_' + TIMESTAMP +'.'+ PREDICTION_OUTPUT_FILE_NAME.split('.')[-1]
    pred_artifact_dir = os.path.join(PREDICTION_ARTIFACT_DIR,TIMESTAMP)
    pred_file_input_dir = os.path.join(pred_artifact_dir, PRED_FILE_INPUR_DIR_NAME)
    pred_file_prediction_output_dir = os.path.join(pred_artifact_dir,PRED_FILE_OUTPUT_DIR_NAME,prediction_file_name)

      



class PCAConfig:
    def __init__(self):
        self.n_components = 2
        self.random_state = 42
        
    def get_pca_config(self):
        return self.__dict__
    
class ClusteringConfig:
    def __init__(self):
        self.n_clusters=3
        self.affinity='euclidean'
        self.linkage='ward'
    
    def get_clustering_config(self):
        return self.__dict__

class SimpleImputerConfig:
    def __init__(self):
        self.strategy = "constant"

        self.fill_value = 0

    def get_simple_imputer_config(self):
        return self.__dict__

# class Prediction_config:
#     def __init__(self):
#         utils = MainUtils()
#         self.prediction_schema = utils.read_yaml_file(PRED_SCHEMA_FILE_PATH)
#     def get_prediction_schema(self):
#         return self.__dict__
    


#         return self.__dict__