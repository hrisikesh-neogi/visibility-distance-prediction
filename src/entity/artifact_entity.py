from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    raw_batch_files_path: str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    valid_raw_files_dir: str

@dataclass
class DataClusteringArtifact:
    clustered_train_data_dir:str
    clustered_test_data_dir:str
    clustering_model_path: str



@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str



@dataclass
class ModelTrainerArtifact:
    trained_model_dir:str 
    # metric_artifact:ClassificationMetricArtifact

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    trained_model_path:str 
    trained_model_name:str
    changed_accuracy:float

@dataclass
class ModelPusherArtifact:
    bucket_name:str
    is_model_pushed:bool 




    
