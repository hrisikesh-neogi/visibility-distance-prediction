import sys
import os

from src.cloud_storage.aws_storage import SimpleStorageService
from src.entity.artifact_entity import (ModelPusherArtifact,
                                           ModelEvaluationArtifact)
from src.entity.config_entity import ModelPusherConfig
from src.exception import VisibilityException
from src.logger import logging
from src.ml.model.s3_estimator import VisibilityEstimator


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.visibility_estimator = VisibilityEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_name = os.path.basename(self.model_evaluation_artifact.trained_model_path)
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        
        """
            Method Name :   initiate_model_pusher
            Description :   This method initiates the model pusher component of the training pipeline.

            
            Output      :   best model object
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            logging.info("Uploading artifacts folder to s3 bucket")
            self.visibility_estimator.save_model(
                from_file=self.model_evaluation_artifact.trained_model_path
            )
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                is_model_pushed= True 
            )

            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            return model_pusher_artifact
        except Exception as e:
            raise VisibilityException(e, sys) from e
