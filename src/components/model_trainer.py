import sys
from typing import List, Tuple
import os
import dask.dataframe as dd
import numpy as np
from src.constant.training_pipeline import *

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from src.ml.model.get_best_model import Get_best_model
from src.exception import VisibilityException
from src.logger import logging
from src.utils.main_utils import MainUtils





class VisibilityModel:
    def __init__(self, preprocessing_object: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object

        self.trained_model_object = trained_model_object

    def predict(self, X: dd.DataFrame) -> dd.DataFrame:
        logging.info("Entered predict method of srcTruckModel class")

        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(X)

            logging.info("Used the trained model to get predictions")

            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise VisibilityException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(self, 
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.utils = MainUtils()

    def get_transformed_train_test_datasets(self):
        
        """
            Method Name :   get_transformed_train_test_datasets
            Description :   This method reads the transformed train and test datasets from artifacts and returns those. 

            
            Output      :   train set and test set are returned
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        try:
            train_set = dd.read_csv(self.data_transformation_artifact.transformed_train_file_path)
            test_set = dd.read_csv(self.data_transformation_artifact.transformed_test_file_path)
            return train_set, test_set

        except Exception as e:
            raise VisibilityException(e, sys) from e



    def train_model(self) :

        """
            Method Name :   train_model
            Description :   This method starts the training of the models based on cluster 

            
            Output      :   NA
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            train_set, test_set = self.get_transformed_train_test_datasets()

            list_of_clusters:list = train_set[CLUSTER_LABEL_COLUMN].compute().unique()
            model = []
            get_best_model = Get_best_model()

            for cluster_no in list_of_clusters:
                train_set_cluster = train_set[train_set[CLUSTER_LABEL_COLUMN] == cluster_no]
                test_set_cluster = test_set[test_set[CLUSTER_LABEL_COLUMN] == cluster_no]

                X_train = train_set_cluster.drop(columns = [TARGET_COLUMN, CLUSTER_LABEL_COLUMN]).to_dask_array(lengths=True)
                X_test = test_set_cluster.drop(columns = [TARGET_COLUMN, CLUSTER_LABEL_COLUMN]).to_dask_array(lengths=True)

                y_train = train_set_cluster[[TARGET_COLUMN]].to_dask_array(lengths=True)
                y_test = test_set_cluster[[TARGET_COLUMN]].to_dask_array(lengths=True)
                logging.info(f"Training on cluster_no {cluster_no}")
                
                
                
                best_model_detail = get_best_model.get_best_model(
                                                    x_train=X_train,
                                                    y_train=y_train,
                                                    x_test=X_test,
                                                    y_test=y_test)

                print("trained cluster no %s" %cluster_no)
                model.append(best_model_detail)
           
                preprocessing_obj = self.utils.load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

                if best_model_detail.best_model_score < self.model_trainer_config.expected_accuracy:
                                logging.info("No best model found with score more than base score")
                                logging.info("best model score: %s" % best_model_detail.best_model_score)
                                raise Exception("No best model found with score more than base score")
                
                visibility_model = VisibilityModel(
                    preprocessing_object=preprocessing_obj,
                    trained_model_object=best_model_detail.best_model_object)

                trained_model_path = self.model_trainer_config.trained_model_dir
                os.makedirs(trained_model_path, exist_ok=True)

                model_path = os.path.join(self.model_trainer_config.trained_model_dir, f"{MODEL_FILE_NAME}_{cluster_no}{MODEL_FILE_EXTENSION}" )
                
                self.utils.save_object(
                                    file_path=model_path,
                                    obj=visibility_model)

                logging.info("Cluster no %s trained" %cluster_no)

            logging.info("model training completed.")

  
        except Exception as e:
            raise VisibilityException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
            Method Name :   initiate_model_trainer
            Description :   This method initiates the model training in the training pipeline.
            
            Output      :   model training artifact is returned 
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")
            self.train_model()
          
            model_trainer_artifact = ModelTrainerArtifact(
            trained_model_dir=self.model_trainer_config.trained_model_dir
            )

            logging.info("Model training completed successfully")

            return model_trainer_artifact

            

        except Exception as e:
            raise VisibilityException(e, sys) from e
