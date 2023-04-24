import sys
from typing import Union
import os
import dask.dataframe as dd
from dask.dataframe import DataFrame
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataClusteringArtifact
from src.constant.training_pipeline import TARGET_COLUMN,CLUSTER_LABEL_COLUMN
from src.exception import VisibilityException
from src.logger import logging
from src.utils.main_utils import MainUtils


class DataTransformation:
    def __init__(self,
                 data_clustering_artifact: DataClusteringArtifact,
                 data_tranasformation_config: DataTransformationConfig):
       
        self.data_clustering_artifact = data_clustering_artifact
        self.data_transformation_config = data_tranasformation_config


        self.utils =  MainUtils()
        
    
    def get_train_and_test_dataset(self):
        
        """
            Method Name :   get_train_and_test_dataset
            Description :   This method reads the clustered train and test datasets from artifacts 
            
            Output      :   a dask DataFrame
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        
        try:
            train_set = dd.read_csv(self.data_clustering_artifact.clustered_train_data_dir)
            test_set = dd.read_csv(self.data_clustering_artifact.clustered_test_data_dir)

            return train_set, test_set

        except Exception as e:
            raise VisibilityException(e,sys)



    

                
   


    def transform_data(self, train_set:dd.DataFrame, test_set:dd.DataFrame) -> DataFrame:
        """
            Method Name :   transform_data
            Description :   This method applies feature transformation and other feature
                            engineering operations and returns train and test X datasets. 
            
            Output      :   preprocessed X train and X test 
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info("Dropping schema columns")
            logging.info("Initialized StandardScaler, SimpleImputer")

            
            
            

            preprocessor = StandardScaler()
            
            X_train = train_set.drop(columns = [TARGET_COLUMN, CLUSTER_LABEL_COLUMN])
            X_test = test_set.drop(columns = [TARGET_COLUMN,CLUSTER_LABEL_COLUMN])

    


            X_train =  preprocessor.fit_transform(X_train)
            X_test  =  preprocessor.transform(X_test)

            #merge cluster label columns
            X_train[CLUSTER_LABEL_COLUMN] = train_set[CLUSTER_LABEL_COLUMN]
            X_test[CLUSTER_LABEL_COLUMN] = test_set[CLUSTER_LABEL_COLUMN]

         


            #save preprocessor
            preprocessor_dump_path = self.data_transformation_config.transformed_object_file_path
            preprocessor_dump_dir = os.path.dirname(preprocessor_dump_path)
            os.makedirs(preprocessor_dump_dir, exist_ok=True)


            self.utils.save_object(preprocessor_dump_path, preprocessor)
            logging.info("saved preprocessor")
        


            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )

            return X_train, X_test

        except Exception as e:
            raise VisibilityException(e, sys) from e

    def initiate_data_transformation(self) :
        """
            Method Name :   initiate_data_transformation
            Description :   This method initiates the data transformation component for the pipeline 
            
            Output      :   data transformation artifact is created and returned 
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        logging.info(
            "Entered initiate_data_transformation method of Data_Transformation class"
        )

        try:
               
            train_set, test_set =  self.get_train_and_test_dataset()
            
            X_train,  X_test  = self.transform_data(train_set, test_set)
            
            
            y_train = train_set[[TARGET_COLUMN]]
            y_test = test_set[[TARGET_COLUMN]]
        
            
            train_set = dd.concat([X_train, y_train], axis=1)
            test_set = dd.concat([X_test, y_test], axis=1)

            train_set.to_csv(self.data_transformation_config.transformed_train_file_path, index=False,header=True, single_file= True)
            logging.info("Transformed train set is saved")
            test_set.to_csv(self.data_transformation_config.transformed_test_file_path, index=False,header=True, single_file= True)
            logging.info("Transformed test set is saved")
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path= self.data_transformation_config.transformed_test_file_path
            )
        
            logging.info("data transformatiuon is done.")
            
            return data_transformation_artifact
        



        except Exception as e:
            raise VisibilityException(e, sys) from e
