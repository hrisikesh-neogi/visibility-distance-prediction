import sys
from typing import Union
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.constant import *
from src.exception import VisibilityException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    data_transformation_dir=os.path.join('artifacts','data_transformation')
    transformed_train_file_path=os.path.join(data_transformation_dir, 'train.npy')
    transformed_test_file_path=os.path.join(data_transformation_dir, 'test.npy') 
    transformed_object_file_path=os.path.join( data_transformation_dir, 'preprocessing.pkl' )






class DataTransformation:
    def __init__(self,
                 clustered_train_data_path,
                 clustered_test_data_path):
       
        self.clustered_train_data_path = clustered_train_data_path
        self.clustered_test_data_path = clustered_test_data_path
        self.data_transformation_config = DataTransformationConfig()


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
            train_set = pd.read_csv(self.clustered_train_data_path)
            test_set = pd.read_csv(self.clustered_test_data_path)

            return train_set, test_set

        except Exception as e:
            raise VisibilityException(e,sys)



    def transform_data(self, train_set:pd.DataFrame, test_set:pd.DataFrame) -> pd.DataFrame:
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

    


            X_train_scaled =  preprocessor.fit_transform(X_train)
            X_test_scaled  =  preprocessor.transform(X_test)

            X_train_final = pd.DataFrame(
                X_train_scaled, columns= X_train.columns, index= X_train.index
            )

            X_test_final = pd.DataFrame(
            X_test_scaled, columns= X_test.columns, index= X_test.index
                )

            #merge cluster label columns
            X_train_final[CLUSTER_LABEL_COLUMN] = train_set[CLUSTER_LABEL_COLUMN]
            X_test_final[CLUSTER_LABEL_COLUMN] = test_set[CLUSTER_LABEL_COLUMN]

         


            #save preprocessor
            preprocessor_dump_path = self.data_transformation_config.transformed_object_file_path
            preprocessor_dump_dir = os.path.dirname(preprocessor_dump_path)
            os.makedirs(preprocessor_dump_dir, exist_ok=True)


            self.utils.save_object(preprocessor_dump_path, preprocessor)
            logging.info("saved preprocessor")
        


            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )

            return X_train_final, X_test_final

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

            return (
                X_train, y_train, X_test, y_test,
                self.data_transformation_config.transformed_object_file_path
                
            )

        except Exception as e:
            raise VisibilityException(e, sys) from e
