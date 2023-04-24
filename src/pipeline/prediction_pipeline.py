import shutil
from src.ml.model.s3_estimator import VisibilityEstimator
from src.ml.model.s3_clustering_model import ClusteringModel
from src.constant.training_pipeline import *

from src.entity.config_entity import PredictionPipelineConfig
from src.logger import logging
from src.utils.main_utils import MainUtils

from src.exception import VisibilityException

import dask.dataframe as dd
import dask.array as da
import sys
from fastapi import File
from dataclasses import dataclass
        
        
@dataclass
class PredictionFileDetail:
    prediction_file_path:str
    prediction_file_name:str


class PredictionPipeline:
    def __init__(self, file:File):

        self.file = file 
        self.utils = MainUtils()
        self.prediction_config = PredictionPipelineConfig()

    def save_input_files(self):

        """
            Method Name :   save_input_files
            Description :   This method saves the input file to the prediction artifacts directory. 
            
            Output      :   input dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            pred_file_input_dir = self.prediction_config.pred_file_input_dir
            os.makedirs(pred_file_input_dir, exist_ok=True)

            pred_file_path = os.path.join(pred_file_input_dir, self.file.filename)
            
            with open(pred_file_path, 'wb') as f:
                shutil.copyfileobj(self.file.file, f)
            
            logging.info("Prediction input file saved")


            return pred_file_path
        except Exception as e:
            raise VisibilityException(e,sys)
    
    def read_prediction_schema_file(self):

        """
            Method Name :   read_prediction_schema_file
            Description :   This method reads prediction schema file and returns that.
            
            Output      :   input dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            return self.utils.read_yaml_file(self.prediction_config.pred_config_file_path)
        except Exception as e:
            raise VisibilityException(e,sys)

    def read_pred_input_file(self, input_file_path:str) -> dd.DataFrame:
        
        """
            Method Name :   read_pred_input_file
            Description :   This method reads the input file and returns that as a dataframe
            
            Output      :   input dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        try:
            
            column_dtypes = self.read_prediction_schema_file()['columns']
            dtypes = {}
            for dtype in column_dtypes:
                dtypes.update(dtype)
            return dd.read_csv(input_file_path, dtype = dtypes)
        except Exception as e:
            raise VisibilityException(e,sys)

        
    def validate_schema_columns(self,dataframe:dd.DataFrame) -> dd.DataFrame:

        """
            Method Name :   validate_schema_columns
            Description :   This method validates the input dataframe column length with the schema
            
            Output      :   True or False based on the validation
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        try:
            
            prediction_schema = self.read_prediction_schema_file()
            schema_columns = prediction_schema['columns']
            schema_validation_status =  len(dataframe.columns) ==  len(schema_columns) 

            logging.info(f"Prediction input file validated:{schema_validation_status}")

            return schema_validation_status


            
        except Exception as e:
            raise VisibilityException(e,sys)

    def drop_schema_columns(self,dataframe:dd.DataFrame) -> dd.DataFrame:
        """
            Method Name :   drop_schema_columns
            Description :   this method drops the schema columns from the input dataframe. 

            
            Output      :   latest clustering model object
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        
        try:
            prediction_schema = self.read_prediction_schema_file()
            columns_to_drop = prediction_schema['drop_columns']

            return dataframe.drop(columns = columns_to_drop)

            
        except Exception as e:
            raise VisibilityException(e,sys)
        
    def get_latest_clustering_model(self):
        
        """
            Method Name :   get_latest_clustering_model
            Description :   this method gets the latest clustering model from s3 and returns that

            
            Output      :   latest clustering model object
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        
        try:
            clustering_model = ClusteringModel(
                bucket_name=self.prediction_config.model_bucket_name,
                model_name=self.prediction_config.clustering_model_name)

            clustering_model_obj = clustering_model.load_model()
            print("got s3 model from s3")

            return clustering_model_obj
        

        except Exception as e:
            raise VisibilityException(e,sys)


    def get_clustered_data(self, dataframe:dd.DataFrame):

        """
            Method Name :   get_clustered_data
            Description :   this method get cluster labels and merge them with the dataframe in the Cluster column.

            
            Output      :   a dataframe with cluster labeled column
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        
        try:
            df  = dataframe.copy()
            clustering_model = self.get_latest_clustering_model()
            cluster_labels = clustering_model.predict(df)

            df[CLUSTER_LABEL_COLUMN] = cluster_labels
            print("clustering is done")

            logging.info("Prediction input data is clustered.")

            return df
        except Exception as e:
            raise VisibilityException(e,sys)


    def get_model_name_from_cluster(self,cluster_number:int):
        
        """
            Method Name :   get_model_name_from_cluster
            Description :   this method returns the model from s3 based on the cluster number. 

            
            Output      :   predicted dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        try:
            model_name = MODEL_FILE_NAME+'_'+str(cluster_number)+MODEL_FILE_EXTENSION
            return model_name

        except Exception as e:
            raise VisibilityException(e,sys)
        
    def get_predicted_dataframe(self, dataframe:dd.DataFrame):

        """
            Method Name :   get_predicted_dataframe
            Description :   this method returns the dataframw with a new column containing predictions

            
            Output      :   predicted dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
   
        try:
            prediction_config = PredictionPipelineConfig()

            list_of_clusters:list = dataframe[CLUSTER_LABEL_COLUMN].compute().unique()
            predicted_dataframe = None
            for cluster_no in list_of_clusters:
                clustered_dataframe = dataframe[dataframe[CLUSTER_LABEL_COLUMN]==cluster_no]
                clustered_dataframe = clustered_dataframe.drop(columns = CLUSTER_LABEL_COLUMN)
                clustered_array= clustered_dataframe.to_dask_array(lengths=True)

                model_name = self.get_model_name_from_cluster(cluster_number=cluster_no)
                
                model = VisibilityEstimator(
                    bucket_name= prediction_config.model_bucket_name,
                    model_name= model_name
                )
                logging.info(f"prediction model:{model_name} is loaded from s3.")
                
                predictions = model.predict(clustered_array)
                predictions = da.from_array(predictions)


                clustered_dataframe[TARGET_COLUMN] = predictions


                if predicted_dataframe is not None:
                    predicted_dataframe = dd.concat([predicted_dataframe, clustered_dataframe], axis = 0)
                else:
                    predicted_dataframe = clustered_dataframe
                logging.info(f"prediction is completed for cluster no:{ cluster_no}")
            

            return predicted_dataframe


        except Exception as e:
            raise VisibilityException(e, sys) from e
        
    def run_pipeline(self) ->PredictionFileDetail:
        
        """
            Method Name :   run_pipeline
            Description :   this method initiates the prediction pipeline components. 

            
            Output      :   predicted dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
   
        try:
            logging.info("initiated the prediction pipeline.")

            input_dataframe_path =  self.save_input_files() 
            input_dataframe = self.read_pred_input_file(input_file_path=input_dataframe_path)

            if self.validate_schema_columns(input_dataframe):
                input_dataframe = self.drop_schema_columns(input_dataframe)
              
                clustered_dataframe = self.get_clustered_data(dataframe= input_dataframe)
                predicted_dataframe = self.get_predicted_dataframe(clustered_dataframe)

                predicted_dataframe.to_csv(self.prediction_config.pred_file_prediction_output_dir, index=False,single_file=True)

                logging.info("prediction is completed. \nExited the prediction pipeline.")
                return PredictionFileDetail(
                    prediction_file_path=self.prediction_config.pred_file_prediction_output_dir,
                    prediction_file_name=self.prediction_config.prediction_file_name
                )

        except Exception as e:
            raise VisibilityException(e, sys)
            
            
        
            
        

 
        

        