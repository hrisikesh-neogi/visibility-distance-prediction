import sys
from typing import Union

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from kneed import KneeLocator
import os
import glob

from src.constant import *
from src.constant.training_pipeline import TARGET_COLUMN
from src.utils.main_utils import MainUtils
from src.exception import VisibilityException
from src.logger import logging
from dataclasses import dataclass



TARGET_COLUMN = "VISIBILITY"
CLUSTER_LABEL_COLUMN = "Cluster"


@dataclass
class DataClusteringConfig:
    bucket_name= AWS_S3_BUCKET_NAME
    clustering_model_name='clustering_model.pkl'
    clustering_dir=os.path.join('artifacts','data_clustering')
    clustered_train_data_dir=os.path.join(clustering_dir,'clustered_train.csv')
    clustered_test_data_dir=os.path.join(clustering_dir,'clustered_test.csv')
    trained_clustering_model_file_path=os.path.join(clustering_dir,'clustering_model.pkl')
    train_test_split_ratio=0.2



class DataClustering:
    def __init__(self, 
                 valid_data_dir:str,
                 
                 ):
       
        self.valid_data_dir = valid_data_dir
        self.data_clustering_config = DataClusteringConfig()
        

        self.utils = MainUtils()
    
    def drop_schema_columns(self, dataframe:pd.DataFrame) -> pd.DataFrame:
        """
        Method Name :   drop_schema_columns
        Description :   This method reads the schema.yml file and drops the column in th dataset based on the schema given. 
        
        Output      :   a pd.DataFrame dropping the schema columns
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            _schema_config = self.utils.read_schema_config_file()
            df = dataframe.drop(columns =  _schema_config["drop_columns"])

            return df
        except Exception as e:
            raise VisibilityException(e,sys)

    @staticmethod
    def get_merged_batch_data(raw_data_dir:str) -> pd.DataFrame:
        """
        Method Name :   get_merged_batch_data
        Description :   This method reads all the validated raw data from the raw_data_dir and returns a pandas DataFrame containing the merged data. 
        
        Output      :   a pandas DataFrame containing the merged data 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            raw_files = os.listdir(raw_data_dir)
            csv_data = []
            for filename in raw_files:
                data = pd.read_csv(os.path.join(raw_data_dir, filename))
                csv_data.append(data)

            merged_data = pd.concat(csv_data)

            return merged_data
        except Exception as e:
            raise VisibilityException(e,sys)
        

    def __select_no_of_clusters(self,dataframe:pd.DataFrame):

        """
            Method Name: select_no_of_clusters
            Description: This method decides the optimum number of clusters to the file.

            Output: Number of clusters 
            On Failure: Write an exception log and then raise an exception

            Version: 1.0
            Revisions: None

        """
        
        wcss=[] # initializing an empty list
        try:
            for i in range (1,11):
                kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42) # initializing the KMeans object
                kmeans.fit(dataframe) # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            logging.info('The optimum number of clusters is: '+str(self.kn.knee)+' . Exited the elbow_plot method of the KMeansClustering class')
            return self.kn.knee

        except Exception as e:
            raise VisibilityException(e, sys)


    def __train_clustering_model(self, dataframe:pd.DataFrame) -> object:

        """
            Method Name: __train_clustering_model
            Description: This method trains the clustering model.

            Output: clustering model
            On Failure: Write an exception log and then raise an exception

            Version: 1.0
            Revisions: None

        """
       
        
        try:
            X = dataframe.drop(columns=TARGET_COLUMN)
            number_of_clusters:int = self.__select_no_of_clusters(dataframe=X)
            kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            kmeans.fit(X)

            clustering_dir = os.path.dirname(self.data_clustering_config.trained_clustering_model_file_path)
            os.makedirs(clustering_dir, exist_ok=True)
            self.utils.save_object(file_path=self.data_clustering_config.trained_clustering_model_file_path,
                                    obj=kmeans)
            logging.info("saved clustering model")
        
            return kmeans
        except Exception as e:
           raise VisibilityException(e, sys)
        

    def get_cluster_labeled_datasets(self, dataframe:pd.DataFrame) -> pd.DataFrame:
 
        """
            Method Name: get_cluster_labeled_datasets
            Description: Create a new dataframe consisting of the cluster information in a new column.

            Output: A datframe with cluster column
            On Failure: Write an exception log and then raise an exception

            Version: 1.0
            Revisions: None


        """
        try:

            kmeans_model = self.__train_clustering_model(dataframe=dataframe)
            X = dataframe.drop(columns=TARGET_COLUMN)
            clustered_data = kmeans_model.predict(X)
            dataframe['Cluster'] = clustered_data

            return dataframe
            
        except Exception as e:
            raise VisibilityException(e, sys)

 
    # def push_clustering_model_to_s3(self):

        
    #     """
    #         Method Name: push_clustering_model_to_s3
    #         Description: This methos pushes the local clustering model to s3 bucket. 

    #         Output: NA
    #         On Failure: Write an exception log and then raise an exception

    #         Version: 1.0
    #         Revisions: None

    #     """

    #     try:
    #         self.clustering_model.save_model(from_file=self.data_clustering_config.trained_clustering_model_file_path)
    #         logging.info("clustering model is pushed to s3")


    #     except Exception as e:
    #         raise VisibilityException(e,sys)

    def apply_outliers_capping(self,dataframe:pd.DataFrame):
        """
            Method Name :   apply_outliers_capping
            Description :   This method reduces the outliers
            
            Output      :   a pd.DataFrame
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """


        try:

            outliers_columns = self.utils.read_schema_config_file()['outlier_columns']

            dataframe = dataframe.compute()
            for column in outliers_columns:

                percentile25 = dataframe[column].quantile(0.25)
                percentile75 = dataframe[column].quantile(0.75)
                iqr = percentile75 - percentile25
                upper_limit = percentile75 + 1.5 * iqr
                lower_limit = percentile25 - 1.5 * iqr
                dataframe.loc[(dataframe[column]>upper_limit), column]= upper_limit
                dataframe.loc[(dataframe[column]<lower_limit), column]= lower_limit   
            
            
            dataframe = pd.from_pandas(dataframe, chunksize=len(dataframe))

            return dataframe

        except Exception as e:
            raise VisibilityException(e,sys)

    def split_data_as_train_test_set(self, dataframe:pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:
        """
            Method Name :   split_data_as_train_test_set
            Description :   This method splits the data in two parts; train_set, test_set. 
            
            Output      :   two pd.DataFrame, train_set, test_set
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        try:

            dataframe = self.apply_outliers_capping(dataframe)
            
            train_set, test_set = train_test_split(dataframe, test_size = self.data_clustering_config.train_test_split_ratio,shuffle=False )

            return train_set, test_set

        except Exception as e:
            raise VisibilityException(e,sys)

    def initiate_clustering(self):
        """
            Method Name :   initiate_clustering
            Description :   This method initiates the data clustering components of training pipeline 
            
            Output      :   clustered dataset is returned and saved to the artifacts directory for data transformation.
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        try:
            if self.data_validation_artifact.validation_status:
                dataframe = self.get_merged_batch_data(raw_data_dir=self.data_validation_artifact.valid_raw_files_dir)
                dataframe = self.drop_schema_columns(dataframe=dataframe)
                dataframe = self.get_cluster_labeled_datasets(dataframe=dataframe)
                train_set, test_set = self.split_data_as_train_test_set(dataframe=dataframe)

                train_set.to_csv(self.data_clustering_config.clustered_train_data_dir ,index=False,header=True, single_file= True)
                test_set.to_csv(self.data_clustering_config.clustered_test_data_dir,index=False,header=True, single_file= True)


                # self.push_clustering_model_to_s3()

               

                # data_clustering_artifact = DataClusteringArtifact(
                #     clustered_train_data_dir=self.data_clustering_config.clustered_train_data_dir,
                #     clustered_test_data_dir= self.data_clustering_config.clustered_test_data_dir,
                #     clustering_model_path=self.data_clustering_config.trained_clustering_model_file_path
                # )
                
                # logging.info("clustering is done.")

                # return data_clustering_artifact
            else:
                raise Exception("data is not validated.")
        except Exception as e:
            raise VisibilityException(e,sys)


        
      

    

    





 
    



        
