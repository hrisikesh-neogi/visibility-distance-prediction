from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataClusteringArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from sklearn.metrics import r2_score
from src.exception import VisibilityException
from src.constant.training_pipeline import TARGET_COLUMN,CLUSTER_LABEL_COLUMN
from src.logger import logging
import os
import sys
import dask.dataframe as dd


from src.ml.model.s3_estimator import VisibilityEstimator
from dataclasses import dataclass
from typing import Optional
from src.utils.main_utils import MainUtils


@dataclass
class EvaluateModelResponse:
    is_model_accepted: bool
    trained_model_path:str 
    trained_model_name:str
    changed_accuracy:float

 


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_clustering_artifact: DataClusteringArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_clustering_artifact = data_clustering_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.utils = MainUtils()
        except Exception as e:
            raise VisibilityException(e, sys) from e

    def get_best_model(self,model_name) -> Optional[VisibilityEstimator]:
        
        """
            Method Name :   get_best_model
            Description :   This method returns the best model from s3 bucket based on the cluster number

            
            Output      :   best model object
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        
        try:
            bucket_name = self.model_eval_config.bucket_name
            visibility_estimator = VisibilityEstimator(bucket_name=bucket_name,
                                               model_name=model_name)

            if visibility_estimator.is_model_present(model_name=model_name):
                return visibility_estimator
            return None
        except Exception as e:
            raise VisibilityException(e, sys)

    def get_model_path_from_cluster_no(self,model_dir:str,cluster_no:int ) -> str:
        
        """
            Method Name :   get_model_from_cluster_no
            Description :   This method returns clustering model path based on cluster number.

            
            Output      :   clustering model object
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        
        try:
            if os.path.exists(model_dir):
                list_of_models = os.listdir(model_dir)
                for model in list_of_models:
                    try:
                        model_name = model.split('.')[0]
                        if (model_name.index(str(cluster_no)) == len(model_name)-1):
                            return os.path.join(model_dir,model)

                    except:
                        continue

        except Exception as e:
            raise VisibilityException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:

              
        """
            Method Name :   evaluate_model
            Description :   This method stars the model evaluation on every cluster.

            
            Output      :   model evaluation result
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            test_set = dd.read_csv(self.data_clustering_artifact.clustered_test_data_dir)
            
            list_of_clusters:list = test_set['Cluster'].compute().unique()
            for cluster_no in list_of_clusters:
                test_set_cluster = test_set[test_set['Cluster'] == cluster_no]

                x_test = test_set_cluster.drop(columns = [TARGET_COLUMN,CLUSTER_LABEL_COLUMN])
                y_test = test_set_cluster[[TARGET_COLUMN]]                   

                model_file_path = self.get_model_path_from_cluster_no( model_dir = self.model_trainer_artifact.trained_model_dir,
                                                        cluster_no=cluster_no)

                model_name = os.path.basename(model_file_path)
                trained_model = self.utils.load_object(file_path=model_file_path)
         
                y_hat_trained_model = trained_model.predict(x_test)

                trained_model_r2_score = r2_score(y_test, y_hat_trained_model)
                best_model_r2_score = None
                best_model = self.get_best_model(model_name=model_name)
                if best_model is not None:
                    y_hat_best_model = best_model.predict(x_test)
                    best_model_r2_score = r2_score(y_test, y_hat_best_model)                # calucate how much percentage training model accuracy is increased/decreased
                tmp_best_model_score = 0 if best_model_r2_score is None else best_model_r2_score

                result = EvaluateModelResponse(
                                            is_model_accepted=trained_model_r2_score > tmp_best_model_score,
                                            trained_model_path= model_file_path,
                                            changed_accuracy=trained_model_r2_score - tmp_best_model_score,
                                            trained_model_name = model_name
                                            
                                            )
                
                logging.info(f"model is evaluated on cluster no:{cluster_no} \n evaluation result: {result}")

                yield result

        except Exception as e:
            raise VisibilityException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

              
        """
            Method Name :   initiate_model_evaluation
            Description :   This method initiates the model evaluation component of training pipeline.
            
            Output      :   model evaluation result
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """


        try:
            model_evaluation_responses = self.evaluate_model()
            for evaluation_response in model_evaluation_responses:
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=evaluation_response.is_model_accepted,
                    trained_model_path= evaluation_response.trained_model_path,
                    trained_model_name= evaluation_response.trained_model_name,
                    changed_accuracy= evaluation_response.changed_accuracy
                   
                )

                
                logging.info(f"Model evaluation artifact for {model_evaluation_artifact.trained_model_name}: {model_evaluation_artifact}")
                yield model_evaluation_artifact

        except Exception as e:
            raise VisibilityException(e, sys) from e
