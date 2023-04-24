
import sys
from typing import Dict, Tuple
from dask.distributed import LocalCluster, Client
from sklearn.tree import DecisionTreeRegressor
from dataclasses import dataclass

from sklearn.linear_model import  Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from dask.dataframe import DataFrame
from sklearn.model_selection import GridSearchCV
import joblib

from src.constant.training_pipeline import *
from src.exception import VisibilityException
from src.logger import logging

from src.utils.main_utils import MainUtils


@dataclass
class BestModel:
    best_model_name:str
    best_model_object:object
    best_model_score:float


class Get_best_model:  

    def __init__(self):
        self.utils = MainUtils()
        
        cluster = LocalCluster()
        self.client = Client(cluster)



        self.models = {}
        self.models['ridge_regression'] = Ridge()
        self.models['random_forest_regressor'] = RandomForestRegressor()
        self.models['decision_tree_regressor'] = DecisionTreeRegressor()


    
    def read_model_config_file(self) -> dict:
        try:
            model_config = self.utils.read_yaml_file(MODEL_TRAINER_MODEL_CONFIG_FILE_PATH)

            return model_config

        except Exception as e:
            raise VisibilityException(e, sys) from e

    def get_model_best_params(
        self, model: object, x_train, y_train) -> Dict:
        logging.info("Entered the get_model_params method of MainUtils class")

        try:
            model_name = model.__class__.__name__

            model_config = self.read_model_config_file()

            model_list:list = model_config["model_selection"]["model"].keys()
            if model_name in model_list:
                model_param_grid = model_config["model_selection"]["model"][model_name]['search_param_grid']
                with joblib.parallel_backend('dask'):
                    logging.info("fitting cv on %s" % model_name)
               
                    model_grid = GridSearchCV(
                        model, model_param_grid, 
                        cv = 2,
                        n_jobs = -1 ,
                        verbose=3
                    )

                    model_grid.fit(x_train, y_train)

                    return model_grid.best_params_
                
            return None

        except Exception as e:
            raise VisibilityException(e, sys) from e


    def get_grid_search_cv_model_details(
        self,
        train_x: DataFrame,
        train_y: DataFrame,
        test_x: DataFrame,
        test_y: DataFrame,
    ) :

        logging.info("Entered the get_tuned_model method of MainUtils class")

        try:
            model_score_details = []

            for model_name in self.models.keys():
                model = self.models[model_name]
                print("training the %s model" % model_name)
                
                model_best_params = self.get_model_best_params(model, x_train=train_x, y_train=train_y)
                if model_best_params:
                    model_instance = model.set_params(**model_best_params)
                else:
                    model_instance = model
                    
                model_instance.fit(train_x, train_y)
                preds = model_instance.predict(test_x)
                model_score = self.get_model_score(test_y, preds)
                model_detail = {
                    'model_name': model.__class__.__name__,
                    'model_object': model,
                    'score': model_score
                    }

                    
                model_score_details.append(model_detail)
                print("%s model is trained" % model_name)
                    
            return model_score_details

        

        except Exception as e:
            raise VisibilityException(e, sys) from e

    @staticmethod
    def get_model_score(test_y: DataFrame, preds: DataFrame) -> float:
        logging.info("Entered the get_model_score method of MainUtils class")

        try:
            model_score = r2_score(test_y, preds)

            logging.info("Model score is {}".format(model_score))

            logging.info("Exited the get_model_score method of MainUtils class")

            return model_score

        except Exception as e:
            raise VisibilityException(e, sys) from e

  
    def get_best_model_detail(self, model_details:dict) -> dict:
        """
        this methods returns the best model dictionary
        """
        best_model_score = max([model_detail['score'] for model_detail in model_details])
        best_model = [ model_detail  for model_detail in model_details if (model_detail['score'] == best_model_score)]
        return best_model[0]



    def get_best_model(
                    self,
                    x_train,
                    y_train,
                    x_test,
                    y_test) -> BestModel:
        logging.info(
            "Entered the get_best_model_with_name_and_score method of MainUtils class"
        )

        try:
            model_details =  self.get_grid_search_cv_model_details(train_x=x_train,
                                                            train_y=y_train,
                                                            test_x=x_test,
                                                            test_y=y_test)

            best_model_detail = self.get_best_model_detail(model_details=model_details)
            best_model_name = best_model_detail['model_name']
            best_model_object = best_model_detail['model_object']
            best_model_score = best_model_detail['score']

            logging.info(
                "Exited the get_best_model_with_name_and_score method of MainUtils class"
            )

            best_model = BestModel(
                                best_model_name=best_model_name,
                                best_model_object=best_model_object,
                                best_model_score = best_model_score
                                 )
            return best_model
        except Exception as e:
            raise VisibilityException(e, sys) from e