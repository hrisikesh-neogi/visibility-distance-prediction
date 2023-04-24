import shutil
import sys
from typing import Dict, Tuple
import os
import dask.array as da
import pandas as pd
import pickle
from sklearn import linear_model
import yaml

from pandas import DataFrame
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import all_estimators
from yaml import safe_dump

from src.constant.training_pipeline import *
from src.exception import VisibilityException
from src.logger import logging



    
def load_numpy_array_data(file_path: str) -> da.array:
    """
    load numpy stacked array data from files
    file_path: str location of folder to load
    return: da.array data loaded
    """
    try:
        return da.from_npy_stack(file_path)
    except Exception as e:
        raise VisibilityException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise VisibilityException(e, sys)
class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise VisibilityException(e, sys) from e

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))

            return schema_config

        except Exception as e:
            raise VisibilityException(e, sys) from e

    

    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        logging.info("Entered the save_object method of MainUtils class")

        try:
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

            logging.info("Exited the save_object method of MainUtils class")

        except Exception as e:
            raise VisibilityException(e, sys) from e

    

    @staticmethod
    def load_object(file_path: str) -> object:
        logging.info("Entered the load_object method of MainUtils class")

        try:
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)

            logging.info("Exited the load_object method of MainUtils class")

            return obj

        except Exception as e:
            raise VisibilityException(e, sys) from e

    @staticmethod
    def create_artifacts_zip(file_name: str, folder_name: str) -> None:
        logging.info("Entered the create_artifacts_zip method of MainUtils class")

        try:
            shutil.make_archive(file_name, "zip", folder_name)

            logging.info("Exited the create_artifacts_zip method of MainUtils class")

        except Exception as e:
            raise VisibilityException(e, sys) from e

    @staticmethod
    def unzip_file(filename: str, folder_name: str) -> None:
        logging.info("Entered the unzip_file method of MainUtils class")

        try:
            shutil.unpack_archive(filename, folder_name)

            logging.info("Exited the unzip_file method of MainUtils class")

        except Exception as e:
            raise VisibilityException(e, sys) from e

    def update_model_score(self, best_model_score: float) -> None:
        logging.info("Entered the update_model_score method of MainUtils class")

        try:
            model_config = self.read_model_config_file()

            model_config["base_model_score"] = str(best_model_score)

            with open(MODEL_TRAINER_MODEL_CONFIG_FILE_PATH, "w+") as fp:
                safe_dump(model_config, fp, sort_keys=False)

            logging.info("Exited the update_model_score method of MainUtils class")

        except Exception as e:
            raise VisibilityException(e, sys) from e
        
    def save_numpy_array_data(self,file_path: str, array: da.array):
        """
        Save numpy array data to file
        file_path: str location of file to save
        array: np.array data to save
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            da.to_npy_stack(file_path, array)
        except Exception as e:
            raise VisibilityException(e, sys) from e


