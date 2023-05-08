# raw_data = "D:/personal-repo-projects/visibility/artifacts/data_validation/validated"

# from src.components.data_transformation import DataTransformation

# dt = DataTransformation(raw_data)
# train_arr, test_arr,preprocessor_path = dt.initiate_data_transformation()

# from src.components.model_trainer import ModelTrainer

# mt = ModelTrainer( )
# score = mt.initiate_model_trainer(train_arr, test_arr , preprocessor_path)
# print(score)

from src.pipeline.prediction_pipeline import PredictionPipeline

pred = PredictionPipeline(request=None)
path = pred.download_model()
print(path)