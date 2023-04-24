from fastapi import FastAPI, Request, File, UploadFile
import shutil
import sys
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from uvicorn import run as app_run
from starlette.responses import FileResponse


from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.exception import VisibilityException
from src.logger import logging
from src.logger import LOG_FILE_PATH, LOG_FILE
from src.constant.application import *

import warnings
warnings.filterwarnings('ignore')

app = FastAPI()




origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()
        logging.info("training successfully completed.")

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


        
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        pred_pipeline = PredictionPipeline(file)
        prediction_file = pred_pipeline.run_pipeline()
        file_location = prediction_file.prediction_file_path
        file_name = prediction_file.prediction_file_name
        logging.info("prediction finished")
        return FileResponse(file_location, media_type='application/octet-stream',filename=file_name)

    except Exception as e:
        raise VisibilityException(e,sys)

@app.post("/logs")
def predict():
    try:
        return FileResponse(LOG_FILE_PATH, media_type='application/octet-stream',filename=LOG_FILE)

    except Exception as e:
        raise VisibilityException(e,sys)






if __name__ == "__main__":
    app_run(app, host = APP_HOST, port =APP_PORT)
    
