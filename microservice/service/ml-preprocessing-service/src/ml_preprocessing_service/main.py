from typing import Union, List

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.responses import FileResponse
from ml_preprocessing_lib.base.feature_engineering import \
    create_preprocessing_pipeline
import joblib
import os
import shutil

app = FastAPI()


class TransfomerParameters(BaseModel):
    key: str
    value: str


class TransformerStep(BaseModel):
    step: str
    column: str
    params: Union[List[TransfomerParameters], None] = None


class PreprocessingSteps(BaseModel):
    steps: Union[List[TransformerStep], None] = None


def delete_directory(folder_path: str):
    shutil.rmtree(folder_path)

def create_directory(dir_path):
    os.makedirs(dir_path)

def create_pipeline_input(transformer_steps: PreprocessingSteps):
    transformer_pipeline_steps = []
    for individual_step in transformer_steps:
        transformer = dict()
        transformer['step'] = individual_step.step
        transformer['column'] = individual_step.column
        if individual_step.params:
            for individual_param in individual_step.params:
                transformer[individual_param.key] = individual_param.value
        transformer_pipeline_steps.append(transformer)
    return transformer_pipeline_steps


@app.post("/preprocessing/createpipeline", response_class=FileResponse)
def create_preprocesing_pipeline(preprocessing_step: PreprocessingSteps,
                                 background_tasks: BackgroundTasks):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    transformer_pipeline_steps = create_pipeline_input(
        preprocessing_step.steps)
    transformer_pipeline = create_preprocessing_pipeline(
        transformer_pipeline_steps)
    pipeline_id = str(id(transformer_pipeline))
    dir_path = ROOT_DIR + '/' + pipeline_id
    file_path = dir_path + "/prprocessing.sav"
    create_directory(dir_path)
    joblib.dump(transformer_pipeline, file_path)
    background_tasks.add_task(delete_directory, dir_path)
    return file_path

#  http://127.0.0.1:8000/preprocessing/createpipeline
# uvicorn src.ml_preprocessing_service.main:app --reload
