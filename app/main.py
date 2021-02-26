from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv

# project_dir = Path(find_dotenv()).parent
# print(f'Project dir: {project_dir}')
# models_dir = project_dir / 'models'
# gmm_pipe_dir = models_dir / 'gmm_pipe.joblib'
# print(f'gmm_pipe_dir.is_dir(): {gmm_pipe_dir.is_dir()}')

app = FastAPI()
gmm_pipe = load('../models/gmm_pipe.joblib')


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health", status_code=200)
def healthcheck():
    return "GMM clustering is all ready to go!"


def format_features(genre: str, age:int, income:int, spending:int):
    result = {'Gender': [genre],
              'Age': [age],
              'Annual Income (k$)': [income],
              'spending Score (1-100)': [spending]}
    return result


@app.get('/mall/customers/segmentation')
def predict(genre: str, age: int, income:int, spending:int):
    obs = pd.DataFrame(format_features(genre, age, income, spending))
    prediction = gmm_pipe.predict(obs)

    return JSONResponse(prediction.tolist())
