from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from pathlib import Path

project_dir = Path.cwd()
models_dir = project_dir / 'models'

app = FastAPI()
gmm_pipe = load(models_dir / 'gmm_pipe.joblib')

@api.get("/")
def read_root():
	return {"Hello": "World"}
	
@api.get("/health")
def healthcheck():
	return "GMM clustering is all ready to go!"
	
def format_features(genre, age, income, spending):
	result = {'genre': genre,
			  'age': age,
			  'income': income,
			  'spending': spending}
	return result
	
@app.get('/mall/customers/segmentation')
def predict(genre, age, income, spending):
	df = pd.DataFrame(format_features(genre, age, income, spending))
	prediction = gmm_pipe.predict(df)
	str_pred = np.array2string(prediction)
	
	return jsonify(str_pred)