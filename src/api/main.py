"""
    A simple FastAPI app to serve locally trained LightGBM model
"""

from fastapi import FastAPI
from fastapi import HTTPException
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import pickle
from src.api.schemas import PredictionRequest, PredictionResponse
from src.config.paths import MODEL_DIR, PROCESSED_DATA_DIR
from src.inference.inference import predict

app = FastAPI()

COL_NAMES = pickle.load(open(MODEL_DIR/"retrained"/"col_names.pkl","rb"))

# Simple endpoint to check the api is alive
@app.get("/")
def root():
    return {"message": "ALLSTATE Claims Loss Prediction API running ..."}


# Load model 
@app.post("/predict", response_model=PredictionResponse)
def predict_batch(request: PredictionRequest):
    # Validate input data
    extra = set(request.columns) - set(COL_NAMES)
    if extra:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown columns: {sorted(extra)}"
        )

    # turn input into pd.DataFrame format
    input_df = pd.DataFrame(request.columns)

    # predict
    prediction = predict(input_df=input_df)

    return {"predictions": prediction.tolist()}
    


