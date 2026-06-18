from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Any
import os

app = FastAPI(title="IDS Backend API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and preprocessors
MODEL = None
SCALER = None
LE_PROTOCOL = None
LE_SERVICE = None
LE_FLAG = None
CLASS_NAMES = None

# ---------------------------------------------------------
# STARTUP EVENT - LOAD MODELS
# ---------------------------------------------------------
@app.on_event("startup")
def load_models():
    global MODEL, SCALER, LE_PROTOCOL, LE_SERVICE, LE_FLAG, CLASS_NAMES
    print("Loading models from disk...")
    if not os.path.exists("models/xgb_model.pkl"):
        print("Models not found! Please run train_model.py first.")
        return
        
    MODEL = joblib.load("models/xgb_model.pkl")
    SCALER = joblib.load("models/scaler.pkl")
    LE_PROTOCOL = joblib.load("models/le_protocol.pkl")
    LE_SERVICE = joblib.load("models/le_service.pkl")
    LE_FLAG = joblib.load("models/le_flag.pkl")
    CLASS_NAMES = joblib.load("models/class_names.pkl")
    print("Models loaded successfully!")

# ---------------------------------------------------------
# API MODELS
# ---------------------------------------------------------
class PredictionRequest(BaseModel):
    protocol: str
    service: str
    flag: str
    duration: float
    src_bytes: float
    dst_bytes: float
    count: float
    serror_rate: float
    dst_host_count: float # Sent by UI but not in Top 15 features

class BatchPredictionRequest(BaseModel):
    records: List[PredictionRequest]

# ---------------------------------------------------------
# PREDICTION LOGIC
# ---------------------------------------------------------
def prepare_features(data: PredictionRequest):
    # Top 15 Features needed by the XGBoost Model
    features = {
        'duration': data.duration, 
        'protocol_type': data.protocol, 
        'service': data.service, 
        'flag': data.flag, 
        'src_bytes': data.src_bytes, 
        'dst_bytes': data.dst_bytes, 
        'wrong_fragment': 0, 
        'hot': 0, 
        'logged_in': 0, 
        'num_compromised': 0, 
        'count': data.count, 
        'srv_count': 0, 
        'serror_rate': data.serror_rate, 
        'srv_serror_rate': 0, 
        'rerror_rate': 0
    }
    
    df = pd.DataFrame([features])

    # Encode categorical
    try:
        df['protocol_type'] = LE_PROTOCOL.transform(df['protocol_type'])
    except ValueError:
        df['protocol_type'] = LE_PROTOCOL.transform(['tcp'])
    
    try:
        df['service'] = LE_SERVICE.transform(df['service'])
    except ValueError:
        df['service'] = LE_SERVICE.transform(['other'])

    try:
        df['flag'] = LE_FLAG.transform(df['flag'])
    except ValueError:
        df['flag'] = LE_FLAG.transform(['SF'])

    df_scaled = pd.DataFrame(SCALER.transform(df), columns=df.columns)
    return df_scaled

def make_prediction(data: PredictionRequest):
    X = prepare_features(data)
    pred_idx = MODEL.predict(X)[0]
    proba = MODEL.predict_proba(X)[0]
    
    prediction = CLASS_NAMES[pred_idx]
    confidence = float(proba[pred_idx])
    
    # Map probabilities to class names for the chart
    probabilities_dict = {str(CLASS_NAMES[i]): float(proba[i]) for i in range(len(CLASS_NAMES))}

    # Generate reasons based on heuristics to match UI expectations
    reasons = []
    normalReasons = []

    if prediction == 'attack':
        reasons.append('Pattern flagged as an anomaly/attack by the XGBoost Model')
        if data.count > 100: reasons.append(f'Unusually high connection count ({data.count})')
        if data.serror_rate > 0.5: reasons.append(f'High SYN error rate ({data.serror_rate})')
        if data.duration > 10: reasons.append(f'Long session duration ({data.duration}s)')
    else:
        if data.flag == 'SF': normalReasons.append('Normal connection finish (SF flag)')
        if data.serror_rate == 0: normalReasons.append('No errors reported')
        normalReasons.append('Feature vector aligns with baseline normal traffic')

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities_dict,
        "reasons": reasons,
        "normalReasons": normalReasons
    }

@app.post("/predict")
def predict_endpoint(data: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready yet")
    result = make_prediction(data)
    return result

@app.post("/predict_batch")
def predict_batch_endpoint(batch: BatchPredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready yet")
    results = [make_prediction(record) for record in batch.records]
    return {"results": results}

# ---------------------------------------------------------
# STATIC FILE SERVING (Frontend) - Must come LAST
# ---------------------------------------------------------
# Serve JS and CSS as static assets
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_root():
    return FileResponse("index.html")

@app.get("/script.js")
def serve_js():
    return FileResponse("script.js")

@app.get("/style.css")
def serve_css():
    return FileResponse("style.css")
