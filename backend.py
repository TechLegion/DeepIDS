from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from typing import List, Optional
import os
import shap

app = FastAPI(title="DeepIDS - Intrusion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
MODEL        = None
SCALER       = None
LE_PROTOCOL  = None
LE_SERVICE   = None
LE_FLAG      = None
LE_TARGET    = None
CLASS_NAMES  = None
FEATURE_COLS = None
EXPLAINER    = None

@app.on_event("startup")
def load_models():
    global MODEL, SCALER, LE_PROTOCOL, LE_SERVICE, LE_FLAG, LE_TARGET, CLASS_NAMES, FEATURE_COLS
    print("Loading models from disk...")
    model_path = "models/gb_model.pkl"
    if not os.path.exists(model_path):
        print("Models not found! Please run train_model.py first.")
        return
    MODEL        = joblib.load("models/gb_model.pkl")
    SCALER       = joblib.load("models/scaler.pkl")
    LE_PROTOCOL  = joblib.load("models/le_protocol.pkl")
    LE_SERVICE   = joblib.load("models/le_service.pkl")
    LE_FLAG      = joblib.load("models/le_flag.pkl")
    LE_TARGET    = joblib.load("models/le_target.pkl")
    CLASS_NAMES  = joblib.load("models/class_names.pkl")
    FEATURE_COLS = joblib.load("models/feature_cols.pkl")
    print(f"Models loaded. Classes: {CLASS_NAMES}")
    
    print("Initializing SHAP Explainer...")
    global EXPLAINER
    # Using KernelExplainer because TreeExplainer doesn't support sklearn multiclass GBC
    background = np.zeros((1, len(FEATURE_COLS)))
    EXPLAINER = shap.KernelExplainer(MODEL.predict_proba, background)
    print("SHAP Explainer initialized.")

# ---------------------------------------------------------
# REQUEST SCHEMAS
# ---------------------------------------------------------
class PredictionRequest(BaseModel):
    # Core features collected from the UI form
    protocol_type: Optional[str] = None
    protocol: Optional[str] = None
    service: str
    flag: str
    duration: float = 0.0
    src_bytes: float = 0.0
    dst_bytes: float = 0.0
    count: float = 2.0
    serror_rate: float = 0.0
    dst_host_count: float = 150.0
    # Optional extras (default to 0 if not provided)
    land: float = 0.0
    wrong_fragment: float = 0.0
    urgent: float = 0.0
    hot: float = 0.0
    num_failed_logins: float = 0.0
    logged_in: float = 1.0
    num_compromised: float = 0.0
    root_shell: float = 0.0
    su_attempted: float = 0.0
    num_root: float = 0.0
    num_file_creations: float = 0.0
    num_shells: float = 0.0
    num_access_files: float = 0.0
    num_outbound_cmds: float = 0.0
    is_host_login: float = 0.0
    is_guest_login: float = 0.0
    srv_count: float = 0.0
    srv_serror_rate: float = 0.0
    rerror_rate: float = 0.0
    srv_rerror_rate: float = 0.0
    same_srv_rate: float = 1.0
    diff_srv_rate: float = 0.0
    srv_diff_host_rate: float = 0.0
    dst_host_srv_count: float = 0.0
    dst_host_same_srv_rate: float = 1.0
    dst_host_diff_srv_rate: float = 0.0
    dst_host_same_src_port_rate: float = 0.0
    dst_host_srv_diff_host_rate: float = 0.0
    dst_host_serror_rate: float = 0.0
    dst_host_srv_serror_rate: float = 0.0
    dst_host_rerror_rate: float = 0.0
    dst_host_srv_rerror_rate: float = 0.0

class BatchPredictionRequest(BaseModel):
    records: List[PredictionRequest]

# ---------------------------------------------------------
# FEATURE PREPARATION
# ---------------------------------------------------------
def safe_encode(encoder, value, fallback):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return encoder.transform([fallback])[0]

def prepare_features(data: PredictionRequest) -> pd.DataFrame:
    protocol_val = data.protocol_type or data.protocol or 'tcp'
    row = {
        'duration':                   data.duration,
        'protocol_type':              safe_encode(LE_PROTOCOL, protocol_val, 'tcp'),
        'service':                    safe_encode(LE_SERVICE,  data.service,       'http'),
        'flag':                       safe_encode(LE_FLAG,     data.flag,          'SF'),
        'src_bytes':                  data.src_bytes,
        'dst_bytes':                  data.dst_bytes,
        'land':                       data.land,
        'wrong_fragment':             data.wrong_fragment,
        'urgent':                     data.urgent,
        'hot':                        data.hot,
        'num_failed_logins':          data.num_failed_logins,
        'logged_in':                  data.logged_in,
        'num_compromised':            data.num_compromised,
        'root_shell':                 data.root_shell,
        'su_attempted':               data.su_attempted,
        'num_root':                   data.num_root,
        'num_file_creations':         data.num_file_creations,
        'num_shells':                 data.num_shells,
        'num_access_files':           data.num_access_files,
        'num_outbound_cmds':          data.num_outbound_cmds,
        'is_host_login':              data.is_host_login,
        'is_guest_login':             data.is_guest_login,
        'count':                      data.count,
        'srv_count':                  data.srv_count,
        'serror_rate':                data.serror_rate,
        'srv_serror_rate':            data.srv_serror_rate,
        'rerror_rate':                data.rerror_rate,
        'srv_rerror_rate':            data.srv_rerror_rate,
        'same_srv_rate':              data.same_srv_rate,
        'diff_srv_rate':              data.diff_srv_rate,
        'srv_diff_host_rate':         data.srv_diff_host_rate,
        'dst_host_count':             data.dst_host_count,
        'dst_host_srv_count':         data.dst_host_srv_count,
        'dst_host_same_srv_rate':     data.dst_host_same_srv_rate,
        'dst_host_diff_srv_rate':     data.dst_host_diff_srv_rate,
        'dst_host_same_src_port_rate':data.dst_host_same_src_port_rate,
        'dst_host_srv_diff_host_rate':data.dst_host_srv_diff_host_rate,
        'dst_host_serror_rate':       data.dst_host_serror_rate,
        'dst_host_srv_serror_rate':   data.dst_host_srv_serror_rate,
        'dst_host_rerror_rate':       data.dst_host_rerror_rate,
        'dst_host_srv_rerror_rate':   data.dst_host_srv_rerror_rate,
    }
    df = pd.DataFrame([row])[FEATURE_COLS]
    return pd.DataFrame(SCALER.transform(df), columns=FEATURE_COLS)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
CATEGORY_LABELS = {
    'normal': 'Normal Traffic',
    'dos':    'DoS Attack',
    'probe':  'Probe / Scan Attack',
    'r2l':    'Remote-to-Local Attack',
    'u2r':    'User-to-Root Attack',
}

ATTACK_DESCRIPTIONS = {
    'normal': 'Connection patterns align with baseline normal traffic.',
    'dos':    'Denial-of-Service attack detected. High volume of connection attempts targeting a single host.',
    'probe':  'Probe / Reconnaissance attack detected. Systematic scanning of ports or services.',
    'r2l':    'Remote-to-Local intrusion detected. Unauthorized access attempt from a remote machine.',
    'u2r':    'User-to-Root privilege escalation detected. Attempt to gain root/admin access.',
}

def make_prediction(data: PredictionRequest):
    X = prepare_features(data)
    pred_idx  = int(MODEL.predict(X)[0])
    proba     = MODEL.predict_proba(X)[0]
    prediction = CLASS_NAMES[pred_idx]
    confidence = float(proba[pred_idx])
    probabilities = {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}

    # Calculate SHAP values for the single instance
    shap_values = EXPLAINER.shap_values(X)
    
    # Handle both single array (3D) and list of arrays (2D per class)
    if isinstance(shap_values, list):
        class_shap_values = shap_values[pred_idx][0]
    elif len(np.shape(shap_values)) == 3:
        class_shap_values = shap_values[0, :, pred_idx]
    else:
        # Fallback
        class_shap_values = shap_values[0] if len(np.shape(shap_values)) == 2 else shap_values[0, :]
        
    # Extract feature impacts
    feature_impacts = []
    for i, col in enumerate(FEATURE_COLS):
        val = class_shap_values[i]
        feature_impacts.append({
            "feature": col,
            "value": float(val),
            "abs_value": abs(float(val))
        })
        
    # Sort by absolute impact to find the most influential features
    feature_impacts.sort(key=lambda x: x["abs_value"], reverse=True)
    top_features = feature_impacts[:3]

    return {
        "prediction":   prediction,
        "label":        CATEGORY_LABELS.get(prediction, prediction.upper()),
        "confidence":   confidence,
        "probabilities": probabilities,
        "description":  ATTACK_DESCRIPTIONS.get(prediction, ''),
        "is_attack":    prediction != 'normal',
        "explanation":  top_features
    }

@app.post("/predict")
def predict_endpoint(data: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")
    return make_prediction(data)

@app.post("/predict_batch")
def predict_batch_endpoint(batch: BatchPredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")
    return {"results": [make_prediction(r) for r in batch.records]}

@app.get("/classes")
def get_classes():
    return {"classes": CLASS_NAMES, "labels": CATEGORY_LABELS}

# ---------------------------------------------------------
# STATIC FILE SERVING (Frontend) — must come LAST
# ---------------------------------------------------------
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
