import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib

URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
FILE_NAME = "KDDTrain+.txt"

def download_dataset():
    if not os.path.exists(FILE_NAME):
        print(f"Downloading {FILE_NAME} from GitHub...")
        response = requests.get(URL)
        response.raise_for_status()
        with open(FILE_NAME, 'w') as f:
            f.write(response.text)
        print("Download complete.")
    else:
        print(f"Found {FILE_NAME} locally.")

def main():
    download_dataset()
    
    print("Loading dataset...")
    df = pd.read_csv(FILE_NAME, header=None)
    
    # 41 features + 1 label + 1 difficulty level
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
               'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
               'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
               'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
               'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
               'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
               'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
               'dst_host_srv_rerror_rate', 'attack', 'level']
    
    df.columns = columns
    
    print("Preprocessing data...")
    # Map to Binary Classification
    df['attack'] = df['attack'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    
    # Top 15 Features selected in the Kaggle notebook
    selected_features = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'wrong_fragment', 
                         'hot', 'logged_in', 'num_compromised', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
                         'rerror_rate']
    
    X = df[selected_features].copy()
    y = df['attack']
    
    # Encoding Categorical Variables
    le_protocol = LabelEncoder()
    le_service = LabelEncoder()
    le_flag = LabelEncoder()
    le_target = LabelEncoder()
    
    X['protocol_type'] = le_protocol.fit_transform(X['protocol_type'])
    X['service'] = le_service.fit_transform(X['service'])
    X['flag'] = le_flag.fit_transform(X['flag'])
    
    y_enc = le_target.fit_transform(y)
    class_names = le_target.classes_
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model Initialization & Training
    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        colsample_bytree=0.5,
        learning_rate=0.1,
        max_depth=6,
        n_estimators=128,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_scaled, y_enc)
    print("Training complete.")
    
    # Save the models
    print("Saving models to disk...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le_protocol, "models/le_protocol.pkl")
    joblib.dump(le_service, "models/le_service.pkl")
    joblib.dump(le_flag, "models/le_flag.pkl")
    joblib.dump(class_names, "models/class_names.pkl")
    print("All models saved successfully to models/ directory.")

if __name__ == "__main__":
    main()
