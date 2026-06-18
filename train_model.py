"""
DeepIDS - Model Training Pipeline
Models: Gradient Boosting Classifier
Task:   5-class classification (normal, dos, probe, r2l, u2r)
Data:   NSL-KDD (KDDTrain+.txt)
SMOTE:  Applied to training set to balance minority classes
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import joblib

FILE_NAME = "KDDTrain+.txt"
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
]

DOS_LIST   = ['neptune','back','land','pod','smurf','teardrop','apache2','udpstorm','processtable','worm']
PROBE_LIST = ['satan','ipsweep','nmap','portsweep','mscan','saint']
R2L_LIST   = ['guess_passwd','ftp_write','imap','phf','multihop','warezmaster','warezclient',
               'spy','xlock','xsnoop','snmpguess','snmpgetattack','httptunnel','sendmail','named']
U2R_LIST   = ['buffer_overflow','loadmodule','rootkit','perl','sqlattack','xterm','ps']

def categorise(x):
    if x == 'normal':     return 'normal'
    elif x in DOS_LIST:   return 'dos'
    elif x in PROBE_LIST: return 'probe'
    elif x in R2L_LIST:   return 'r2l'
    elif x in U2R_LIST:   return 'u2r'
    else:                 return 'dos'

def main():
    if not os.path.exists(FILE_NAME):
        raise FileNotFoundError(
            f"{FILE_NAME} not found. Please place the NSL-KDD dataset in the project folder.")

    print("Loading dataset...")
    df = pd.read_csv(FILE_NAME, header=None)
    df.columns = COLUMNS

    print("Mapping attack labels to 5 categories...")
    df['category'] = df['attack'].apply(categorise)
    print(df['category'].value_counts().to_string())

    # All 41 features retained as per dissertation
    feature_cols = [c for c in COLUMNS if c not in ('attack', 'level')]
    X = df[feature_cols].copy()
    y = df['category']

    print("\nEncoding categorical features...")
    le_protocol = LabelEncoder()
    le_service  = LabelEncoder()
    le_flag     = LabelEncoder()
    le_target   = LabelEncoder()

    X['protocol_type'] = le_protocol.fit_transform(X['protocol_type'])
    X['service']       = le_service.fit_transform(X['service'])
    X['flag']          = le_flag.fit_transform(X['flag'])
    y_enc              = le_target.fit_transform(y)
    class_names        = list(le_target.classes_)
    print("Classes:", class_names)

    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nSplitting into 80% train / 20% test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("\nApplying SMOTE to training set...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"  Training set after SMOTE: {len(X_train_sm):,} samples")
    from collections import Counter
    print("  Class counts after SMOTE:", Counter(y_train_sm))

    print("\nTraining Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_sm, y_train_sm)
    print("Training complete.")

    from sklearn.metrics import classification_report, accuracy_score
    preds = model.predict(X_test)
    print("\n--- Test Set Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=class_names))

    print("\nSaving models to disk...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,       "models/gb_model.pkl")
    joblib.dump(scaler,      "models/scaler.pkl")
    joblib.dump(le_protocol, "models/le_protocol.pkl")
    joblib.dump(le_service,  "models/le_service.pkl")
    joblib.dump(le_flag,     "models/le_flag.pkl")
    joblib.dump(le_target,   "models/le_target.pkl")
    joblib.dump(class_names, "models/class_names.pkl")
    joblib.dump(feature_cols,"models/feature_cols.pkl")
    print("All models saved to models/ directory.")

if __name__ == "__main__":
    main()
