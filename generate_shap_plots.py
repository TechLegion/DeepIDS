import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    print("Loading models and encoders...")
    model_path = "models/gb_model.pkl"
    if not os.path.exists(model_path):
        print("Model not found! Please run train_model.py first.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    class_names = joblib.load("models/class_names.pkl")

    # Load a small sample from the dataset to compute SHAP values
    data_file = "KDDTrain+.txt"
    print(f"Loading data from {data_file} for SHAP analysis...")
    # Load just enough rows to get a representative sample (e.g., 2000 rows)
    # The columns match train_model.py
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
    
    df = pd.read_csv(data_file, header=None, names=COLUMNS, nrows=5000)
    
    # We need to preprocess it the same way as in train_model.py
    # Load encoders
    le_protocol = joblib.load("models/le_protocol.pkl")
    le_service = joblib.load("models/le_service.pkl")
    le_flag = joblib.load("models/le_flag.pkl")

    X = df[feature_cols].copy()
    
    # Handle unseen labels by mapping to a common one if needed, or just transform
    # To be safe against unseen labels in our small sample:
    X['protocol_type'] = X['protocol_type'].apply(lambda x: x if x in le_protocol.classes_ else le_protocol.classes_[0])
    X['service'] = X['service'].apply(lambda x: x if x in le_service.classes_ else le_service.classes_[0])
    X['flag'] = X['flag'].apply(lambda x: x if x in le_flag.classes_ else le_flag.classes_[0])

    X['protocol_type'] = le_protocol.transform(X['protocol_type'])
    X['service'] = le_service.transform(X['service'])
    X['flag'] = le_flag.transform(X['flag'])

    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    print("Initializing SHAP KernelExplainer...")
    # Initialize the explainer using K-means summary as background for KernelExplainer
    # TreeExplainer doesn't support sklearn's multiclass GradientBoostingClassifier
    background = shap.kmeans(X_scaled_df, 10)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    print("Calculating SHAP values (this may take a minute)...")
    # Evaluate on a smaller subset to save time since KernelExplainer is computationally heavier
    X_sample = X_scaled_df.iloc[:100]
    shap_values = explainer.shap_values(X_sample)

    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, X_sample, plot_type="bar", class_names=class_names, show=False)
    elif len(np.shape(shap_values)) == 3: 
        # (n_samples, n_features, n_classes)
        shap.summary_plot(shap_values, X_sample, plot_type="bar", class_names=class_names, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, show=False)
        
    plt.tight_layout()
    plt.savefig("shap_summary_plot_bar.png", dpi=300)
    plt.close()
    
    print("Generating SHAP feature importance plot for normal traffic...")
    normal_idx = class_names.index('normal')
    plt.figure(figsize=(10, 8))
    
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[normal_idx], X_sample, show=False)
    elif len(np.shape(shap_values)) == 3:
        shap.summary_plot(shap_values[:, :, normal_idx], X_sample, show=False)
        
    plt.tight_layout()
    plt.savefig("shap_summary_plot_normal.png", dpi=300)
    plt.close()

    print("Saved 'shap_summary_plot_bar.png' and 'shap_summary_plot_normal.png' to disk.")
    print("These static plots can be used in your Final Report.")

if __name__ == "__main__":
    main()
