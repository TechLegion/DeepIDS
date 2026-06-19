import joblib
import shap
import numpy as np
import pandas as pd

def main():
    model = joblib.load("models/gb_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    class_names = joblib.load("models/class_names.pkl")

    # The data is StandardScaled, so a vector of 0s represents the mean of the training set.
    # This is a very efficient and mathematically sound background for SHAP.
    background = np.zeros((1, len(feature_cols)))

    print("Initializing KernelExplainer...")
    explainer = shap.KernelExplainer(model.predict_proba, background)

    print("Creating a dummy test instance...")
    # Just a random instance for testing
    X_test = np.random.randn(1, len(feature_cols))

    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_test)

    print("Shape of shap_values:", np.shape(shap_values))
    print("Is list?", isinstance(shap_values, list))
    if isinstance(shap_values, list):
        print("Length of list:", len(shap_values))
        print("Shape of first item:", np.shape(shap_values[0]))
    print("Success!")

if __name__ == "__main__":
    main()
