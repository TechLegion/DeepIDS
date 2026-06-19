# Comprehensive NIDS Report, Code, and Evaluation

This document contains all requested implementation code snippets (including imported libraries), detailed evaluation results, confusion matrices, model selection rationale, and the final executive summary and recommendations.

---

## 1. Code for Duplicate Record Removal

*Note: The NSL-KDD dataset removes duplicates by design during its creation to resolve the bias issues present in the original KDD Cup 99 dataset. The following code verifies that no duplicate rows exist.*

```python
import pandas as pd

# Load dataset
df = pd.read_csv('KDDTrain+.txt', header=None)

# Check number of duplicate records in the dataframe
duplicate_count = df.duplicated().sum()
print(f"Total duplicate records found: {duplicate_count}")

# Output:
# Total duplicate records found: 0
# "Dataset doesn't contain any duplicated row"
```

---

## 2. Feature Encoding and Scaling Code

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Initialize label encoders for categorical features and the target label
le_protocol = LabelEncoder()
le_service  = LabelEncoder()
le_flag     = LabelEncoder()
le_target   = LabelEncoder()

# Fit and transform categorical columns
X['protocol_type'] = le_protocol.fit_transform(X['protocol_type'])
X['service']       = le_service.fit_transform(X['service'])
X['flag']          = le_flag.fit_transform(X['flag'])

# Encode target variable
y_enc              = le_target.fit_transform(y)
class_names        = list(le_target.classes_)

# Apply Standard Scaling to normalize all 41 feature columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 3. Imbalanced Dataset Split Code

```python
from sklearn.model_selection import train_test_split

# Stratified split ensures train and test sets have identical
# proportions of the 5 attack categories (dos, probe, r2l, u2r, normal)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, 
    y_enc, 
    test_size=0.20, 
    random_state=42, 
    stratify=y_enc
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")
```

---

## 4. Implementation Code of SMOTE for the Balanced Dataset

```python
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------
# Method A: Standard SMOTE library implementation
# ---------------------------------------------------------
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# ---------------------------------------------------------
# Method B: Custom SMOTE implementation (from scratch)
# ---------------------------------------------------------
def simple_smote(X, y, k=5, random_state=42):
    np.random.seed(random_state)
    X_resampled, y_resampled = [X.copy()], [y.copy()]
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        n_samples = len(cls_idx)
        n_needed = max_count - n_samples
        if n_needed <= 0: continue
        
        X_cls = X.iloc[cls_idx]
        nn = NearestNeighbors(n_neighbors=min(k, n_samples))
        nn.fit(X_cls)
        
        new_samples = []
        for _ in range(n_needed):
            idx = np.random.randint(0, n_samples)
            sample = X_cls.iloc[idx]
            _, neighbors = nn.kneighbors([sample])
            neighbor_idx = np.random.choice(neighbors[0][1:])
            neighbor = X_cls.iloc[neighbor_idx]
            
            diff = neighbor.values - sample.values
            new_sample = sample.values + np.random.random() * diff
            new_samples.append(new_sample)
            
        X_resampled.append(pd.DataFrame(new_samples, columns=X.columns))
        y_resampled.append(np.full(len(new_samples), cls))
        
    return pd.concat(X_resampled, ignore_index=True), np.concatenate(y_resampled)
```

---

## 5. Training the Three Algorithms on the 80:20 Split (Imbalanced Dataset)

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the three algorithms
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42)
}

results_imbalanced = {}

# Train on the original imbalanced training set (X_train_eng)
for name, model in models.items():
    print(f"Training {name} on imbalanced data...")
    model.fit(X_train_eng, y_train)
    
    # Predict on the imbalanced test set
    y_pred = model.predict(X_test_eng)
    
    # Store metrics
    results_imbalanced[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
```

---

## 6. Training the Three Algorithms on the 80:20 Split (Balanced Dataset)

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the models (same hyperparameters as imbalanced run)
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42)
}

results_balanced = {}

# Train on the SMOTE-balanced training set (X_train_bal)
for name, model in models.items():
    print(f"Training {name} on SMOTE-balanced data...")
    model.fit(X_train_bal, y_train_bal)
    
    # Predict on the true imbalanced test set to reflect real-world data distribution
    y_pred = model.predict(X_test_eng)
    
    # Store metrics
    results_balanced[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
```

---

## 7. Confusion Matrices on the Balanced Training

Below are the confusion matrices of the three models evaluated on the imbalanced test set after training on the SMOTE-balanced dataset. The order of the rows and columns represents the attack classes: **`[dos, normal, probe, r2l, u2r]`**.

### Random Forest Classifier
```text
          Pred_dos  Pred_normal  Pred_probe  Pred_r2l  Pred_u2r
True_dos    9185          1           0         0         0
True_norm      2      13464           1         1         1
True_prob      0          6        2324         1         0
True_r2l       0          7           0       192         0
True_u2r       0          2           0         0         8
```

### Gradient Boosting Classifier
```text
          Pred_dos  Pred_normal  Pred_probe  Pred_r2l  Pred_u2r
True_dos    9185          1           0         0         0
True_norm      2      13452           2         8         5
True_prob      0         12        2319         0         0
True_r2l       0          2           0       197         0
True_u2r       0          0           0         0        10
```

### Decision Tree Classifier
```text
          Pred_dos  Pred_normal  Pred_probe  Pred_r2l  Pred_u2r
True_dos    9129         35          14         0         8
True_norm     46      12976         124       253        70
True_prob      0         27        2301         0         3
True_r2l       1          0           0       197         1
True_u2r       0          0           0         0        10
```

---

## 8. Evaluation Results on the Tested Dataset

All three models were trained on the SMOTE-balanced training split and evaluated on the true, imbalanced test split (25,195 connection records) to ensure realism.

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **99.91%** | **99.91%** | **99.91%** | **99.91%** |
| **Gradient Boosting** | 99.87% | 99.88% | 99.87% | 99.88% |
| **Decision Tree** | 97.69% | 98.56% | 97.69% | 98.01% |

---

## 9. Explanation of Per-Class Performance and Model Selection Rationale

**Why Random Forest is the Best Model (and Why We Selected It):**

While Gradient Boosting and Random Forest both perform exceptionally well overall, **Random Forest is selected as the optimal model** for this Network Intrusion Detection System (NIDS) because of its superior handling of critical minority classes (specifically Remote-to-Local `r2l` and User-to-Root `u2r` attacks).

1. **Optimal Balance of Precision on Rare Classes:** 
   User-to-Root (U2R) attacks are extremely rare in the dataset (only 10 samples in the test set). While Gradient Boosting achieves a perfect recall of **100%** on U2R (catching all 10 attacks), it generates **5 false positives** by misclassifying normal traffic as U2R, resulting in a poor precision of `66.67%`. In a real-world enterprise network, false positives trigger manual incident responses, wasting cybersecurity analysts' time (alert fatigue). Random Forest detects **80%** of U2R attacks but only produces **1 false positive**, raising precision to `88.89%` and achieving the highest U2R F1-Score of **84.21%** (compared to Gradient Boosting's `80.00%`).
2. **Mitigation of SMOTE-Induced Overfitting:** 
   SMOTE balances datasets by generating synthetic data points between existing minority class samples. **Boosting algorithms** (like Gradient Boosting) build trees sequentially to minimize residuals; they tend to over-focus on the artificial patterns created by SMOTE, which leads to misclassifying normal traffic in the real-world test set. **Bagging algorithms** (like Random Forest) train independent trees on bootstrapped subsets of the data and average their predictions. This averaging smooths out the decision boundary, reducing model variance and making it highly robust against synthetic SMOTE noise.
3. **Operational Efficiency:** 
   Random Forest is fully parallelizable (`n_jobs=-1`), allowing individual decision trees to be trained simultaneously on multiple CPU cores. This allows for significantly faster retraining lifecycles compared to Gradient Boosting, which must build trees sequentially.

---

## 10. Summary, Conclusion, and Future Works & Recommendations

### Executive Summary
This project successfully designed, trained, and evaluated an advanced multi-class Network Intrusion Detection System (NIDS) tailored to categorize high-throughput network telemetry into five distinct categories: Normal, DoS, Probe, R2L, and U2R. Utilizing the NSL-KDD dataset, the pipeline executed robust data preprocessing, encompassing categorical feature encoding, standard scaling, and the application of the Synthetic Minority Over-sampling Technique (SMOTE) to rectify extreme class imbalances. We benchmarked three robust machine learning algorithms: Decision Tree, Gradient Boosting, and Random Forest. Evaluation on the true, imbalanced test split demonstrated that **Random Forest is the superior predictive model**, achieving a peak accuracy of **99.91%** and an F1-Score of **99.91%**, effectively isolating rare, high-severity attacks while maintaining an exceptional operational false-positive rate.

### Conclusion
1. **The Necessity of Class Balancing:** Untreated class imbalance causes machine learning classifiers to disregard rare but critical attacks (e.g., U2R, R2L) in favor of optimizing for majority classes (e.g., Normal, DoS). Implementing SMOTE allowed our algorithms to learn the intricate feature geometries of minority attacks.
2. **Real-World Evaluation Integrity:** Testing models on an artificially balanced dataset yields inflated, unrealistic scores. By strictly evaluating our models on an *imbalanced* test split, we guarantee that the reported metrics faithfully mirror real-world production network conditions.
3. **Random Forest's Structural Superiority in NIDS:** By employing bootstrap aggregating (bagging) and randomized feature subspace selection, Random Forest generated smoother decision boundaries than Gradient Boosting. This successfully prevented the algorithm from overfitting to the synthetic data points generated by SMOTE, culminating in higher precision and fewer false alarms.

### Future Works & Recommendations
To maximize the system's defensive capabilities, we recommend the following strategic next steps for future deployment:

1. **Deploy Random Forest to Production:** Transition the current baseline models to the Random Forest architecture to immediately minimize false alarms (alert fatigue) and optimize the detection accuracy of critical privilege escalation (U2R) and unauthorized access (R2L) threats.
2. **Implement an Ensemble Voting Classifier:** Develop a hybrid voting ensemble that combines the strengths of Random Forest and Gradient Boosting. By taking a weighted average or soft vote, the system can leverage Gradient Boosting's high recall capabilities alongside Random Forest's high precision.
3. **Incorporate Deep Learning (Autoencoders):** For zero-day attack detection (novel attacks absent from the training dataset), integrate an unsupervised anomaly detection model such as a Deep Neural Autoencoder. This model will establish a baseline of "normal" traffic geometry and flag entirely novel connection structures, acting as a secondary safety net.
4. **Transition to Online Incremental Learning:** To ensure the NIDS remains resilient against shifting network conditions and rapidly evolving threat vectors, deploy incremental learning models (e.g., Stochastic Gradient Descent classifiers) that continuously adapt to new data without requiring a full pipeline retrain.
