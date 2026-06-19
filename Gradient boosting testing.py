"""

GRADIENT BOOSTING CLASSIFIER: TRAIN ON SMOTE-BALANCED, TEST ON IMBALANCED
Complete evaluation with all 5 metrics + ROC-AUC + per-class performance

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD OR GENERATE DATA
# ============================================================
# For real NSL-KDD, replace this with:
# df = pd.read_csv('NSL_KDD_train.csv')

np.random.seed(42)
n = 10000

protocol_type = np.random.choice(['tcp','udp','icmp'], n, p=[0.65,0.25,0.10])
service = np.random.choice(['http','smtp','ftp','private','domain','ssh','other'], n, p=[0.40,0.15,0.12,0.10,0.08,0.08,0.07])
flag = np.random.choice(['SF','S0','REJ','RSTO','SH','RSTR','OTH','S1','S2','S3'], n, p=[0.55,0.15,0.10,0.06,0.04,0.03,0.03,0.02,0.01,0.01])
duration = np.random.exponential(50,n).clip(0,5000)
src_bytes = np.random.lognormal(8,2,n).clip(0,1e6)
dst_bytes = np.random.lognormal(7,2,n).clip(0,1e6)
land = np.random.choice([0,1],n,p=[0.999,0.001])
wrong_fragment = np.random.poisson(0.1,n).clip(0,3)
urgent = np.random.poisson(0.01,n).clip(0,2)
hot = np.random.poisson(0.3,n).clip(0,15)
num_failed_logins = np.random.poisson(0.05,n).clip(0,5)
logged_in = np.random.choice([0,1],n,p=[0.45,0.55])
num_compromised = np.random.poisson(0.1,n).clip(0,5)
root_shell = np.random.choice([0,1],n,p=[0.999,0.001])
su_attempted = np.random.choice([0,1],n,p=[0.998,0.002])
num_root = np.random.poisson(0.05,n).clip(0,3)
num_file_creations = np.random.poisson(0.1,n).clip(0,5)
num_shells = np.random.poisson(0.01,n).clip(0,2)
num_access_files = np.random.poisson(0.05,n).clip(0,3)
num_outbound_cmds = np.zeros(n,dtype=int)
is_host_login = np.random.choice([0,1],n,p=[0.99,0.01])
is_guest_login = np.random.choice([0,1],n,p=[0.97,0.03])
count = np.random.poisson(50,n).clip(0,500)
srv_count = np.random.poisson(20,n).clip(0,300)
serror_rate = np.random.beta(2,8,n)
srv_serror_rate = np.random.beta(2,8,n)
rerror_rate = np.random.beta(2,10,n)
srv_rerror_rate = np.random.beta(2,10,n)
same_srv_rate = np.random.beta(8,2,n)
diff_srv_rate = np.random.beta(2,8,n)
srv_diff_host_rate = np.random.beta(3,7,n)
dst_host_count = np.random.poisson(100,n).clip(0,255)
dst_host_srv_count = np.random.poisson(50,n).clip(0,255)
dst_host_same_srv_rate = np.random.beta(7,3,n)
dst_host_diff_srv_rate = np.random.beta(2,8,n)
dst_host_same_src_port_rate = np.random.beta(6,4,n)
dst_host_srv_diff_host_rate = np.random.beta(3,7,n)
dst_host_serror_rate = np.random.beta(2,8,n)
dst_host_srv_serror_rate = np.random.beta(2,8,n)
dst_host_rerror_rate = np.random.beta(2,10,n)
dst_host_srv_rerror_rate = np.random.beta(2,10,n)
class_labels = np.random.choice(['normal','dos','probe','r2l','u2r'],n,p=[0.53,0.37,0.06,0.03,0.01])

df = pd.DataFrame({
    'duration':duration,'protocol_type':protocol_type,'service':service,'flag':flag,
    'src_bytes':src_bytes,'dst_bytes':dst_bytes,'land':land,'wrong_fragment':wrong_fragment,
    'urgent':urgent,'hot':hot,'num_failed_logins':num_failed_logins,'logged_in':logged_in,
    'num_compromised':num_compromised,'root_shell':root_shell,'su_attempted':su_attempted,
    'num_root':num_root,'num_file_creations':num_file_creations,'num_shells':num_shells,
    'num_access_files':num_access_files,'num_outbound_cmds':num_outbound_cmds,
    'is_host_login':is_host_login,'is_guest_login':is_guest_login,'count':count,
    'srv_count':srv_count,'serror_rate':serror_rate,'srv_serror_rate':srv_serror_rate,
    'rerror_rate':rerror_rate,'srv_rerror_rate':srv_rerror_rate,'same_srv_rate':same_srv_rate,
    'diff_srv_rate':diff_srv_rate,'srv_diff_host_rate':srv_diff_host_rate,
    'dst_host_count':dst_host_count,'dst_host_srv_count':dst_host_srv_count,
    'dst_host_same_srv_rate':dst_host_same_srv_rate,'dst_host_diff_srv_rate':dst_host_diff_srv_rate,
    'dst_host_same_src_port_rate':dst_host_same_src_port_rate,
    'dst_host_srv_diff_host_rate':dst_host_srv_diff_host_rate,
    'dst_host_serror_rate':dst_host_serror_rate,'dst_host_srv_serror_rate':dst_host_srv_serror_rate,
    'dst_host_rerror_rate':dst_host_rerror_rate,'dst_host_srv_rerror_rate':dst_host_srv_rerror_rate,
    'class':class_labels
})

X = df.drop(['class'],axis=1)
y = df['class']

# ============================================================
# STEP 2: ENCODE CATEGORICAL VARIABLES
# ============================================================
le_protocol = LabelEncoder()
le_service = LabelEncoder()
le_flag = LabelEncoder()
le_target = LabelEncoder()

X_enc = X.copy()
X_enc['protocol_type'] = le_protocol.fit_transform(X_enc['protocol_type'])
X_enc['service'] = le_service.fit_transform(X_enc['service'])
X_enc['flag'] = le_flag.fit_transform(X_enc['flag'])
y_enc = le_target.fit_transform(y)
class_names = le_target.classes_

print("Class mapping:", dict(zip(class_names, range(len(class_names)))))

# ============================================================
# STEP 3: STRATIFIED TRAIN/TEST SPLIT (80/20)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")

# ============================================================
# STEP 4: FEATURE SCALING
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================
X_train_eng = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_eng = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

X_train_eng['bytes_ratio'] = (X_train['src_bytes'] / (X_train['dst_bytes'] + 1)).values
X_train_eng['srv_ratio'] = (X_train['srv_count'] / (X_train['count'] + 1)).values
X_train_eng['error_interaction'] = (X_train['serror_rate'] * X_train['rerror_rate']).values

X_test_eng['bytes_ratio'] = (X_test['src_bytes'] / (X_test['dst_bytes'] + 1)).values
X_test_eng['srv_ratio'] = (X_test['srv_count'] / (X_test['count'] + 1)).values
X_test_eng['error_interaction'] = (X_test['serror_rate'] * X_test['rerror_rate']).values

X_train_eng = X_train_eng.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_eng = X_test_eng.replace([np.inf, -np.inf], np.nan).fillna(0)

for col in ['bytes_ratio', 'srv_ratio', 'error_interaction']:
    mean = X_train_eng[col].mean()
    std = X_train_eng[col].std()
    if std > 0:
        X_train_eng[col] = (X_train_eng[col] - mean) / std
        X_test_eng[col] = (X_test_eng[col] - mean) / std
    else:
        X_train_eng[col] = 0
        X_test_eng[col] = 0

# ============================================================
# STEP 6: SMOTE (Synthetic Minority Over-sampling Technique)
# ============================================================
def simple_smote(X, y, k=5, random_state=42):
    """
    Simple SMOTE implementation.
    Balances all classes to the size of the majority class.
    """
    np.random.seed(random_state)
    X_resampled = [X.copy()]
    y_resampled = [y.copy()]

    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        n_samples = len(cls_idx)
        n_needed = max_count - n_samples

        if n_needed <= 0:
            continue

        X_cls = X.iloc[cls_idx]

        nn = NearestNeighbors(n_neighbors=min(k, n_samples))
        nn.fit(X_cls)

        new_samples = []
        for _ in range(n_needed):
            idx = np.random.randint(0, n_samples)
            sample = X_cls.iloc[idx]
            _, neighbors = nn.kneighbors([sample])
            # Exclude self (first neighbor is always the sample itself)
            neighbor_idx = np.random.choice(neighbors[0][1:])
            neighbor = X_cls.iloc[neighbor_idx]

            diff = neighbor.values - sample.values
            gap = np.random.random()
            new_sample = sample.values + gap * diff
            new_samples.append(new_sample)

        if new_samples:
            X_resampled.append(pd.DataFrame(new_samples, columns=X.columns))
            y_resampled.append(np.full(len(new_samples), cls))

    X_final = pd.concat(X_resampled, ignore_index=True)
    y_final = np.concatenate(y_resampled)
    return X_final, y_final

# Apply SMOTE ONLY to training data
X_train_bal, y_train_bal = simple_smote(X_train_eng, y_train, k=5, random_state=42)

print(f"\nSMOTE applied!")
print(f"Before: {pd.Series(y_train).value_counts().sort_index().values}")
print(f"After:  {pd.Series(y_train_bal).value_counts().sort_index().values}")
print(f"Synthetic samples created: {len(y_train_bal) - len(y_train)}")

# ============================================================
# STEP 7: TRAIN GRADIENT BOOSTING CLASSIFIER
# ============================================================
print(f"\n{'='*70}")
print("GRADIENT BOOSTING CLASSIFIER")
print(f"{'='*70}")

gb_model = GradientBoostingClassifier(
    n_estimators=100,      # Number of boosting stages
    max_depth=5,            # Maximum depth of each tree
    learning_rate=0.1,     # Step size shrinkage
    subsample=0.8,         # Fraction of samples for each tree (stochastic GB)
    random_state=42        # Reproducibility
)

print("\nModel Parameters:")
print(f"  n_estimators:    {gb_model.n_estimators}")
print(f"  max_depth:       {gb_model.max_depth}")
print(f"  learning_rate:   {gb_model.learning_rate}")
print(f"  subsample:       {gb_model.subsample}")
print(f"  random_state:    {gb_model.random_state}")

# Train on SMOTE-balanced data
print(f"\nTraining on {len(y_train_bal)} balanced samples...")
gb_model.fit(X_train_bal, y_train_bal)
print("Training complete!")

# ============================================================
# STEP 8: PREDICT ON TEST DATA (IMBALANCED - reflects reality)
# ============================================================
print(f"\nPredicting on {len(y_test)} imbalanced test samples...")
y_pred = gb_model.predict(X_test_eng)
y_proba = gb_model.predict_proba(X_test_eng)
print("Prediction complete!")

# ============================================================
# STEP 9: EVALUATION - ALL 5 METRICS + ROC-AUC
# ============================================================
print(f"\n{'='*70}")
print("EVALUATION RESULTS ON TEST DATA")
print(f"{'='*70}")

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n1. ACCURACY:  {accuracy:.4f}")
print(f"   → {accuracy*100:.2f}% of predictions are correct")

# 2. Precision (weighted)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"\n2. PRECISION: {precision:.4f}")
print(f"   → Of all predicted positives, {precision*100:.2f}% are actually positive")

# 3. Recall (weighted)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"\n3. RECALL:    {recall:.4f}")
print(f"   → Of all actual positives, {recall*100:.2f}% are correctly identified")

# 4. F1-Score (weighted) - MOST IMPORTANT FOR IMBALANCED DATA
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"\n4. F1-SCORE:  {f1:.4f}")
print(f"   → Harmonic mean of precision and recall")
print(f"   → Best balanced metric for imbalanced datasets")

# 5. ROC-AUC (One-vs-Rest, weighted)
y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='weighted')
print(f"\n5. ROC-AUC:   {roc_auc:.4f}")
print(f"   → Measures separability across all thresholds")
print(f"   → 0.5 = random, 1.0 = perfect")

# Summary table
print(f"\n{'-'*70}")
print("SUMMARY TABLE")
print(f"{'-'*70}")
print(f"{'Metric':<15} {'Score':>10} {'Interpretation'}")
print(f"{'-'*70}")
print(f"{'Accuracy':<15} {accuracy:>10.4f}  Overall correctness")
print(f"{'Precision':<15} {precision:>10.4f}  True positives / All predicted positives")
print(f"{'Recall':<15} {recall:>10.4f}  True positives / All actual positives")
print(f"{'F1-Score':<15} {f1:>10.4f}  Balance of precision & recall (BEST METRIC)")
print(f"{'ROC-AUC':<15} {roc_auc:>10.4f}  Separability across all thresholds")
print(f"{'-'*70}")

# ============================================================
# STEP 10: PER-CLASS PERFORMANCE
# ============================================================
print(f"\n{'='*70}")
print("PER-CLASS PERFORMANCE")
print(f"{'='*70}")

report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print(f"{'-'*55}")
for cls in class_names:
    print(f"{cls:<10} {report[cls]['precision']:>10.4f} {report[cls]['recall']:>10.4f} {report[cls]['f1-score']:>10.4f} {int(report[cls]['support']):>10}")

# ============================================================
# STEP 11: CONFUSION MATRIX
# ============================================================
print(f"\n{'='*70}")
print("CONFUSION MATRIX")
print(f"{'='*70}")

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=[f'True_{c}' for c in class_names], columns=[f'Pred_{c}' for c in class_names])
print(cm_df)

# ============================================================
# STEP 12: SAVE RESULTS
# ============================================================
print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")
print(f"\nBest Model: Gradient Boosting Classifier")
print(f"Selection Criterion: Highest F1-Score ({f1:.4f})")
print(f"Training Data: SMOTE-balanced ({len(y_train_bal)} samples)")
print(f"Test Data: Imbalanced ({len(y_test)} samples)")
