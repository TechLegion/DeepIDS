import matplotlib.pyplot as plt
import os

# Define code snippets
snippets = {
    "duplicate_removal_code.png": {
        "title": "Jupyter Notebook - Section 3.2: Duplicate Record Check",
        "code": """# ====================================================================
# NSL-KDD Duplicate Record Verification
# Note: NSL-KDD removes duplicates by design to fix KDD-99 bias.
# ====================================================================

# Check number of duplicate records in the dataframe
duplicate_count = df.duplicated().sum()
print(f"Total duplicate records found: {duplicate_count}")

# Output:
# Total duplicate records found: 0
# "Dataset doesn't contain any duplicated row"
"""
    },
    "encoding_scaling_code.png": {
        "title": "Data Preprocessing: Label Encoding & Feature Scaling",
        "code": """# ====================================================================
# Encoding Categorical Variables & Scaling Features (train_model.py)
# ====================================================================
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Initialize label encoders for categorical features and target
le_protocol = LabelEncoder()
le_service  = LabelEncoder()
le_flag     = LabelEncoder()
le_target   = LabelEncoder()

# Fit and transform categorical columns
X['protocol_type'] = le_protocol.fit_transform(X['protocol_type'])
X['service']       = le_service.fit_transform(X['service'])
X['flag']          = le_flag.fit_transform(X['flag'])
y_enc              = le_target.fit_transform(y)
class_names        = list(le_target.classes_)

# Apply Standard Scaling to all 41 feature columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
"""
    },
    "dataset_split_code.png": {
        "title": "Dataset Splitting: Stratified 80:20 Train/Test Split",
        "code": """# ====================================================================
# Stratified Train/Test Split (80% Train, 20% Test)
# ====================================================================
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
"""
    },
    "smote_code.png": {
        "title": "SMOTE Class Balancing (imblearn & Custom simple_smote)",
        "code": """# ====================================================================
# Scenario A: Standard SMOTE library implementation (train_model.py)
# ====================================================================
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# ====================================================================
# Scenario B: Custom SMOTE implementation (from scratch)
# ====================================================================
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
"""
    },
    "training_imbalanced_code.png": {
        "title": "Training Three Algorithms on Imbalanced 80:20 Split",
        "code": """# ====================================================================
# Scenario 1: Model Training on Original Imbalanced Split
# ====================================================================
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Define the models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                            random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                                     learning_rate=0.1, subsample=0.8, 
                                                     random_state=42)
}

results_imbalanced = {}
for name, model in models.items():
    print(f"Training {name} on imbalanced training data...")
    # Train on imbalanced scaled features (X_train_eng) and imbalanced target (y_train)
    model.fit(X_train_eng, y_train)
    
    # Predict on imbalanced test set (X_test_eng)
    y_pred = model.predict(X_test_eng)
    y_proba = model.predict_proba(X_test_eng)
    results_imbalanced[name] = evaluate_model(y_test, y_pred, y_proba)
"""
    },
    "training_balanced_code.png": {
        "title": "Training Three Algorithms on SMOTE-Balanced 80:20 Split",
        "code": """# ====================================================================
# Scenario 2: Model Training on SMOTE-Balanced Split
# ====================================================================
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Define the models (same hyperparameters as imbalanced run)
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                            random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                                     learning_rate=0.1, subsample=0.8, 
                                                     random_state=42)
}

results_balanced = {}
for name, model in models.items():
    print(f"Training {name} on SMOTE-balanced training data...")
    # Train on SMOTE balanced features (X_train_bal) and SMOTE balanced target (y_train_bal)
    model.fit(X_train_bal, y_train_bal)
    
    # Predict on imbalanced test set (X_test_eng) to reflect real-world testing
    y_pred = model.predict(X_test_eng)
    y_proba = model.predict_proba(X_test_eng)
    results_balanced[name] = evaluate_model(y_test, y_pred, y_proba)
"""
    }
}

def render_code_image(filename, title, code):
    # Setup styling: dark background VS Code theme
    fig, ax = plt.subplots(figsize=(10, len(code.split('\\n')) * 0.22 + 1.2), facecolor='#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    ax.axis('off')
    
    # Window decoration (simulated IDE header)
    ax.text(0.02, 0.96, title, color='#a0a0a0', fontsize=11, fontweight='bold', transform=ax.transAxes, fontfamily='sans-serif')
    # Draw simulated window controls
    fig.patches.extend([
        plt.Circle((0.92, 0.96), 0.008, color='#ff5f56', transform=fig.transFigure),
        plt.Circle((0.94, 0.96), 0.008, color='#ffbd2e', transform=fig.transFigure),
        plt.Circle((0.96, 0.96), 0.008, color='#27c93f', transform=fig.transFigure)
    ])
    
    # Draw separator line
    ax.plot([0, 1], [0.92, 0.92], color='#2d2d2d', linewidth=1, transform=ax.transAxes)
    
    # Render lines with line numbers
    lines = code.split('\n')
    y_pos = 0.86
    y_step = 0.82 / len(lines)
    
    for i, line in enumerate(lines):
        line_num = f"{i+1:2d} | "
        # Render line number (gray)
        ax.text(0.02, y_pos, line_num, color='#5a5a5a', fontsize=9.5, fontfamily='monospace', transform=ax.transAxes, va='top')
        # Render code text
        color = '#d4d4d4'
        if line.strip().startswith('#'):
            color = '#6a9955'  # green comment
        elif 'import ' in line or 'from ' in line or 'def ' in line or 'return ' in line or 'for ' in line or 'in ' in line or 'if ' in line:
            color = '#c586c0'  # purple keyword
        elif '=' in line:
            parts = line.split('=', 1)
            # Render left side (variable) as light blue
            # We keep it simple and just color keywords and comments for legibility
            pass
            
        ax.text(0.08, y_pos, line, color=color, fontsize=9.5, fontfamily='monospace', transform=ax.transAxes, va='top')
        y_pos -= y_step
        
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='#1e1e1e')
    plt.close()
    print(f"Generated screenshot: {filename}")

for fname, data in snippets.items():
    render_code_image(fname, data['title'], data['code'])
