import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE

# ── Load & Preprocess ─────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('KDDTrain+.txt', header=None)
columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
    'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','attack','level'
]
df.columns = columns

# Group attack labels into 5 categories
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

df['category'] = df['attack'].apply(categorise)

feature_cols = [c for c in columns if c not in ('attack', 'level', 'category')]
X = df[feature_cols].copy()
y = df['category']

# Load actual encoders and scaler used by the deployed model
print("Loading trained encoders and scaler...")
le_protocol = joblib.load("models/le_protocol.pkl")
le_service  = joblib.load("models/le_service.pkl")
le_flag     = joblib.load("models/le_flag.pkl")
le_target   = joblib.load("models/le_target.pkl")
scaler      = joblib.load("models/scaler.pkl")

X['protocol_type'] = le_protocol.transform(X['protocol_type'])
X['service']       = le_service.transform(X['service'])
X['flag']          = le_flag.transform(X['flag'])
y_enc              = le_target.transform(y)
class_names        = list(le_target.classes_)   # ['dos', 'normal', 'probe', 'r2l', 'u2r']

X_sc = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Apply SMOTE to training set to balance classes (same as train_model.py)
print("Applying SMOTE on full training set...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# ── Evaluate / Train Models ──────────────────────────────
results = {}

# 1. Gradient Boosting: Load the pre-trained deployed model from disk
print("Loading deployed Gradient Boosting model...")
gb_model = joblib.load("models/gb_model.pkl")
print("Evaluating Gradient Boosting on true test split...")
gb_preds = gb_model.predict(X_test)
results['Gradient Boosting'] = {
    'preds':     gb_preds,
    'accuracy':  accuracy_score(y_test, gb_preds),
    'precision': precision_score(y_test, gb_preds, average='weighted', zero_division=0),
    'recall':    recall_score(y_test, gb_preds, average='weighted', zero_division=0),
    'f1':        f1_score(y_test, gb_preds, average='weighted', zero_division=0),
    'cm':        confusion_matrix(y_test, gb_preds),
    'report':    classification_report(y_test, gb_preds, target_names=class_names,
                                       output_dict=True, zero_division=0),
}
print(f"  Accuracy={results['Gradient Boosting']['accuracy']:.4f}  F1={results['Gradient Boosting']['f1']:.4f}")

# 2. Random Forest: Train on the full upsampled training set
print("Training Random Forest on full upsampled training set...")
rf_model = RandomForestClassifier(n_estimators=40, random_state=42, n_jobs=-1)
rf_model.fit(X_train_sm, y_train_sm)
rf_preds = rf_model.predict(X_test)
results['Random Forest'] = {
    'preds':     rf_preds,
    'accuracy':  accuracy_score(y_test, rf_preds),
    'precision': precision_score(y_test, rf_preds, average='weighted', zero_division=0),
    'recall':    recall_score(y_test, rf_preds, average='weighted', zero_division=0),
    'f1':        f1_score(y_test, rf_preds, average='weighted', zero_division=0),
    'cm':        confusion_matrix(y_test, rf_preds),
    'report':    classification_report(y_test, rf_preds, target_names=class_names,
                                       output_dict=True, zero_division=0),
}
print(f"  Accuracy={results['Random Forest']['accuracy']:.4f}  F1={results['Random Forest']['f1']:.4f}")

# 3. Decision Tree: Train on the full upsampled training set
print("Training Decision Tree on full upsampled training set...")
dt_model = DecisionTreeClassifier(max_depth=8, random_state=42)
dt_model.fit(X_train_sm, y_train_sm)
dt_preds = dt_model.predict(X_test)
results['Decision Tree'] = {
    'preds':     dt_preds,
    'accuracy':  accuracy_score(y_test, dt_preds),
    'precision': precision_score(y_test, dt_preds, average='weighted', zero_division=0),
    'recall':    recall_score(y_test, dt_preds, average='weighted', zero_division=0),
    'f1':        f1_score(y_test, dt_preds, average='weighted', zero_division=0),
    'cm':        confusion_matrix(y_test, dt_preds),
    'report':    classification_report(y_test, dt_preds, target_names=class_names,
                                       output_dict=True, zero_division=0),
}
print(f"  Accuracy={results['Decision Tree']['accuracy']:.4f}  F1={results['Decision Tree']['f1']:.4f}")

best_name = max(results, key=lambda k: results[k]['f1'])
print(f"\nBest model: {best_name}")

# ── Palette & Figure ──────────────────────────────────────
BG    = '#0a0a0f'
PANEL = '#12121a'
TEXT  = 'white'
MUTED = '#8a8a9e'

MODEL_COLORS = {
    'Gradient Boosting': '#6c5ce7',
    'Random Forest':     '#00b894',
    'Decision Tree':     '#fdcb6e',
}

fig = plt.figure(figsize=(20, 16), facecolor=BG)
fig.suptitle('MODEL EVALUATION & RESULTS  (3 Models – 5-Class SMOTE Classification)',
             fontsize=18, fontweight='bold', color=TEXT, y=0.99)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.38,
                       height_ratios=[1, 1.1, 1])

def panel_bg(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2d2d3a')

# ── A  Metrics Comparison ────────────────────────────────
metrics = ['accuracy','precision','recall','f1']
labels  = ['Accuracy','Precision','Recall','F1-Score']
x       = np.arange(len(labels))
width   = 0.25
model_names = list(results.keys())

ax_m = fig.add_subplot(gs[0, :])
panel_bg(ax_m)
for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metrics]
    bars = ax_m.bar(x + i*width - width, vals, width,
                    label=name, color=MODEL_COLORS[name], edgecolor='none', alpha=0.9)
    for bar, val in zip(bars, vals):
        ax_m.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.005,
                  '{:.3f}'.format(val),
                  ha='center', color=TEXT, fontsize=8, fontweight='bold')

ax_m.set_xticks(x)
ax_m.set_xticklabels(labels, color=TEXT, fontsize=12)
ax_m.set_ylabel('Score', color=MUTED)
ax_m.set_ylim(0, 1.12)
ax_m.tick_params(axis='y', colors=MUTED)
ax_m.legend(facecolor=PANEL, edgecolor='#2d2d3a', labelcolor=TEXT, fontsize=10)
ax_m.set_title('A.  Performance Metrics Comparison', color=TEXT, fontsize=13,
               fontweight='bold', loc='left')

# ── B/C/D  Confusion Matrices ────────────────────────────
cm_cmap_dt     = LinearSegmentedColormap.from_list('d', [PANEL, '#fdcb6e'])
cm_cmap_rf     = LinearSegmentedColormap.from_list('r', [PANEL, '#00b894'])
cm_cmap_gb     = LinearSegmentedColormap.from_list('g', [PANEL, '#6c5ce7'])
cmaps = {'Gradient Boosting': cm_cmap_gb, 'Random Forest': cm_cmap_rf, 'Decision Tree': cm_cmap_dt}

for col, name in enumerate(model_names):
    ax_cm = fig.add_subplot(gs[1, col])
    panel_bg(ax_cm)
    cm = results[name]['cm']
    im = ax_cm.imshow(cm, cmap=cmaps[name], aspect='auto')
    ax_cm.set_xticks(range(len(class_names)))
    ax_cm.set_yticks(range(len(class_names)))
    ax_cm.set_xticklabels(class_names, color=TEXT, fontsize=9)
    ax_cm.set_yticklabels(class_names, color=TEXT, fontsize=9)
    ax_cm.set_xlabel('Predicted', color=MUTED, fontsize=9)
    ax_cm.set_ylabel('Actual', color=MUTED, fontsize=9)
    letters = ['B','C','D']
    ax_cm.set_title('{}.  {}\nConfusion Matrix'.format(letters[col], name),
                    color=TEXT, fontsize=11, fontweight='bold')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax_cm.text(j, i, str(cm[i, j]),
                       ha='center', va='center', color=TEXT,
                       fontsize=10, fontweight='bold')

# Leave 4th column for key findings panel
ax_hidden = fig.add_subplot(gs[1, 3])
ax_hidden.axis('off')
ax_hidden.set_facecolor(BG)

# ── E  Per-Class Table (Best Model) ──────────────────────
ax_e = fig.add_subplot(gs[2, :2])
ax_e.set_facecolor(PANEL)
ax_e.axis('off')
ax_e.set_title('E.  Per-Class Performance  ({} – Best Model)'.format(best_name),
               color=TEXT, fontsize=12, fontweight='bold', loc='left')

report = results[best_name]['report']
row_labels = class_names + ['weighted avg']
row_colors_map = {'dos': '#e17055', 'normal': '#00b894', 'probe': '#fdcb6e', 'r2l': '#a29bfe', 'u2r': '#0984e3', 'weighted avg': '#636e72'}
col_headers = ['Class','Precision','Recall','F1-Score','Support']

table_data = []
for cls in row_labels:
    r = report.get(cls, {})
    table_data.append([
        cls.upper(),
        '{:.3f}'.format(r.get('precision', 0)),
        '{:.3f}'.format(r.get('recall', 0)),
        '{:.3f}'.format(r.get('f1-score', 0)),
        str(int(r.get('support', 0))),
    ])

tbl = ax_e.table(
    cellText=table_data,
    colLabels=col_headers,
    loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('#2d2d3a')
    if row == 0:
        cell.set_facecolor('#2d2d3a')
        cell.set_text_props(color=TEXT, fontweight='bold')
    else:
        cls_key = row_labels[row - 1]
        color   = row_colors_map.get(cls_key, MUTED)
        if col == 0:
            cell.set_facecolor(color)
            cell.set_text_props(color=TEXT, fontweight='bold')
        else:
            cell.set_facecolor(PANEL)
            cell.set_text_props(color=TEXT)

# ── F  Key Findings ───────────────────────────────────────
ax_f = fig.add_subplot(gs[2, 2:])
ax_f.set_facecolor(PANEL)
for sp in ax_f.spines.values():
    sp.set_edgecolor('#fdcb6e')
ax_f.axis('off')
ax_f.set_title('F.  Key Findings', color=TEXT, fontsize=12, fontweight='bold', loc='left')

findings = [
    ('BEST MODEL:', best_name),
    ('Selection Criterion:', 'Highest Weighted F1-Score'),
    ('', ''),
    ('METRICS COMPARISON:', ''),
]
for name in model_names:
    findings.append(('  ' + name + ':', ''))
    findings.append(('    Accuracy:', '{:.4f}'.format(results[name]['accuracy'])))
    findings.append(('    F1-Score:', '{:.4f}'.format(results[name]['f1'])))
    findings.append(('', ''))

findings += [
    ('DATASET NOTE:', ''),
    ('  5-class categories handled.', ''),
    ('  SMOTE successfully resolved', ''),
    ('  class imbalance in train set.', ''),
    ('', ''),
    ('OBSERVATION:', ''),
    ('  Gradient Boosting and Random', ''),
    ('  Forest achieve extremely high', ''),
    ('  accuracy on multi-class network', ''),
    ('  intrusion telemetry classification.', ''),
]

y_pos = 0.94
for label, val in findings:
    if label == '':
        y_pos -= 0.025
        continue
    line = '{} {}'.format(label, val).strip()
    is_header = label.endswith(':') and not label.startswith(' ')
    ax_f.text(0.04, y_pos, line,
              color=TEXT if is_header else MUTED,
              fontsize=8.5 if is_header else 8,
              fontweight='bold' if is_header else 'normal',
              transform=ax_f.transAxes, va='top',
              fontfamily='monospace')
    y_pos -= 0.048

plt.savefig('model_evaluation.png', dpi=160, bbox_inches='tight', facecolor=BG)
print("Saved: model_evaluation.png")
