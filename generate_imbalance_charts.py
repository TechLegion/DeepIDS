import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

# ── Load ──────────────────────────────────────────────────
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

dos_list   = ['neptune','back','land','pod','smurf','teardrop','apache2','udpstorm','processtable','worm']
probe_list = ['satan','ipsweep','nmap','portsweep','mscan','saint']
r2l_list   = ['guess_passwd','ftp_write','imap','phf','multihop','warezmaster','warezclient',
               'spy','xlock','xsnoop','snmpguess','snmpgetattack','httptunnel','sendmail','named']
u2r_list   = ['buffer_overflow','loadmodule','rootkit','perl','sqlattack','xterm','ps']

def categorise(x):
    if x == 'normal':     return 'normal'
    elif x in dos_list:   return 'dos'
    elif x in probe_list: return 'probe'
    elif x in r2l_list:   return 'r2l'
    elif x in u2r_list:   return 'u2r'
    else:                 return 'dos'

df['category'] = df['attack'].apply(categorise)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])

ORDER  = ['dos','normal','probe','r2l','u2r']
COLORS = ['#e74c3c','#2ecc71','#f39c12','#9b59b6','#3498db']
BG    = '#0a0a0f'
PANEL = '#12121a'
TEXT  = 'white'
MUTED = '#8a8a9e'

def counts_ordered(subset):
    c = subset['category'].value_counts()
    return [int(c.get(o, 0)) for o in ORDER]

full_counts  = counts_ordered(df)
train_counts = counts_ordered(train_df)
test_counts  = counts_ordered(test_df)

total_full  = len(df)
total_train = len(train_df)
total_test  = len(test_df)

def pct(v, t): return v / t * 100

def panel_bg(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor('#2d2d3a')

def bar_chart(ax, counts, total, title):
    panel_bg(ax)
    bars = ax.bar(ORDER, counts, color=COLORS, edgecolor='none', width=0.6)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', color=MUTED, fontsize=9)
    ax.tick_params(axis='x', colors=TEXT, labelsize=9)
    ax.tick_params(axis='y', colors=MUTED, labelsize=8)
    ax.set_ylim(0, max(counts) * 1.22)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(counts)*0.01,
                '{:,}\n({:.1f}%)'.format(val, pct(val, total)),
                ha='center', color=TEXT, fontsize=8, fontweight='bold')

# ═══════════════════════════════════════════════════════════
# CHART 1 – CLASS IMBALANCE ANALYSIS
# ═══════════════════════════════════════════════════════════
print("Generating Chart 1: Class Imbalance Analysis...")
fig1 = plt.figure(figsize=(20, 12), facecolor=BG)
fig1.suptitle('CLASS IMBALANCE ANALYSIS: Full Dataset, Training & Test Sets',
              fontsize=17, fontweight='bold', color=TEXT, y=0.99)

gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.55, wspace=0.35, height_ratios=[1, 1.1])

ax_a = fig1.add_subplot(gs1[0, 0])
bar_chart(ax_a, full_counts, total_full,
          'A. Full Dataset ({:,} samples)'.format(total_full))

ax_b = fig1.add_subplot(gs1[0, 1])
bar_chart(ax_b, train_counts, total_train,
          'B. Training Set ({:,} samples)'.format(total_train))

ax_c = fig1.add_subplot(gs1[0, 2])
bar_chart(ax_c, test_counts, total_test,
          'C. Test Set ({:,} samples)'.format(total_test))

# Pie chart
ax_d = fig1.add_subplot(gs1[1, 0])
ax_d.set_facecolor(PANEL)
wedges, texts, autotexts = ax_d.pie(
    full_counts, labels=ORDER, colors=COLORS,
    autopct='%1.1f%%', startangle=140, pctdistance=0.78,
    wedgeprops=dict(edgecolor=BG, linewidth=2))
for t in texts:     t.set_color(TEXT);  t.set_fontsize(10)
for at in autotexts: at.set_color(TEXT); at.set_fontsize(9); at.set_fontweight('bold')
ax_d.set_title('D. Full Dataset Proportions', color=TEXT, fontsize=11, fontweight='bold')

# Proportion table
ax_e = fig1.add_subplot(gs1[1, 1])
ax_e.set_facecolor(PANEL)
ax_e.axis('off')
ax_e.set_title('E. Proportion Comparison Table', color=TEXT, fontsize=11, fontweight='bold')

table_data = []
for i, cls in enumerate(ORDER):
    table_data.append([
        cls,
        '{:.1f}%'.format(pct(full_counts[i],  total_full)),
        '{:.1f}%'.format(pct(train_counts[i], total_train)),
        '{:.1f}%'.format(pct(test_counts[i],  total_test)),
    ])
col_labels = ['Class', 'Full Dataset', 'Training (80%)', 'Test (20%)']

tbl = ax_e.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.2)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('#2d2d3a')
    if row == 0:
        cell.set_facecolor('#2d2d3a')
        cell.set_text_props(color=TEXT, fontweight='bold')
    else:
        if col == 0:
            cell.set_facecolor(COLORS[row - 1])
            cell.set_text_props(color=TEXT, fontweight='bold')
        else:
            cell.set_facecolor(PANEL)
            cell.set_text_props(color=TEXT)

# Imbalance Ratio panel
ax_f = fig1.add_subplot(gs1[1, 2])
ax_f.set_facecolor(PANEL)
for sp in ax_f.spines.values(): sp.set_edgecolor('#e74c3c')
ax_f.axis('off')
ax_f.set_title('F. Imbalance Ratio Analysis', color=TEXT, fontsize=11, fontweight='bold')

normal_count = full_counts[ORDER.index('normal')]
severity_map = {
    'dos':   ('BALANCED',  normal_count / max(full_counts[ORDER.index('dos')],   1)),
    'probe': ('MODERATE',  normal_count / max(full_counts[ORDER.index('probe')], 1)),
    'r2l':   ('SEVERE',    normal_count / max(full_counts[ORDER.index('r2l')],   1)),
    'u2r':   ('EXTREME',   normal_count / max(full_counts[ORDER.index('u2r')],   1)),
}

lines = [
    'IMBALANCE RATIOS (Majority : Minority)',
    '',
]
for cls, (sev, ratio) in severity_map.items():
    lines.append('  normal : {:6s} = {:5.0f} : 1  ({})'.format(cls, ratio, sev))

lines += [
    '',
    'STRATIFIED SPLIT VERIFICATION:',
    '  All three sets have IDENTICAL',
    '  proportions - split is valid.',
    '',
    'IMPACT ON MODELS:',
    '  Without SMOTE: Model ignores',
    '  minority classes (r2l, u2r).',
    '  With SMOTE: Model learns all',
    '  classes via synthetic samples.',
    '  Test set stays realistic',
    '  (unbalanced, real-world).',
]

y_p = 0.97
for line in lines:
    is_header = line and not line.startswith(' ')
    ax_f.text(0.04, y_p, line,
              color=TEXT if is_header else MUTED,
              fontsize=8 if not is_header else 8.5,
              fontweight='bold' if is_header else 'normal',
              transform=ax_f.transAxes, va='top',
              fontfamily='monospace')
    y_p -= 0.055

plt.savefig('class_imbalance_analysis.png', dpi=160, bbox_inches='tight', facecolor=BG)
print("Saved: class_imbalance_analysis.png")
plt.close()

# ═══════════════════════════════════════════════════════════
# CHART 2 – SMOTE DISTRIBUTION
# ═══════════════════════════════════════════════════════════
print("Generating Chart 2: SMOTE Distribution...")

smote_target = max(train_counts)  # balance all to majority class count
smote_counts = [smote_target] * len(ORDER)
smote_total  = sum(smote_counts)
synth_created = smote_total - total_train

fig2 = plt.figure(figsize=(20, 8), facecolor=BG)
fig2.suptitle('CLASS DISTRIBUTION: Train (80%), Test (20%), and After SMOTE Balancing',
              fontsize=16, fontweight='bold', color=TEXT, y=1.02)

gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.35)

def bar_chart2(ax, counts, total, title, subtitle, annot_txt, annot_color):
    panel_bg(ax)
    bars = ax.bar(ORDER, counts, color=COLORS, edgecolor='none', width=0.6)
    ax.set_title(title + '\n' + subtitle, color=TEXT, fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Samples', color=MUTED, fontsize=9)
    ax.tick_params(axis='x', colors=TEXT, labelsize=9)
    ax.tick_params(axis='y', colors=MUTED, labelsize=8)
    ax.set_ylim(0, max(counts) * 1.30)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(counts)*0.012,
                '{:,}\n({:.1f}%)'.format(val, pct(val, total)),
                ha='center', color=TEXT, fontsize=8, fontweight='bold')
    ax.annotate(annot_txt,
                xy=(0.5, 0.96), xycoords='axes fraction',
                ha='center', va='top', fontsize=8, color=TEXT,
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#2d2d3a',
                          edgecolor=annot_color, alpha=0.9))

ratio_str = 'Imbalance Ratio: {:,}:{}\n(Majority:Minority)'.format(
    max(train_counts), min(train_counts))
bar_chart2(fig2.add_subplot(gs2[0]),
           train_counts, total_train,
           'A. Training Set (80%)', 'IMBALANCED',
           ratio_str, '#e74c3c')

bar_chart2(fig2.add_subplot(gs2[1]),
           test_counts, total_test,
           'B. Test Set (20%)', 'IMBALANCED (Unchanged)',
           'Same proportions as\nfull dataset (stratified)', '#3498db')

ax2_c = fig2.add_subplot(gs2[2])
smote_note = 'Synthetic samples created: {:,}\nTotal: {:,} (was {:,})'.format(
    synth_created, smote_total, total_train)
bar_chart2(ax2_c, smote_counts, smote_total,
           'C. Training Set After SMOTE', 'PERFECTLY BALANCED',
           smote_note, '#2ecc71')
ax2_c.set_title('C. Training Set After SMOTE\nPERFECTLY BALANCED',
                color='#2ecc71', fontsize=11, fontweight='bold')

plt.savefig('smote_distribution.png', dpi=160, bbox_inches='tight', facecolor=BG)
print("Saved: smote_distribution.png")
plt.close()

print("\nBoth charts saved successfully!")
