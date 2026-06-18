import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Load dataset
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

# Binary label
df['label'] = df['attack'].apply(lambda x: 'Normal' if x == 'normal' else 'Attack')

# Attack categories
dos_list    = ['neptune','back','land','pod','smurf','teardrop','apache2','udpstorm','processtable','worm']
probe_list  = ['satan','ipsweep','nmap','portsweep','mscan','saint']
r2l_list    = ['guess_passwd','ftp_write','imap','phf','multihop','warezmaster','warezclient','spy','xlock','xsnoop','snmpguess','snmpgetattack','httptunnel','sendmail','named']
u2r_list    = ['buffer_overflow','loadmodule','rootkit','perl','sqlattack','xterm','ps']

def categorise(x):
    if x == 'normal':    return 'Normal'
    elif x in dos_list:  return 'DoS'
    elif x in probe_list:return 'Probe'
    elif x in r2l_list:  return 'R2L'
    elif x in u2r_list:  return 'U2R'
    else:                return 'Other'

df['category'] = df['attack'].apply(categorise)

# ── Palette ───────────────────────────────────────────────
BG    = '#0a0a0f'
PANEL = '#12121a'
TEXT  = 'white'
MUTED = '#8a8a9e'
COLS  = ['#6c5ce7','#00b894','#fd79a8','#fdcb6e','#74b9ff','#a29bfe']

# ── Figure ────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle('DATASET OVERVIEW: NSL-KDD (KDDTrain+)',
             fontsize=20, fontweight='bold', color=TEXT, y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

def panel_bg(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2d2d3a')

counts    = df['label'].value_counts()
n_normal  = int(counts.get('Normal', 0))
n_attack  = int(counts.get('Attack', 0))

# ── 1  Binary Donut ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
panel_bg(ax1)
wedges, texts, autotexts = ax1.pie(
    counts, labels=counts.index, autopct='%1.1f%%',
    colors=['#00b894','#ff7675'], startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=3))
for t  in texts:     t.set_color(TEXT);  t.set_fontsize(12)
for at in autotexts: at.set_color(TEXT); at.set_fontweight('bold'); at.set_fontsize(11)
ax1.set_title('Binary Class Distribution', color=TEXT, fontsize=13, fontweight='bold', pad=15)
ax1.text(0, -1.3,
         'Normal: {:,}   Attack: {:,}'.format(n_normal, n_attack),
         ha='center', color=MUTED, fontsize=10)

# ── 2  Attack Category Bar ───────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
panel_bg(ax2)
cat_counts = df['category'].value_counts()
bars2 = ax2.barh(cat_counts.index, cat_counts.values,
                 color=COLS[:len(cat_counts)], edgecolor='none', height=0.6)
ax2.set_title('Attack Category Distribution', color=TEXT, fontsize=13, fontweight='bold')
ax2.set_xlabel('Count', color=MUTED, fontsize=10)
for bar, val in zip(bars2, cat_counts.values):
    ax2.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
             '{:,}'.format(val), va='center', color=TEXT, fontsize=9)
ax2.set_xlim(0, cat_counts.max() * 1.18)
ax2.tick_params(axis='x', colors=MUTED)
ax2.tick_params(axis='y', colors=TEXT)

# ── 3  Protocol Distribution ─────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
panel_bg(ax3)
proto = df['protocol_type'].value_counts()
bars3 = ax3.bar(proto.index, proto.values,
                color=['#6c5ce7','#00b894','#fd79a8'], edgecolor='none', width=0.5)
ax3.set_title('Protocol Type Distribution', color=TEXT, fontsize=13, fontweight='bold')
ax3.set_ylabel('Count', color=MUTED, fontsize=10)
for bar, val in zip(bars3, proto.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
             '{:,}'.format(val), ha='center', color=TEXT, fontsize=10, fontweight='bold')
ax3.tick_params(axis='x', colors=TEXT)
ax3.tick_params(axis='y', colors=MUTED)
ax3.set_ylim(0, proto.max() * 1.15)

# ── 4  Top Attack Types ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
panel_bg(ax4)
top_attacks = df['attack'].value_counts().head(11)
colors4 = ['#00b894' if i == 0 else '#ff7675' if i < 4 else '#fdcb6e' if i < 7 else '#6c5ce7'
           for i in range(len(top_attacks))]
bars4 = ax4.bar(top_attacks.index, top_attacks.values,
                color=colors4, edgecolor='none', width=0.65)
ax4.set_title('Top Attack Types (+ Normal)', color=TEXT, fontsize=13, fontweight='bold')
ax4.tick_params(axis='x', rotation=30, colors=TEXT, labelsize=9)
ax4.tick_params(axis='y', colors=MUTED)
ax4.set_ylabel('Count', color=MUTED, fontsize=10)
for bar, val in zip(bars4, top_attacks.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             '{:,}'.format(val), ha='center', color=TEXT, fontsize=8.5)
ax4.set_ylim(0, top_attacks.max() * 1.12)

# ── 5  Info Panel ────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(PANEL)
for sp in ax5.spines.values(): sp.set_edgecolor('#6c5ce7')
ax5.axis('off')

info = [
    ('DATASET',         'NSL-KDD (KDDTrain+)'),
    ('SOURCE',          'Univ. of New Brunswick'),
    ('TOTAL RECORDS',   '{:,}'.format(len(df))),
    ('TOTAL FEATURES',  '41 (+ binary label)'),
    ('NORMAL RECORDS',  '{:,} (53.5%)'.format(n_normal)),
    ('ATTACK RECORDS',  '{:,} (46.5%)'.format(n_attack)),
    ('MISSING VALUES',  '0'),
    ('DUPLICATES',      '0 (vs KDD99)'),
    ('ATTACK CLASSES',  '4  (DoS, Probe, R2L, U2R)'),
    ('MODEL TASK',      '5-Class Classification'),
    ('BALANCE STATUS',  'IMBALANCED (SMOTE applied)'),
]
y_pos = 0.96
for label, val in info:
    ax5.text(0.03, y_pos,        label + ':',
             color=MUTED, fontsize=8.5, transform=ax5.transAxes, va='top')
    ax5.text(0.03, y_pos - 0.045, val,
             color=TEXT, fontsize=9.5, fontweight='bold', transform=ax5.transAxes, va='top')
    y_pos -= 0.088
ax5.set_title('Dataset Summary', color=TEXT, fontsize=13, fontweight='bold')

plt.savefig('dataset_overview.png', dpi=180, bbox_inches='tight', facecolor=BG)
print('Saved: dataset_overview.png')
