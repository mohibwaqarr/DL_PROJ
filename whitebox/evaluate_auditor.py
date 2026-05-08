"""
evaluate_auditor.py
===================
Loads the best-checkpoint Transformer auditor and evaluates it.
Dynamically handles dataset paths and random seeds.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# --- Dynamic Environment Variables ---
DATASET_NAME = os.environ.get('DATASET_NAME', 'CIFAR10')
RUN_SEED     = int(os.environ.get('RUN_SEED', '42'))

sys.path.insert(0, os.path.dirname(__file__))
from src.auditor_arch import TrajectoryTransformer

# ─────────────────────────────────────────────────────────────────────────────
#  Config 
# ─────────────────────────────────────────────────────────────────────────────
INPUT_SIZE = 33
SEQ_LEN    = 12
BATCH_SIZE = 128

AUDITOR_DIR = f'models/auditors/{DATASET_NAME}'
MODEL_PATH  = f'{AUDITOR_DIR}/transformer_auditor_best.pth'
SCALER_DIR  = AUDITOR_DIR
TRAJ_DIR    = f'data/trajectories/{DATASET_NAME}'

# ─────────────────────────────────────────────────────────────────────────────
#  1. Load & split data
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running evaluation on: {device} for {DATASET_NAME} (Seed: {RUN_SEED})")

m_trajs  = np.load(f'{TRAJ_DIR}/members.npy')
nm_trajs = np.load(f'{TRAJ_DIR}/non_members.npy')

X = np.concatenate([m_trajs, nm_trajs], axis=0)
y = np.concatenate([np.ones(len(m_trajs)), np.zeros(len(nm_trajs))], axis=0)

# Using the dynamic RUN_SEED for variance sampling
_, X_val, _, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RUN_SEED, stratify=y
)

# ─────────────────────────────────────────────────────────────────────────────
#  2. Apply the training scaler
# ─────────────────────────────────────────────────────────────────────────────
for fname in ('X_mean.npy', 'X_std.npy'):
    if not os.path.exists(os.path.join(SCALER_DIR, fname)):
        raise FileNotFoundError(
            f"Scaler file '{fname}' not found in {SCALER_DIR}. Run train_auditor.py first."
        )

X_mean = np.load(os.path.join(SCALER_DIR, 'X_mean.npy'))
X_std  = np.load(os.path.join(SCALER_DIR, 'X_std.npy'))

X_val_s = (X_val - X_mean) / (X_std + 1e-7)

X_t = torch.from_numpy(X_val_s).float()
y_t = torch.from_numpy(y_val).float().unsqueeze(1)
val_loader = DataLoader(
    TensorDataset(X_t, y_t),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ─────────────────────────────────────────────────────────────────────────────
#  3. Load the best-checkpoint Transformer
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. Run train_auditor.py first."
    )

auditor = TrajectoryTransformer(
    input_size=INPUT_SIZE,
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_ff=256,
    dropout=0.1,
    seq_len=SEQ_LEN,
).to(device)

auditor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
auditor.eval()
print(f"Model loaded from {MODEL_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
#  4. Inference
# ─────────────────────────────────────────────────────────────────────────────
all_probs, all_preds, all_labels = [], [], []

print("Auditing unseen trajectories...")
with torch.no_grad():
    for trajs, labels in val_loader:
        trajs  = trajs.to(device)
        probs  = auditor(trajs)
        preds  = (probs > 0.5).float()

        all_probs.extend(probs.cpu().numpy().flatten())
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.numpy().flatten())

all_probs  = np.array(all_probs)
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ─────────────────────────────────────────────────────────────────────────────
#  5. Metrics
# ─────────────────────────────────────────────────────────────────────────────
acc   = accuracy_score(all_labels,  all_preds)
prec  = precision_score(all_labels, all_preds)
rec   = recall_score(all_labels,    all_preds)
f1    = f1_score(all_labels,        all_preds)
auc   = roc_auc_score(all_labels,   all_probs)

print("\n" + "="*45)
print(f"    TRAJECTORY TRANSFORMER AUDIT RESULTS ({DATASET_NAME})    ")
print("="*45)
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1-Score  : {f1*100:.2f}%")
print(f"  ROC AUC   : {auc:.4f}")
print("="*45)

# ─────────────────────────────────────────────────────────────────────────────
#  6. Visualisations
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Trajectory Transformer ({DATASET_NAME} | Seed {RUN_SEED}) — Audit Results', fontsize=13)

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
    xticklabels=['Non-Member', 'Member'],
    yticklabels=['Non-Member', 'Member'],
    annot_kws={'size': 13}
)
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix')

fpr, tpr, _ = roc_curve(all_labels, all_probs)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.4f}')
axes[1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='darkorange')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend(loc='lower right')

sns.kdeplot(all_probs[all_labels == 1], fill=True, color='royalblue', label='Members', ax=axes[2], alpha=0.7)
sns.kdeplot(all_probs[all_labels == 0], fill=True, color='tomato', label='Non-Members', ax=axes[2], alpha=0.7)
axes[2].axvline(x=0.5, color='black', linestyle='--', lw=1, label='Threshold=0.5')
axes[2].set_xlabel('Predicted Membership Probability')
axes[2].set_title('Score Distributions')
axes[2].legend()

plt.tight_layout()
# Appending the seed so it doesn't overwrite plots between the 3 runs
plot_path = f'results/audit_results_{DATASET_NAME}_seed{RUN_SEED}.png'
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\n[DONE] Plots saved to {plot_path}")