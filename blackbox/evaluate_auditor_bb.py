"""
evaluate_auditor_bb.py
======================
Evaluates the Blackbox Auditor.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

DATASET_NAME = os.environ.get('DATASET_NAME', 'CIFAR10')
RUN_SEED     = int(os.environ.get('RUN_SEED', '42'))

sys.path.insert(0, os.path.dirname(__file__))
from src.auditor_arch import TrajectoryTransformer

INPUT_SIZE = 21 # Changed for Blackbox
SEQ_LEN    = 12
BATCH_SIZE = 128

AUDITOR_DIR = f'models/auditors_bb/{DATASET_NAME}'
MODEL_PATH  = f'{AUDITOR_DIR}/transformer_auditor_best.pth'
TRAJ_DIR    = f'data/trajectories_bb/{DATASET_NAME}'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m_trajs  = np.load(f'{TRAJ_DIR}/members.npy')
nm_trajs = np.load(f'{TRAJ_DIR}/non_members.npy')
X = np.concatenate([m_trajs, nm_trajs], axis=0)
y = np.concatenate([np.ones(len(m_trajs)), np.zeros(len(nm_trajs))], axis=0)

_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=RUN_SEED, stratify=y)

X_mean = np.load(f'{AUDITOR_DIR}/X_mean.npy')
X_std  = np.load(f'{AUDITOR_DIR}/X_std.npy')
X_val_s = (X_val - X_mean) / (X_std + 1e-7)

X_t = torch.from_numpy(X_val_s).float()
y_t = torch.from_numpy(y_val).float().unsqueeze(1)
val_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=False)

auditor = TrajectoryTransformer(
    input_size=INPUT_SIZE, d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.1, seq_len=SEQ_LEN,
).to(device)
auditor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
auditor.eval()

all_probs, all_preds, all_labels = [], [], []
with torch.no_grad():
    for trajs, labels in val_loader:
        trajs = trajs.to(device)
        probs = auditor(trajs)
        all_probs.extend(probs.cpu().numpy().flatten())
        all_preds.extend((probs > 0.5).float().cpu().numpy().flatten())
        all_labels.extend(labels.numpy().flatten())

acc = accuracy_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

print(f"\n[DONE] Blackbox Eval -> Accuracy: {acc*100:.2f}%, AUC: {auc:.4f}")