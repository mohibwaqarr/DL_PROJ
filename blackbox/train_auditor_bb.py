"""
train_auditor_bb.py
===================
Trains the Trajectory Transformer auditor on BLACKBOX trajectories.
Input size is reduced to 21 features (7 features x 3 scales).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

DATASET_NAME = os.environ.get('DATASET_NAME', 'CIFAR10')
RUN_SEED     = int(os.environ.get('RUN_SEED', '42'))

sys.path.insert(0, os.path.dirname(__file__))
from src.auditor_arch import TrajectoryTransformer

TOTAL_EPOCHS   = 40
BATCH_SIZE     = 128
LEARNING_RATE  = 3e-4
WEIGHT_DECAY   = 1e-4      
INPUT_SIZE     = 21   # CHANGED: 7 features * 3 scales
SEQ_LEN        = 12        

print(f"Loading BLACKBOX trajectory data for {DATASET_NAME}...")
traj_dir = f'data/trajectories_bb/{DATASET_NAME}'
m_trajs  = np.load(f'{traj_dir}/members.npy')
nm_trajs = np.load(f'{traj_dir}/non_members.npy')

X = np.concatenate([m_trajs, nm_trajs], axis=0)
y = np.concatenate([np.ones(len(m_trajs)), np.zeros(len(nm_trajs))], axis=0)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RUN_SEED, stratify=y
)

X_mean = np.mean(X_train, axis=(0, 1))   
X_std  = np.std(X_train,  axis=(0, 1))   
X_train_s = (X_train - X_mean) / (X_std + 1e-7)
X_val_s   = (X_val   - X_mean) / (X_std + 1e-7)

auditor_out_dir = f'models/auditors_bb/{DATASET_NAME}'
os.makedirs(auditor_out_dir, exist_ok=True)
np.save(f'{auditor_out_dir}/X_mean.npy', X_mean)
np.save(f'{auditor_out_dir}/X_std.npy',  X_std)

def make_loader(X, y, shuffle):
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float().unsqueeze(1)
    return DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train_s, y_train, shuffle=True)
val_loader   = make_loader(X_val_s,   y_val,   shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

auditor = TrajectoryTransformer(
    input_size=INPUT_SIZE, d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.1, seq_len=SEQ_LEN,
).to(device)

criterion = nn.BCELoss()
optimizer = optim.AdamW(auditor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

best_val_auc  = 0.0
best_ckpt     = f'{auditor_out_dir}/transformer_auditor_best.pth'

for epoch in range(1, TOTAL_EPOCHS + 1):
    auditor.train()
    for trajs, labels in train_loader:
        trajs, labels = trajs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = auditor(trajs)
        loss  = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(auditor.parameters(), 1.0)
        optimizer.step()

    scheduler.step()

    auditor.eval()
    all_probs, all_labels_ep = [], []
    with torch.no_grad():
        for trajs, labels in val_loader:
            trajs = trajs.to(device)
            preds = auditor(trajs)
            all_probs.extend(preds.cpu().numpy().flatten())
            all_labels_ep.extend(labels.numpy().flatten())

    va_auc = roc_auc_score(all_labels_ep, all_probs)
    if va_auc > best_val_auc:
        best_val_auc = va_auc
        torch.save(auditor.state_dict(), best_ckpt)

print(f"\n[DONE] Blackbox Auditor best AUC ({best_val_auc:.4f}) saved to {best_ckpt}")