"""
train_auditor.py
================
Trains the Trajectory Transformer auditor.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# --- Dynamic Environment Variables ---
DATASET_NAME = os.environ.get('DATASET_NAME', 'CIFAR10')
RUN_SEED     = int(os.environ.get('RUN_SEED', '42'))

sys.path.insert(0, os.path.dirname(__file__))
from src.auditor_arch import TrajectoryTransformer

# ─────────────────────────────────────────────────────────────────────────────
#  Config (Updated to 40 Epochs)
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_EPOCHS   = 40        
BATCH_SIZE     = 128
LEARNING_RATE  = 3e-4
WEIGHT_DECAY   = 1e-4      
GRAD_CLIP      = 1.0       
LABEL_SMOOTH   = 0.05      
INPUT_SIZE     = 33        
SEQ_LEN        = 12        

print(f"Loading trajectory data for {DATASET_NAME}...")
traj_dir = f'data/trajectories/{DATASET_NAME}'
m_trajs  = np.load(f'{traj_dir}/members.npy')
nm_trajs = np.load(f'{traj_dir}/non_members.npy')

X = np.concatenate([m_trajs, nm_trajs], axis=0)
y = np.concatenate([np.ones(len(m_trajs)), np.zeros(len(nm_trajs))], axis=0)

# ─────────────────────────────────────────────────────────────────────────────
#  Train / Val split (Using dynamic RUN_SEED)
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RUN_SEED, stratify=y
)

X_mean = np.mean(X_train, axis=(0, 1))   
X_std  = np.std(X_train,  axis=(0, 1))   

X_train_s = (X_train - X_mean) / (X_std + 1e-7)
X_val_s   = (X_val   - X_mean) / (X_std + 1e-7)

auditor_out_dir = f'models/auditors/{DATASET_NAME}'
os.makedirs(auditor_out_dir, exist_ok=True)

np.save(f'{auditor_out_dir}/X_mean.npy', X_mean)
np.save(f'{auditor_out_dir}/X_std.npy',  X_std)

def make_loader(X, y, shuffle):
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float().unsqueeze(1)
    return DataLoader(
        TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=True
    )

train_loader = make_loader(X_train_s, y_train, shuffle=True)
val_loader   = make_loader(X_val_s,   y_val,   shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nTraining on: {device}")

auditor = TrajectoryTransformer(
    input_size=INPUT_SIZE, d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.1, seq_len=SEQ_LEN,
).to(device)

class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        import torch.nn.functional as F
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy(pred, target)

criterion = LabelSmoothingBCE(smoothing=LABEL_SMOOTH)
optimizer = optim.AdamW(auditor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

history = {
    'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []
}
target_epochs = {10, 20, TOTAL_EPOCHS} # Adjusted snapshot epochs
margin_data   = {}

best_val_auc  = 0.0
best_ckpt     = f'{auditor_out_dir}/transformer_auditor_best.pth'

print(f"\nTraining Trajectory Transformer for {TOTAL_EPOCHS} epochs...\n")

for epoch in range(1, TOTAL_EPOCHS + 1):
    auditor.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0

    for trajs, labels in train_loader:
        trajs, labels = trajs.to(device), labels.to(device)
        optimizer.zero_grad()

        preds = auditor(trajs)
        loss  = criterion(preds, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(auditor.parameters(), GRAD_CLIP)
        optimizer.step()

        tr_loss    += loss.item()
        tr_correct += ((preds > 0.5).float() == labels).sum().item()
        tr_total   += labels.size(0)

    scheduler.step()

    avg_tr_loss = tr_loss / len(train_loader)
    tr_acc      = 100.0 * tr_correct / tr_total

    auditor.eval()
    va_loss, va_correct, va_total = 0.0, 0, 0
    all_probs, all_labels_ep = [], []

    with torch.no_grad():
        for trajs, labels in val_loader:
            trajs, labels = trajs.to(device), labels.to(device)
            preds = auditor(trajs)
            loss  = criterion(preds, labels)

            va_loss    += loss.item()
            va_correct += ((preds > 0.5).float() == labels).sum().item()
            va_total   += labels.size(0)

            all_probs.extend(preds.cpu().numpy().flatten())
            all_labels_ep.extend(labels.cpu().numpy().flatten())

    avg_va_loss = va_loss / len(val_loader)
    va_acc      = 100.0 * va_correct / va_total
    va_auc      = roc_auc_score(all_labels_ep, all_probs)
    current_lr  = scheduler.get_last_lr()[0]

    history['epoch'].append(epoch)
    history['train_loss'].append(avg_tr_loss)
    history['val_loss'].append(avg_va_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(va_acc)
    history['val_auc'].append(va_auc)

    if epoch in target_epochs:
        margin_data[epoch] = {'probs': np.array(all_probs), 'labels': np.array(all_labels_ep)}

    if va_auc > best_val_auc:
        best_val_auc = va_auc
        torch.save(auditor.state_dict(), best_ckpt)

    print(
        f"Epoch [{epoch:02d}/{TOTAL_EPOCHS}]  Loss (T/V): {avg_tr_loss:.4f}/{avg_va_loss:.4f}  "
        f"Acc (T/V): {tr_acc:.1f}%/{va_acc:.1f}%  AUC: {va_auc:.4f}  LR: {current_lr:.6f}"
        + ("  <-- best" if va_auc == best_val_auc else "")
    )

torch.save(auditor.state_dict(), f'{auditor_out_dir}/transformer_auditor_final.pth')
print(f"\n[DONE] Best model saved to {best_ckpt}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Trajectory Transformer ({DATASET_NAME} | Seed {RUN_SEED})', fontsize=13)

axes[0].plot(history['epoch'], history['train_acc'], label='Train', color='royalblue')
axes[0].plot(history['epoch'], history['val_acc'], label='Val', color='orange', linestyle='--')
axes[0].set_title('Accuracy'); axes[0].set_ylabel('Accuracy (%)'); axes[0].legend()

axes[1].plot(history['epoch'], history['train_loss'], label='Train', color='royalblue')
axes[1].plot(history['epoch'], history['val_loss'], label='Val', color='orange', linestyle='--')
axes[1].set_title('Loss'); axes[1].set_ylabel('BCE Loss'); axes[1].legend()

axes[2].plot(history['epoch'], history['val_auc'], color='green', label='Val AUC')
axes[2].axhline(y=best_val_auc, color='red', linestyle='--', label=f'Best = {best_val_auc:.4f}')
axes[2].set_title('Validation AUC'); axes[2].set_ylabel('ROC AUC'); axes[2].legend()

for ax in axes: ax.set_xlabel('Epoch')

plt.tight_layout()
plt.savefig(f'{auditor_out_dir}/training_curves_seed{RUN_SEED}.png', dpi=150)
plt.close()