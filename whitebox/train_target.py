"""
train_target.py
===============
Trains the ResNet-50 "victim" model on the member split of the specified dataset.
Dynamically handles datasets via environment variables.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt

# --- Dynamic Environment Variables ---
DATASET_NAME = os.environ.get('DATASET_NAME', 'CIFAR10')
NUM_CLASSES = int(os.environ.get('NUM_CLASSES', '10'))

import sys
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import get_dataset_split

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_EPOCHS = 50          
BATCH_SIZE   = 128
LR           = 0.001       

def main():
    # ─────────────────────────────────────────────────────────────────────────────
    #  Setup
    # ─────────────────────────────────────────────────────────────────────────────
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training target model on: {device}")
    print(f"Dataset: {DATASET_NAME} | Classes: {NUM_CLASSES}")

    mem_loader, non_mem_loader = get_dataset_split(dataset_name=DATASET_NAME, batch_size=BATCH_SIZE)

    # ─────────────────────────────────────────────────────────────────────────────
    #  Model (Changed to ResNet-50)
    # ─────────────────────────────────────────────────────────────────────────────
    target_net = models.resnet50(num_classes=NUM_CLASSES).to(device)
    loss_fn    = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(target_net.parameters(), lr=LR)

    # ─────────────────────────────────────────────────────────────────────────────
    #  Training loop
    # ─────────────────────────────────────────────────────────────────────────────
    history = {
        'epoch':           [],
        'mem_acc':         [],
        'non_mem_acc':     [],
        'mem_loss':        [],
        'non_mem_loss':    [],
        'gen_gap':         [],     
    }

    print("\n  Epoch  | Mem Acc  | Non-Mem Acc | Gen Gap  | Mem Loss")
    print("  " + "-"*54)

    for ep in range(1, TOTAL_EPOCHS + 1):

        target_net.train()
        ep_loss, tr_hits, tr_total = 0.0, 0, 0

        for imgs, lbls in mem_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            preds = target_net(imgs)
            loss  = loss_fn(preds, lbls)
            loss.backward()
            optimizer.step()

            ep_loss  += loss.item()
            tr_hits  += preds.argmax(1).eq(lbls).sum().item()
            tr_total += lbls.size(0)

        mem_acc  = 100.0 * tr_hits / tr_total
        mem_loss = ep_loss / len(mem_loader)

        target_net.eval()
        nv_hits, nv_total, nv_loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for imgs, lbls in non_mem_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds     = target_net(imgs)
                nv_loss_sum += loss_fn(preds, lbls).item()
                nv_hits     += preds.argmax(1).eq(lbls).sum().item()
                nv_total    += lbls.size(0)

        non_mem_acc  = 100.0 * nv_hits / nv_total
        non_mem_loss = nv_loss_sum / len(non_mem_loader)
        gen_gap      = mem_acc - non_mem_acc

        history['epoch'].append(ep)
        history['mem_acc'].append(mem_acc)
        history['non_mem_acc'].append(non_mem_acc)
        history['mem_loss'].append(mem_loss)
        history['non_mem_loss'].append(non_mem_loss)
        history['gen_gap'].append(gen_gap)

        print(
            f"  {ep:5d}  | {mem_acc:7.2f}% | {non_mem_acc:10.2f}% | "
            f"{gen_gap:7.2f}% | {mem_loss:.4f}"
        )

    # ─────────────────────────────────────────────────────────────────────────────
    #  Save (Updated naming to resnet50)
    # ─────────────────────────────────────────────────────────────────────────────
    os.makedirs('models/targets', exist_ok=True)
    
    ckpt_path = f'models/targets/resnet50_{DATASET_NAME}_baseline.pth'
    torch.save(target_net.state_dict(), ckpt_path)

    final_gap = history['gen_gap'][-1]
    print(f"\n[DONE] Target model saved to {ckpt_path}")
    print(f"  Final Member Acc     : {history['mem_acc'][-1]:.2f}%")
    print(f"  Final Non-Member Acc : {history['non_mem_acc'][-1]:.2f}%")
    print(f"  Generalization Gap   : {final_gap:.2f}%  ← privacy signal ceiling")

    # ─────────────────────────────────────────────────────────────────────────────
    #  Visualisations
    # ─────────────────────────────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Target Model Training (ResNet-50 on {DATASET_NAME})', fontsize=13)

    eps = history['epoch']

    axes[0].plot(eps, history['mem_acc'],     label='Member (Train)',     color='royalblue', lw=2)
    axes[0].plot(eps, history['non_mem_acc'], label='Non-Member (Unseen)', color='tomato',    lw=2)
    axes[0].set_title('Accuracy'); axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps, history['mem_loss'],     label='Member',     color='royalblue', lw=2)
    axes[1].plot(eps, history['non_mem_loss'], label='Non-Member', color='tomato',    lw=2)
    axes[1].set_title('Loss'); axes[1].set_ylabel('Cross-Entropy Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(eps, history['gen_gap'], color='green', lw=2)
    axes[2].fill_between(eps, 0, history['gen_gap'], alpha=0.15, color='green')
    axes[2].set_title('Generalization Gap')
    axes[2].set_ylabel('Gap (%)')
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(f'results/target_training_{DATASET_NAME}.png', dpi=150)
    plt.close()
    print(f"[DONE] Plot saved to results/target_training_{DATASET_NAME}.png")

if __name__ == '__main__':
    main()