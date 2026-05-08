"""
generate_data.py
================
Extracts high-dimensional adversarial trajectory fingerprints.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchvision.models import resnet50  # Changed import

# --- Dynamic Environment Variables ---
DATASET_NAME = os.environ.get('DATASET_NAME', 'CIFAR10')
NUM_CLASSES = int(os.environ.get('NUM_CLASSES', '10'))

import sys
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import get_dataset_split

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
NUM_STEPS       = 12          
MAX_SAMPLES     = 10_000      
FEATURES_PER_SCALE = 11
NUM_SCALES      = 3

EPSILON_SCALES = [
    (0.01, 0.00125),   
    (0.04, 0.005),     
    (0.08, 0.010),     
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Generating trajectories on: {device} for {DATASET_NAME}")

# Changed to resnet50
model = resnet50(num_classes=NUM_CLASSES).to(device)
ckpt_path = f'models/targets/resnet50_{DATASET_NAME}_baseline.pth'

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"Target model not found at '{ckpt_path}'. Run train_target.py first."
    )
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print("Target model loaded.")

_layer4_cache: dict = {}

def _layer4_hook(module, inp, out):
    _layer4_cache['feat'] = out          

_hook_handle = model.layer4.register_forward_hook(_layer4_hook)

def extract_trajectories(loader, split_name: str) -> np.ndarray:
    all_batch_trajs = []
    sample_count = 0

    print(f"\nProbing {split_name}  (up to {MAX_SAMPLES} samples)...")

    for images, labels in tqdm(loader, desc=split_name):
        if sample_count >= MAX_SAMPLES:
            break

        images = images.to(device)   
        labels = labels.to(device)   
        B = images.size(0)

        with torch.no_grad():
            _ = model(images)
            clean_feat = F.adaptive_avg_pool2d(
                _layer4_cache['feat'], (1, 1)
            ).view(B, -1).detach()                 

        scale_trajs = []   

        for eps, alpha in EPSILON_SCALES:
            adv_images    = images.clone().detach()
            prev_loss     = None
            prev_conf     = None
            prev_grad_flat = None
            step_feats    = []   

            for step in range(NUM_STEPS):
                adv_images.requires_grad_(True)
                outputs = model(adv_images)         

                curr_feat = F.adaptive_avg_pool2d(
                    _layer4_cache['feat'], (1, 1)
                ).view(B, -1)                       

                loss_per_sample = F.cross_entropy(outputs, labels, reduction='none')

                model.zero_grad()
                loss_per_sample.sum().backward()

                grad_data = adv_images.grad.data    

                with torch.no_grad():
                    f_loss = loss_per_sample.detach()           
                    probs  = torch.softmax(outputs.detach(), dim=1)
                    f_conf = probs.max(dim=1)[0]                

                    g_flat    = grad_data.view(B, -1)
                    f_gnorm   = g_flat.norm(p=2, dim=1)         

                    f_dloss = (f_loss - prev_loss) if prev_loss is not None else torch.zeros(B, device=device)
                    f_dconf = (f_conf - prev_conf) if prev_conf is not None else torch.zeros(B, device=device)

                    if prev_grad_flat is not None:
                        f_gcos = F.cosine_similarity(g_flat, prev_grad_flat, dim=1)
                    else:
                        f_gcos = torch.ones(B, device=device)

                    f_entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)

                    top2, _ = probs.topk(k=2, dim=1)
                    f_pgap  = top2[:, 0] - top2[:, 1]

                    f_correct = (outputs.detach().argmax(dim=1) == labels).float()

                    cf = curr_feat.detach()
                    f_l4_cos = F.cosine_similarity(cf, clean_feat, dim=1)
                    f_l4_l2 = (cf - clean_feat).norm(p=2, dim=1) / (clean_feat.shape[1] ** 0.5)

                    step_vec = torch.stack([
                        f_loss, f_conf, f_gnorm,
                        f_dloss, f_dconf,
                        f_gcos, f_entropy, f_pgap,
                        f_correct,
                        f_l4_cos, f_l4_l2,
                    ], dim=1)

                step_feats.append(step_vec.cpu().numpy())  

                prev_loss      = f_loss.clone()
                prev_conf      = f_conf.clone()
                prev_grad_flat = g_flat.clone()

                with torch.no_grad():
                    adv_images = adv_images + alpha * grad_data.sign()
                    eta        = (adv_images - images).clamp(-eps, eps)
                    adv_images = (images + eta).clamp(0.0, 1.0).detach()

            scale_arr = np.array(step_feats).transpose(1, 0, 2)
            scale_trajs.append(scale_arr)

        batch_traj = np.concatenate(scale_trajs, axis=2)
        all_batch_trajs.append(batch_traj)
        sample_count += B

    full = np.concatenate(all_batch_trajs, axis=0)          
    print(f"  → {full.shape[0]} samples,  shape: {full.shape}")
    return full

def main():
    dataset_dir = f'data/trajectories/{DATASET_NAME}'
    os.makedirs(dataset_dir, exist_ok=True)

    member_loader, non_member_loader = get_dataset_split(dataset_name=DATASET_NAME)

    m_trajs  = extract_trajectories(member_loader,     "Members")
    nm_trajs = extract_trajectories(non_member_loader, "Non-Members")

    np.save(f'{dataset_dir}/members.npy',     m_trajs)
    np.save(f'{dataset_dir}/non_members.npy', nm_trajs)

    print(f"\n[DONE] Saved to {dataset_dir}/")

    global _hook_handle
    _hook_handle.remove()

if __name__ == '__main__':
    main()