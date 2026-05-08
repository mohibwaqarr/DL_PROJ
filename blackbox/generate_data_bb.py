"""
generate_data_bb.py
===================
Extracts SCORE-BASED BLACKBOX trajectories.
No gradients. No internal layer hooks.
Uses a SimBA-style random search to perturb the image.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet50

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
FEATURES_PER_SCALE = 7  # Reduced from 11 (Removed grad and layer4 features)
NUM_SCALES      = 3

EPSILON_SCALES = [
    (0.01, 0.00125),   
    (0.04, 0.005),     
    (0.08, 0.010),     
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Generating BLACKBOX trajectories on: {device} for {DATASET_NAME}")

model = resnet50(num_classes=NUM_CLASSES).to(device)
ckpt_path = f'models/targets/resnet50_{DATASET_NAME}_baseline.pth'

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Target model not found at '{ckpt_path}'. Run train_target.py first.")
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print("Target model loaded.")

def extract_trajectories_bb(loader, split_name: str) -> np.ndarray:
    all_batch_trajs = []
    sample_count = 0

    print(f"\nProbing {split_name} (Blackbox, up to {MAX_SAMPLES} samples)...")

    for images, labels in tqdm(loader, desc=split_name):
        if sample_count >= MAX_SAMPLES:
            break

        images = images.to(device)   
        labels = labels.to(device)   
        B = images.size(0)

        scale_trajs = []   

        for eps, alpha in EPSILON_SCALES:
            adv_images = images.clone().detach()
            prev_loss = None
            prev_conf = None
            step_feats = []   

            for step in range(NUM_STEPS):
                # 1. Query the model (NO GRADIENTS)
                with torch.no_grad():
                    outputs = model(adv_images)
                    probs  = torch.softmax(outputs, dim=1)
                    
                    f_loss = F.cross_entropy(outputs, labels, reduction='none')
                    f_conf = probs.max(dim=1)[0]                
                    f_dloss = (f_loss - prev_loss) if prev_loss is not None else torch.zeros(B, device=device)
                    f_dconf = (f_conf - prev_conf) if prev_conf is not None else torch.zeros(B, device=device)
                    f_entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
                    
                    top2, _ = probs.topk(k=2, dim=1)
                    f_pgap  = top2[:, 0] - top2[:, 1]
                    f_correct = (outputs.argmax(dim=1) == labels).float()

                    # Blackbox Step Vector (7 Features)
                    step_vec = torch.stack([
                        f_loss, f_conf, f_dloss, f_dconf, f_entropy, f_pgap, f_correct
                    ], dim=1)

                    step_feats.append(step_vec.cpu().numpy())  

                    prev_loss = f_loss.clone()
                    prev_conf = f_conf.clone()

                    # 2. Blackbox Score-Based Update (SimBA-lite)
                    # Generate random noise direction
                    noise = torch.randn_like(images).sign() * alpha
                    
                    # Query positive and negative directions
                    out_plus = model((adv_images + noise).clamp(0.0, 1.0))
                    loss_plus = F.cross_entropy(out_plus, labels, reduction='none')
                    
                    out_minus = model((adv_images - noise).clamp(0.0, 1.0))
                    loss_minus = F.cross_entropy(out_minus, labels, reduction='none')
                    
                    # Keep the direction that maximizes the loss (hurts the model more)
                    mask_plus = (loss_plus > loss_minus).view(B, 1, 1, 1)
                    best_noise = torch.where(mask_plus, noise, -noise)
                    
                    # Update and clip
                    adv_images = adv_images + best_noise
                    eta = (adv_images - images).clamp(-eps, eps)
                    adv_images = (images + eta).clamp(0.0, 1.0)

            scale_arr = np.array(step_feats).transpose(1, 0, 2)
            scale_trajs.append(scale_arr)

        batch_traj = np.concatenate(scale_trajs, axis=2)
        all_batch_trajs.append(batch_traj)
        sample_count += B

    full = np.concatenate(all_batch_trajs, axis=0)          
    return full

def main():
    dataset_dir = f'data/trajectories_bb/{DATASET_NAME}'
    os.makedirs(dataset_dir, exist_ok=True)

    member_loader, non_member_loader = get_dataset_split(dataset_name=DATASET_NAME)

    m_trajs  = extract_trajectories_bb(member_loader,     "Members")
    nm_trajs = extract_trajectories_bb(non_member_loader, "Non-Members")

    np.save(f'{dataset_dir}/members.npy',     m_trajs)
    np.save(f'{dataset_dir}/non_members.npy', nm_trajs)

    print(f"\n[DONE] Blackbox Trajectories Saved to {dataset_dir}/")
    print(f"Shape: {m_trajs.shape} (Steps: {NUM_STEPS}, Features: {FEATURES_PER_SCALE * NUM_SCALES})")

if __name__ == '__main__':
    main()