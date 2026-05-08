"""
generate_data_imia_stl10.py
===========================
Extracts trajectories using the IMIA-matched ResNet-50 target on native STL-10 resolution.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet50
from torchvision import datasets, transforms

NUM_STEPS = 12          
MAX_SAMPLES = 10_000      
EPSILON_SCALES = [(0.01, 0.00125), (0.04, 0.005), (0.08, 0.010)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Generating trajectories on: {device} for STL-10 (Native 96x96, IMIA Split)")

model = resnet50(num_classes=10).to(device)
ckpt_path = 'models/targets/resnet50_STL10_IMIA_baseline.pth'
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

_layer4_cache: dict = {}
def _layer4_hook(module, inp, out): _layer4_cache['feat'] = out          
_hook_handle = model.layer4.register_forward_hook(_layer4_hook)

def extract_trajectories(loader, split_name: str) -> np.ndarray:
    all_batch_trajs = []
    sample_count = 0
    print(f"\nProbing {split_name} (up to {MAX_SAMPLES} samples)...")

    for images, labels in tqdm(loader, desc=split_name):
        if sample_count >= MAX_SAMPLES: break
        images, labels = images.to(device), labels.to(device)   
        B = images.size(0)

        with torch.no_grad():
            _ = model(images)
            clean_feat = F.adaptive_avg_pool2d(_layer4_cache['feat'], (1, 1)).view(B, -1).detach()                 

        scale_trajs = []   
        for eps, alpha in EPSILON_SCALES:
            adv_images, prev_loss, prev_conf, prev_grad_flat = images.clone().detach(), None, None, None
            step_feats = []   

            for step in range(NUM_STEPS):
                adv_images.requires_grad_(True)
                outputs = model(adv_images)         
                curr_feat = F.adaptive_avg_pool2d(_layer4_cache['feat'], (1, 1)).view(B, -1)                       
                loss_per_sample = F.cross_entropy(outputs, labels, reduction='none')

                model.zero_grad()
                loss_per_sample.sum().backward()
                grad_data = adv_images.grad.data    

                with torch.no_grad():
                    f_loss = loss_per_sample.detach()           
                    probs  = torch.softmax(outputs.detach(), dim=1)
                    f_conf = probs.max(dim=1)[0]                
                    g_flat = grad_data.view(B, -1)
                    f_gnorm = g_flat.norm(p=2, dim=1)         
                    f_dloss = (f_loss - prev_loss) if prev_loss is not None else torch.zeros(B, device=device)
                    f_dconf = (f_conf - prev_conf) if prev_conf is not None else torch.zeros(B, device=device)
                    f_gcos = F.cosine_similarity(g_flat, prev_grad_flat, dim=1) if prev_grad_flat is not None else torch.ones(B, device=device)
                    f_entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
                    top2, _ = probs.topk(k=2, dim=1)
                    f_pgap  = top2[:, 0] - top2[:, 1]
                    f_correct = (outputs.detach().argmax(dim=1) == labels).float()
                    cf = curr_feat.detach()
                    f_l4_cos = F.cosine_similarity(cf, clean_feat, dim=1)
                    f_l4_l2 = (cf - clean_feat).norm(p=2, dim=1) / (clean_feat.shape[1] ** 0.5)

                    step_vec = torch.stack([f_loss, f_conf, f_gnorm, f_dloss, f_dconf, f_gcos, f_entropy, f_pgap, f_correct, f_l4_cos, f_l4_l2], dim=1)

                step_feats.append(step_vec.cpu().numpy())  
                prev_loss, prev_conf, prev_grad_flat = f_loss.clone(), f_conf.clone(), g_flat.clone()

                with torch.no_grad():
                    adv_images = adv_images + alpha * grad_data.sign()
                    eta = (adv_images - images).clamp(-eps, eps)
                    adv_images = (images + eta).clamp(0.0, 1.0).detach()

            scale_trajs.append(np.array(step_feats).transpose(1, 0, 2))

        all_batch_trajs.append(np.concatenate(scale_trajs, axis=2))
        sample_count += B

    return np.concatenate(all_batch_trajs, axis=0)

def main():
    dataset_dir = 'data/trajectories/STL10_IMIA'
    os.makedirs(dataset_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(), # No resizing!
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    mem_data = datasets.STL10(root='./data/raw', split='train', download=True, transform=transform)
    non_mem_data = datasets.STL10(root='./data/raw', split='test', download=True, transform=transform)
    
    member_loader = torch.utils.data.DataLoader(mem_data, batch_size=128, shuffle=False)
    non_member_loader = torch.utils.data.DataLoader(non_mem_data, batch_size=128, shuffle=False)

    m_trajs  = extract_trajectories(member_loader, "Members (5k)")
    nm_trajs = extract_trajectories(non_member_loader, "Non-Members (8k)")

    np.save(f'{dataset_dir}/members.npy', m_trajs)
    np.save(f'{dataset_dir}/non_members.npy', nm_trajs)

    print(f"\n[DONE] Saved to {dataset_dir}/")
    _hook_handle.remove()

if __name__ == '__main__':
    main()