"""
train_target_imia_stl10.py
==========================
Trains ResNet-50 on native STL-10 (96x96) matching the exact IMIA methodology.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms

TOTAL_EPOCHS = 100
BATCH_SIZE   = 128

def main():
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training IMIA-Matched STL-10 Target (Native 96x96) on: {device}")

    # --- NATIVE RESOLUTION TRANSFORMS ---
    train_transform = transforms.Compose([
        transforms.RandomCrop(96, padding=4), # Native STL-10 size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(), # No resizing!
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    mem_data = datasets.STL10(root='./data/raw', split='train', download=True, transform=train_transform)
    non_mem_data = datasets.STL10(root='./data/raw', split='test', download=True, transform=test_transform)

    mem_loader = torch.utils.data.DataLoader(mem_data, batch_size=BATCH_SIZE, shuffle=True)
    non_mem_loader = torch.utils.data.DataLoader(non_mem_data, batch_size=BATCH_SIZE, shuffle=False)

    target_net = models.resnet50(num_classes=10).to(device)
    loss_fn    = nn.CrossEntropyLoss()
    
    optimizer  = optim.SGD(target_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)

    history = {'epoch': [], 'mem_acc': [], 'non_mem_acc': [], 'mem_loss': [], 'non_mem_loss': [], 'gen_gap': []}

    print("\n  Epoch  | Mem Acc  | Non-Mem Acc | Gen Gap  | Mem Loss | LR")
    print("  " + "-"*62)

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
        current_lr   = scheduler.get_last_lr()[0]
        
        scheduler.step()

        history['epoch'].append(ep)
        history['mem_acc'].append(mem_acc)
        history['non_mem_acc'].append(non_mem_acc)
        history['mem_loss'].append(mem_loss)
        history['non_mem_loss'].append(non_mem_loss)
        history['gen_gap'].append(gen_gap)

        print(f"  {ep:5d}  | {mem_acc:7.2f}% | {non_mem_acc:10.2f}% | {gen_gap:7.2f}% | {mem_loss:.4f} | {current_lr:.4f}")

    os.makedirs('models/targets', exist_ok=True)
    ckpt_path = 'models/targets/resnet50_STL10_IMIA_baseline.pth'
    torch.save(target_net.state_dict(), ckpt_path)

    print(f"\n[DONE] Target model saved to {ckpt_path}")
    print(f"  Final Member Acc     : {history['mem_acc'][-1]:.2f}%")
    print(f"  Final Non-Member Acc : {history['non_mem_acc'][-1]:.2f}%")
    print(f"  Generalization Gap   : {history['gen_gap'][-1]:.2f}%")

if __name__ == '__main__':
    main()