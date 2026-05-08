import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import numpy as np

def get_dataset_split(dataset_name='CIFAR10', root='./data/raw', batch_size=128):
    transform_ops = transforms.Compose([
        transforms.Resize((32, 32)), # Force 32x32 for consistent ResNet/grad dims
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset_name == 'CIFAR10':
        full_data = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_ops)
        split_idx = 25000
    elif dataset_name == 'CIFAR100':
        full_data = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_ops)
        split_idx = 25000
    elif dataset_name == 'STL10':
        # STL10 'train' split has 5000 images. Split in half = 2500 members / 2500 non-members
        full_data = datasets.STL10(root=root, split='train', download=True, transform=transform_ops)
        split_idx = 2500
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    idx_list = list(range(len(full_data)))
    random.shuffle(idx_list)

    mem_idx     = idx_list[:split_idx]
    non_mem_idx = idx_list[split_idx:]

    mem_loader = DataLoader(Subset(full_data, mem_idx), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    non_mem_loader = DataLoader(Subset(full_data, non_mem_idx), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return mem_loader, non_mem_loader