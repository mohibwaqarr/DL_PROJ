# Sequence-Aware Membership Inference: Memorization via Trajectory Transformers

> An advanced privacy-auditing framework that implements temporal sequence learning to dynamically detect training data memorization..


---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Key Innovations](#key-innovations)
  - [The 33-Feature Resistance Spectrum](#the-33-feature-resistance-spectrum)
  - [Trajectory Transformer Auditor](#trajectory-transformer-auditor)
  - [Black-Box Stochastic Probing via SimBA](#black-box-stochastic-probing-via-simba)
- [Citations](#citations)

---

## Project Overview

Membership Inference Attacks (MIA) serve as a critical audit for the privacy of individuals within machine learning datasets. Traditional auditing methodologies rely on scalar iteration counts or static posterior metrics, which collapse high-dimensional behavioral signals into a single value, leading to significant information loss.

Our framework addresses this by recording the **full behavioral trajectory** of a target model step-by-step as it is subjected to an adversarial probe. We track features across three separate perturbation budgets simultaneously to build a multi-resolution **Resistance Spectrum**. This trajectory sequence is then passed to a custom **Trajectory Transformer** auditor utilizing self-attention to identify temporal markers of unintended memorization.

---

## Repository Structure

```
project/
├── src/
│   ├── data_loader.py                # Unified data splitting (CIFAR10/100, STL10)
│   └── auditor_arch.py               # Trajectory Transformer & Bi-LSTM architectures
│
├── whitebox/
│   ├── master_table3.py              # Orchestrator for White-Box runs across all datasets
│   ├── train_target.py               # Trains ResNet victim models
│   ├── generate_data.py              # PGD White-Box feature extraction (33 features)
│   ├── train_auditor.py              # Trains White-Box Transformer auditor
│   └── evaluate_auditor.py           # Evaluates White-Box model
│
├── blackbox/
│   ├── master_bb.py                  # Orchestrator for Black-Box runs across all datasets
│   ├── generate_data_bb.py           # SimBA Black-Box feature extraction (21 features)
│   ├── train_auditor_bb.py           # Trains Black-Box Transformer auditor
│   └── evaluate_auditor_bb.py        # Evaluates Black-Box model
│
├── stl_comparison_multiple_archs/
│   ├── master_imia_stl10.py          # Orchestrator for strict IMIA-matched STL-10 ablation
│   ├── resume_imia_stl10.py          # Smart-resume orchestrator for STL-10
│   ├── train_target_imia_stl10.py    # Trains strict IMIA-matched STL-10 target (Native 96x96)
│   └── generate_data_imia_stl10.py   # Extracts trajectories from IMIA-matched target
│
├── deliverables/
│   ├── additions_deliverable3.ipynb
│   └── additions_deliverable4.ipynb
│
└── .gitignore                        # Blocks large .npy and .pth files
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- numpy
- matplotlib
- tqdm

### Running the Code

Our pipeline operates sequentially. Execute all scripts from the **root directory** of the project.

---

#### 1. Standard White-Box Evaluation (PGD-12)

Run the automated orchestrator:

```bash
python whitebox/master_table3.py
```

Or execute the scripts manually, step-by-step:

```bash
python whitebox/train_target.py
python whitebox/generate_data.py
python whitebox/train_auditor.py
python whitebox/evaluate_auditor.py
```

---

#### 2. Score-Based Black-Box Evaluation (SimBA-20)

Run the automated orchestrator:

```bash
python blackbox/master_bb.py
```

Or execute the scripts manually, step-by-step:

```bash
python blackbox/generate_data_bb.py
python blackbox/train_auditor_bb.py
python blackbox/evaluate_auditor_bb.py
```

---

#### 3. Strict IMIA-Matched STL-10 Ablation Study

Run the automated orchestrator:

```bash
python stl_comparison_multiple_archs/master_imia_stl10.py
```

Or resume an interrupted run safely using:

```bash
python stl_comparison_multiple_archs/resume_imia_stl10.py
```

---

## Key Innovations

### The 33-Feature Resistance Spectrum

To capture a sample's robust resistance to optimization, we log **11 core behavioral metrics** at every step of a 12-step PGD probe across three independent epsilon budgets ($\epsilon \in \{0.01, 0.04, 0.08\}$):

```python
def extract_resistance_spectrum_step(model, images, labels, clean_features):
    # Extracts the 11-dimensional step vector
    loss = compute_cross_entropy(model, images, labels)
    confidence = compute_softmax_confidence(model, images)
    entropy = compute_prediction_entropy(model, images)
    prob_gap = compute_top_two_margin(model, images)

    grad_norm = compute_input_gradient_norm(model, images, labels)
    grad_cosine = compute_gradient_cosine_similarity(current_grad, prev_grad)

    # Internal representation dynamics
    layer4_features = get_layer4_activations(model, images)
    l4_cosine = compute_cosine_similarity(layer4_features, clean_features)
    l4_drift = compute_l2_activation_drift(layer4_features, clean_features)

    return torch.stack([
        loss, confidence, entropy, prob_gap, grad_norm, grad_cosine,
        delta_loss, delta_confidence, l4_cosine, l4_drift, is_correct
    ], dim=1)
```

By tracing internal activations alongside posterior changes, the auditor observes **"latent rigidity"** in intermediate layers — where a memorized member's representations resist moving under attack.

---

### Trajectory Transformer Auditor

Instead of recurrent models which dilute early sequence features with late-stage convergence noise, we use a custom self-attention block with **Learnable Positional Embeddings** to isolate highly discriminative steps.

```python
class TrajectoryTransformer(nn.Module):
    def __init__(self, input_size=33, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        # Learnable temporal parameters rather than static sinusoids
        self.pos_embeddings = nn.Parameter(torch.randn(12, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.3, norm_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [Batch, Steps, Features]
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embeddings[:seq_len]
        x = self.transformer(x)

        # Threshold-independent Global Mean Pooling
        z = x.mean(dim=1)
        return torch.sigmoid(self.fc(z))
```

---

### Black-Box Stochastic Probing via SimBA

In hard-label or score-only environments, gradient backpropagation is unavailable. We implement a **SimBA-style random-walk optimization**. The auditor tracks the model's posterior response across 20 stochastic steps, locating stable local minima where member samples resist random coordinate walks:

```python
def generate_simba_trajectory(model, x, y, steps=20, alpha=0.005):
    trajectory = []
    x_adv = x.clone()

    for t in range(steps):
        q = sample_random_orthonormal_basis(x)

        # Probe positive and negative directions
        loss_plus = get_loss(model, x_adv + alpha * q, y)
        loss_minus = get_loss(model, x_adv - alpha * q, y)

        # Keep the perturbation that maximizes loss
        if loss_plus > loss_minus and loss_plus > current_loss:
            x_adv = x_adv + alpha * q
        elif loss_minus > loss_plus and loss_minus > current_loss:
            x_adv = x_adv - alpha * q

        trajectory.append([current_loss, current_confidence, current_entropy])

    return trajectory
```

---

## Citations

```bibtex
@article{xue2025imia,
  author  = {Xue, M. et al.},
  title   = {IMIA: Iterative Membership Inference Attack via Adversarial Perturbation},
  year    = {2025}
}

@inproceedings{carlini2022membership,
  author    = {Carlini, N. and Chien, S. and Nasr, M. and Song, S. and Terzis, A. and Tramer, F.},
  title     = {Membership Inference Attacks From First Principles},
  booktitle = {IEEE S\&P},
  year      = {2022}
}

@article{aubinais2025theoretical,
  author  = {Aubinais et al.},
  title   = {Theoretical Foundations of Membership Inference and Non-Linear Memorization Bounds},
  year    = {2025}
}

@inproceedings{feldman2020neural,
  author    = {Feldman, V. and Zhang, C.},
  title     = {What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation},
  booktitle = {NeurIPS},
  year      = {2020}
}
```
