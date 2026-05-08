Sequence-Aware Membership Inference: Memorization via Trajectory Transformers
Team: Shahzaib Ali & Muhammad Mohib Waqar  

What This Project Is About
The core question this project answers is: given a trained machine learning model, can you figure out whether a specific data sample was part of its training set? This is called a Membership Inference Attack (MIA) and is a primary method for measuring whether a model leaks private training data.

We build upon the IMIA (Iterative Membership Inference Attack) paper (Xue et al., 2025). IMIA formally proved that running an adversarial attack on a model leaks membership information. Because a model carves out a "valley" in its loss landscape for training data, adversarial attacks require more iterations to break a "member" sample than an unseen "non-member" sample. 

Our Contribution: Sequence-Aware Trajectory Auditing
IMIA reduces the entire adversarial attack process to a single scalar: how many steps did it take to flip the prediction? We hypothesized that collapsing a dynamic attack into a single integer throws away a massive amount of geometric and behavioral information. 

Instead of just recording the endpoint, we record the full behavioral trajectory of the target model step-by-step during the attack. We capture cross-entropy loss, confidence collapse, gradient directionality, and internal latent representations. 

To process this, we developed a Trajectory Transformer, a secondary auditor model utilizing multi-head self-attention, learnable positional embeddings, and global mean pooling. This architecture allows us to identify the specific temporal "fingerprints" of memorization embedded within the optimization paths of adversarial probes.

Threat Models & The Resistance Spectrum
Our pipeline is heavily engineered to evaluate privacy leakage across different datasets and attacker capabilities.

1. White-Box Attack (PGD)
   * Uses a 12-step Projected Gradient Descent (PGD) probe.
   * Extracts a 33-feature "Resistance Spectrum" per step (Loss, Confidence, Gradient Norms, Gradient Direction Cosine Similarity, Probability Gap, Entropy, and Layer-4 activation drift).
   * Attacks are run simultaneously at 3 distinct $\epsilon$-scales (0.01, 0.04, 0.08) to capture the target model's behavior under different optimization pressures.
2. Score-Based Black-Box Attack (SimBA)
   * Attacker has no access to model weights or gradients.
   * Uses a 20-step batched random-walk stochastic search (SimBA-style).
   * Extracts 21 features per step (Loss, Confidence, Entropy, Deltas).

## Supported Architectures & Datasets
* **Target Architecture:** ResNet Models (ResNet-50)
* **Auditor Architectures:** Trajectory Transformer (and a Bi-LSTM baseline for ablation)
* **Datasets Supported:** CIFAR-10, CIFAR-100, and STL-10. 
* **Ablation Ready:** The pipeline includes dynamic data loaders and a strict IMIA-matched replication setup (native 96x96 resolution, SGD, standard augmentation) to isolate the effects of target model regularization on privacy leakage.

## Repository Structure
```text
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
|
├── additions_deliverable4.ipynb      # Notebook containing additional deliverables
├── eval.ipynb                        # Evaluation notebook
└── .gitignore                        # Blocks large .npy and .pth files
