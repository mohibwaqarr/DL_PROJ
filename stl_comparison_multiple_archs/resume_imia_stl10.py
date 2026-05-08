import os
import subprocess
import re
import numpy as np

print("\n" + "="*70)
print(" Resuming Pipeline for STL-10 (IMIA-Matched Replication)")
print("="*70)

env = os.environ.copy()
env['DATASET_NAME'] = 'STL10_IMIA'
env['PYTHONIOENCODING'] = 'utf-8'

try:
    # --- SMART SKIP 1: Target Model ---
    target_path = 'models/targets/resnet50_STL10_IMIA_baseline.pth'
    if os.path.exists(target_path):
        print(f"--> [1/4] Target model found at {target_path}. SKIPPING TRAINING!")
    else:
        print("--> [1/4] Target model missing. You must place the .pth file back in models/targets/!")
        exit(1)

    # --- SMART SKIP 2: Trajectories ---
    traj_path = "data/trajectories/STL10_IMIA/members.npy"
    if os.path.exists(traj_path):
        print(f"--> [2/4] Trajectories found for STL10_IMIA. SKIPPING GENERATION!")
    else:
        print("--> [2/4] Generating Trajectories (IMIA Setup: 5k/8k split)...")
        subprocess.run(["python", "generate_data_imia_stl10.py"], env=env, check=True)

    dataset_accs = []
    dataset_aucs = []

    # --- STEPS 3 & 4: Auditor Training & Evaluation (3 Runs) ---
    for run_idx in range(3):
        current_seed = 42 + run_idx
        env['RUN_SEED'] = str(current_seed)
        
        print(f"\n  --- Auditor Run {run_idx + 1}/3 (Seed: {current_seed}) ---")
        print("  --> [3/4] Training Trajectory Transformer...")
        subprocess.run(["python", "train_auditor.py"], env=env, check=True)

        print("  --> [4/4] Evaluating Auditor...")
        eval_proc = subprocess.run(["python", "evaluate_auditor.py"], env=env, check=True, capture_output=True, text=True, encoding='utf-8')
        
        acc_match = re.search(r"Accuracy\s*:\s*([\d\.]+)%", eval_proc.stdout)
        auc_match = re.search(r"ROC AUC\s*:\s*([\d\.]+)", eval_proc.stdout)

        if acc_match and auc_match:
            acc = float(acc_match.group(1))
            auc = float(auc_match.group(1)) * 100
            dataset_accs.append(acc)
            dataset_aucs.append(auc)
            print(f"  [OK] Run {run_idx + 1} Complete: Acc={acc:.2f}%, AUC={auc:.2f}%")
        else:
            print(f"  [ERROR] Parsing failed. Raw Output:\n{eval_proc.stdout}")

    # --- FINAL METRICS ---
    if dataset_accs and dataset_aucs:
        print("\n\n" + "="*70)
        print(" FINAL METRICS: STL-10 (IMIA-MATCHED TARGET | NATIVE 96x96)")
        print("="*70)
        print(f"  Trajectory Transformer Accuracy : {np.mean(dataset_accs):.2f} \u00B1 {np.std(dataset_accs):.2f}%")
        print(f"  Trajectory Transformer AUROC    : {np.mean(dataset_aucs):.2f} \u00B1 {np.std(dataset_aucs):.2f}%")
        print("="*70)

except subprocess.CalledProcessError as e:
    print("\n[ERROR] Pipeline crashed.")
    print(f"--- Python Error Traceback ---\n{e.stderr}\n------------------------------")