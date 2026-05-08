import os
import subprocess
import re
import numpy as np

datasets = {
    'CIFAR100': 100,
    'CIFAR10': 10,
    'STL10': 10
}

NUM_RUNS = 3
results = {}

for ds_name, num_classes in datasets.items():
    print(f"\n{'='*60}\n Executing Pipeline for {ds_name} \n{'='*60}")
    
    env = os.environ.copy()
    env['DATASET_NAME'] = ds_name
    env['NUM_CLASSES'] = str(num_classes)
    
    # Force Python to use UTF-8 for subprocess piping to be extra safe on Windows
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        # Removed Smart Skip so it forces ResNet-50 training and generation
        print(f"--> [1/4] Training Target Model (ResNet-50)...")
        subprocess.run(["python", "train_target.py"], env=env, check=True)

        print(f"--> [2/4] Generating Trajectories (ResNet-50)...")
        subprocess.run(["python", "generate_data.py"], env=env, check=True)

        dataset_accs = []
        dataset_aucs = []

        # Step 3 & 4: Loop the Auditor 3 times for variance
        for run_idx in range(NUM_RUNS):
            current_seed = 42 + run_idx
            env['RUN_SEED'] = str(current_seed)
            
            print(f"\n  --- Auditor Run {run_idx + 1}/{NUM_RUNS} (Seed: {current_seed}) ---")
            
            print(f"  --> [3/4] Training Trajectory Transformer...")
            subprocess.run(["python", "train_auditor.py"], env=env, check=True)

            print(f"  --> [4/4] Evaluating Auditor...")
            eval_proc = subprocess.run(["python", "evaluate_auditor.py"], env=env, check=True, capture_output=True, text=True, encoding='utf-8')
            
            # Parse metrics
            acc_match = re.search(r"Accuracy\s*:\s*([\d\.]+)%", eval_proc.stdout)
            auc_match = re.search(r"ROC AUC\s*:\s*([\d\.]+)", eval_proc.stdout)

            if acc_match and auc_match:
                acc = float(acc_match.group(1))
                auc = float(auc_match.group(1)) * 100  # Convert AUC to percentage to match paper
                dataset_accs.append(acc)
                dataset_aucs.append(auc)
                print(f"  [OK] Run {run_idx + 1} Complete: Acc={acc:.2f}%, AUC={auc:.2f}%")
            else:
                print(f"  [ERROR] Failed to parse metrics for run {run_idx + 1}. Raw Output:\n{eval_proc.stdout}")

        # Calculate Mean and Standard Deviation if runs succeeded
        if dataset_accs and dataset_aucs:
            mean_acc = np.mean(dataset_accs)
            std_acc = np.std(dataset_accs)
            mean_auc = np.mean(dataset_aucs)
            std_auc = np.std(dataset_aucs)

            results[ds_name] = {
                "Acc_Str": f"{mean_acc:.2f} \u00B1 {std_acc:.2f}",
                "AUC_Str": f"{mean_auc:.2f} \u00B1 {std_auc:.2f}"
            }

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Subprocess crashed during {ds_name}.")
        print(f"--- Python Error Traceback ---\n{e.stderr}\n------------------------------")
        break

# Final Output formatting
print("\n\n" + "="*70)
print(" FINAL METRICS FOR TABLE 3 (ResNet-50 White-Box)")
print("="*70)
print(f"{'Dataset':<12} | {'Accuracy (%)':<20} | {'AUROC (%)':<20}")
print("-" * 70)
for ds_name, metrics in results.items():
    print(f"{ds_name:<12} | {metrics['Acc_Str']:<20} | {metrics['AUC_Str']:<20}")
print("="*70)