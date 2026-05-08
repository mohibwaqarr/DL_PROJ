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
    print(f"\n{'='*70}\n Executing BLACKBOX Pipeline for {ds_name} \n{'='*70}")
    
    env = os.environ.copy()
    env['DATASET_NAME'] = ds_name
    env['NUM_CLASSES'] = str(num_classes)
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        # Note: We DO NOT train the target model. We use the existing baseline targets.
        
        # Step 1: Generate Blackbox Trajectories
        traj_path = f"data/trajectories_bb/{ds_name}/members.npy"
        if os.path.exists(traj_path):
            print(f"--> [1/2] Blackbox Trajectories found for {ds_name}. Skipping generation!")
        else:
            print(f"--> [1/2] Generating BLACKBOX Trajectories...")
            subprocess.run(["python", "generate_data_bb.py"], env=env, check=True)

        dataset_accs = []
        dataset_aucs = []

        # Step 2: Loop the Blackbox Auditor 3 times
        for run_idx in range(NUM_RUNS):
            current_seed = 42 + run_idx
            env['RUN_SEED'] = str(current_seed)
            
            print(f"\n  --- BB Auditor Run {run_idx + 1}/{NUM_RUNS} (Seed: {current_seed}) ---")
            subprocess.run(["python", "train_auditor_bb.py"], env=env, check=True)

            eval_proc = subprocess.run(["python", "evaluate_auditor_bb.py"], env=env, check=True, capture_output=True, text=True, encoding='utf-8')
            
            acc_match = re.search(r"Accuracy:\s*([\d\.]+)%", eval_proc.stdout)
            auc_match = re.search(r"AUC:\s*([\d\.]+)", eval_proc.stdout)

            if acc_match and auc_match:
                acc = float(acc_match.group(1))
                auc = float(auc_match.group(1)) * 100 
                dataset_accs.append(acc)
                dataset_aucs.append(auc)
                print(f"  [OK] BB Run {run_idx + 1} Complete: Acc={acc:.2f}%, AUC={auc:.2f}%")
            else:
                print(f"  [ERROR] Parsing failed. Output:\n{eval_proc.stdout}")

        if dataset_accs and dataset_aucs:
            results[ds_name] = {
                "Acc_Str": f"{np.mean(dataset_accs):.2f} \u00B1 {np.std(dataset_accs):.2f}",
                "AUC_Str": f"{np.mean(dataset_aucs):.2f} \u00B1 {np.std(dataset_aucs):.2f}"
            }

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pipeline crashed on {ds_name}.")
        print(e.stderr)
        break

print("\n\n" + "="*70)
print(" FINAL METRICS: SCORE-BASED BLACKBOX MIA (Table 1 Comparison)")
print("="*70)
print(f"{'Dataset':<12} | {'Accuracy (%)':<20} | {'AUROC (%)':<20}")
print("-" * 70)
for ds_name, metrics in results.items():
    print(f"{ds_name:<12} | {metrics['Acc_Str']:<20} | {metrics['AUC_Str']:<20}")
print("="*70)