#!/usr/bin/env python3
"""Generate canonical runs.csv from master_manifest.json.

This is the single canonical result table for the preprint.
Every number in the manuscript maps to a run ID in this table.
"""
import json
import csv
import os

MANIFEST = '/Users/davidsvensson/Desktop/dlp_grokking/paper/master_manifest.json'
OUTPUT = '/Users/davidsvensson/Desktop/dlp_grokking/results/runs.csv'

COLUMNS = [
    'experiment_id', 'p', 'q', 'group_type', 'split_type', 'seed',
    'd_model', 'n_heads', 'n_layers', 'P_total', 'P_core',
    'N_train', 'train_frac', 'N_test',
    'T_mem', 'T_50', 'T_90', 'T_95', 'T_99', 'G_delay',
    'best_test_acc', 'final_test_acc', 'total_epochs',
    'run_classification', 'source', 'notes',
]

with open(MANIFEST) as f:
    data = json.load(f)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

with open(OUTPUT, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
    writer.writeheader()
    for exp in data['experiments']:
        writer.writerow(exp)

print(f"Written {len(data['experiments'])} experiments to {OUTPUT}")

# Also write pending experiments
pending_output = '/Users/davidsvensson/Desktop/dlp_grokking/results/pending_runs.csv'
with open(pending_output, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['experiment_id', 'description', 'priority'], extrasaction='ignore')
    writer.writeheader()
    for exp in data.get('pending_experiments', []):
        writer.writerow(exp)

print(f"Written {len(data.get('pending_experiments', []))} pending experiments to {pending_output}")
