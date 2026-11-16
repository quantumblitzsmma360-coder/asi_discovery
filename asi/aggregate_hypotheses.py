import os
import sys
import json
import glob
import numpy as np

print("=" * 70)
print("AGGREGATING HYPOTHESIS RESULTS")
print("=" * 70)

# Find all result files
result_files = glob.glob('results_*.json')
print(f"Found {len(result_files)} result files")

results_by_hypothesis = {}
for file in result_files:
    with open(file, 'r') as f:
        result = json.load(f)
    h_id = result.get('hypothesis_id', 'unknown')
    if h_id not in results_by_hypothesis:
        results_by_hypothesis[h_id] = []
    results_by_hypothesis[h_id].append(result)

# Print summary
for h_id, results in results_by_hypothesis.items():
    print(f"
{h_id}: {len(results)} workers")
    for r in results:
        param = r.get('parameter', {})
        print(f"  Worker {r.get('worker_id', '?')}: {param}")

# Save aggregated results
with open('aggregated_results.json', 'w') as f:
    json.dump(results_by_hypothesis, f, indent=2, default=str)

print("
Aggregated results saved to aggregated_results.json")
print("=" * 70)
