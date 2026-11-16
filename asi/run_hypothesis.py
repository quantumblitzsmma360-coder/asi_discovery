"""
Hypothesis Runner for ASI Swarm
Each worker tests ONE hypothesis with controlled variables
Seed: 137 | Classification: WAR-FOOTING
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime

sys.path.append('.')

from deterministic_asi import DeterministicASI, ASIConfig
from reflective_neuralese import ReflectiveNeuralese, NeuraleseCore, TrueDifferentiableSelfModel, CausalWorldModel
from physics_enhanced_reflective_neuralese import PhysicsEnhancedReflectiveNeuralese, PhysicsEnhancedReasoningBlock
from verification_engine import SMTDVerificationEngine, FormalSelfModel
from intelligence_operator import ProvableIntelligenceOperator
from causal_world_model import CausalEnvironment, CausalWorldModel, CausalTrainer

from run_hypothesis_funcs import *


def run_hypothesis(hypothesis_id, worker_id, seed=137):
    import torch, numpy as np
    torch.manual_seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    
    hypothesis = HYPOTHESES[hypothesis_id]
    param_name = hypothesis["param"]
    param_values = hypothesis["values"]
    param_value = param_values[worker_id % len(param_values)]
    
    print(f"=" * 70)
    print(f"HYPOTHESIS: {hypothesis_id}")
    print(f"Worker: {worker_id}")
    print(f"Parameter: {hypothesis["param"]} = {param_value}")
    print(f"Seed: {seed + worker_id}")
    print("=" * 70)
    
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    result = run_experiment(hypothesis_id, hypothesis["param"], hypothesis["values"][worker_id % len(hypothesis["values"])], device, seed + worker_id)
    
    result["hypothesis_id"] = hypothesis_id
    result["worker_id"] = worker_id
    result["parameter"] = {hypothesis["param"]: hypothesis["values"][worker_id % len(hypothesis["values"])]}
    result["seed"] = seed + worker_id
    from datetime import datetime
    result["timestamp"] = datetime.utcnow().isoformat()
    return result

def run_experiment(hypothesis_id, param_name, param_value, device, seed):
    import torch
    torch.manual_seed(seed)
    if hypothesis_id == "H1_neuralese_act_arc": return run_h1_neuralese_act_arc(param_value, device, seed)
    elif hypothesis_id == "H2_causal_wm_clevrer": return run_h2_causal_wm_clevrer(param_value, device, seed)
    elif hypothesis_id == "H3_fast_weight_gsm8k": return run_h3_fast_weight_gsm8k(param_value, device, seed)
    elif hypothesis_id == "H4_physics_modes_humaneval": return run_h4_physics_modes_humaneval(param_value, device, seed)
    elif hypothesis_id == "H5_z3_semantic_verification": return run_h5_z3_semantic_verification(param_value, device, seed)
    elif hypothesis_id == "H6_mergechain_real_benchmarks": return run_h6_mergechain_real_benchmarks(param_value, device, seed)
    elif hypothesis_id == "H7_selfmod_adversarial_safety": return run_h7_selfmod_adversarial_safety(param_value, device, seed)
    elif hypothesis_id == "H8_neuralese_bandwidth": return run_h8_neuralese_bandwidth(param_value, device, seed)
    elif hypothesis_id == "H9_genome_sync_fidelity": return run_h9_genome_sync_fidelity(param_value, device, seed)
    elif hypothesis_id == "H10_agentpub_verification": return run_h10_agentpub_verification(param_value, device, seed)
    else: return {"error": f"Unknown hypothesis: {hypothesis_id}"}

if __name__ == "__main__":
    import os, json
    hypothesis_id = os.environ.get("HYPOTHESIS", "H1_neuralese_act_arc")
    worker_id = int(os.environ.get("WORKER_ID", "1"))
    seed = int(os.environ.get("SEED", "137"))
    print(f"Starting hypothesis: {hypothesis_id}, Worker: {worker_id}")
    result = run_hypothesis(hypothesis_id, worker_id, seed)
    output_file = f"results_{hypothesis_id}_worker_{worker_id}.json"
    with open(output_file, 'w') as f:
        import json
        json.dump(result, f, indent=2)
    print(f"Result saved to {output_file}")
    print(f"Result: {result}")