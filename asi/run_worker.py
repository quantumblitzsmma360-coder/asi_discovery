import torch
import os
import sys
sys.path.append('.')

from deterministic_asi import DeterministicASI, ASIConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
worker_id = os.environ.get('WORKER_ID', '1')
seed = int(os.environ.get('SEED', '137'))

print(f'Worker {worker_id} on {device}')

# Config for this worker
config = ASIConfig(
    latent_dim=512,
    max_ast_depth=10,
    verification_timeout=30,
    modification_budget=1000,
    seed=seed + int(worker_id)
)

asi = DeterministicASI(config).to(device)

# Generate synthetic CVE data
torch.manual_seed(config.seed)
num_samples = 1000
num_features = 100
num_classes = 10
X = torch.randn(num_samples, num_features, device=device)
y = torch.randint(0, num_classes, (num_samples,), device=device)

# Run autonomous RSI loop
results = asi.autonomous_rsi_loop(
    X, y,
    generations=5,
    verification_interval=10
)

print(f'Worker {worker_id}: Final intelligence = {results["final_intelligence"]:.4f}')
print(f'Worker {worker_id}: Modifications = {results["total_modifications"]}')
print(f'Worker {worker_id}: Verified = {results["verified_modifications"]}')

# Save genome
torch.save({
    'genome': asi.export_genome(),
    'config': config.__dict__,
    'results': results,
    'worker_id': worker_id
}, f'deterministic_asi_genome_worker_{worker_id}.pt')
