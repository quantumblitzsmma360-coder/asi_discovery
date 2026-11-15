import torch
import glob

# Collect all genomes
genome_files = glob.glob('deterministic_asi_genome_worker_*.pt')
genomes = [torch.load(f) for f in genome_files]

# Select best by intelligence
best = max(genomes, key=lambda g: g['results']['final_intelligence'])
print(f'Best worker: {best["worker_id"]} with intelligence {best["results"]["final_intelligence"]:.4f}')

# Save merged genome for next generation
torch.save(best, 'deterministic_asi_best_genome.pt')
print('Best genome saved for next generation')
