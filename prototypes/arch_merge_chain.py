import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import math

# Merge-Chain RSI with Architecture Mutation
# Extends weight-only soup to include architecture genome crossover

class ArchitectureGenome:
    """Genome encoding neural architecture - mutable and crossable"""
    def __init__(self, 
                 depth=4,
                 width=512,
                 connectivity='residual',
                 attention_type='standard',
                 ffn_type='standard',
                 physics_mode='none',
                 num_heads=8):
        
        self.depth = depth
        self.width = width
        self.connectivity = connectivity
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.physics_mode = physics_mode
        self.num_heads = num_heads
        
        self._validate()
        
    def _validate(self):
        assert 1 <= self.depth <= 12
        assert 128 <= self.width <= 1024
        assert self.width % self.num_heads == 0
        assert self.connectivity in ['sequential', 'residual', 'dense', 'skip']
        assert self.attention_type in ['standard', 'linear', 'sparse', 'neuralese']
        assert self.ffn_type in ['standard', 'gated', 'moe', 'reversible']
        assert self.physics_mode in ['none', 'reversible', 'sparse', 'adiabatic', 'spiking', 'hybrid']
        
    def mutate(self, mutation_rate=0.3):
        new = copy.deepcopy(self)
        
        if random.random() < mutation_rate:
            new.depth = max(1, min(12, new.depth + random.choice([-1, 1])))
        if random.random() < mutation_rate:
            new.width = max(128, min(1024, new.width + random.choice([-128, 128, -64, 64])))
            new.width = (new.width // new.num_heads) * new.num_heads
            new.width = max(128, new.width)
        if random.random() < mutation_rate:
            new.connectivity = random.choice(['sequential', 'residual', 'dense', 'skip'])
        if random.random() < mutation_rate:
            new.attention_type = random.choice(['standard', 'linear', 'sparse', 'neuralese'])
        if random.random() < mutation_rate:
            new.ffn_type = random.choice(['standard', 'gated', 'moe', 'reversible'])
        if random.random() < mutation_rate:
            new.physics_mode = random.choice(['none', 'reversible', 'sparse', 'adiabatic', 'spiking', 'hybrid'])
        if random.random() < mutation_rate:
            new.num_heads = random.choice([4, 8, 16])
            new.width = (new.width // new.num_heads) * new.num_heads
            
        new._validate()
        return new
    
    def crossover(self, other):
        new = copy.deepcopy(self)
        
        genes = ['depth', 'width', 'connectivity', 'attention_type', 'ffn_type', 'physics_mode', 'num_heads']
        for gene in genes:
            if random.random() < 0.5:
                setattr(new, gene, getattr(other, gene))
                
        new._validate()
        return new
    
    def to_dict(self):
        return {
            'depth': self.depth,
            'width': self.width,
            'connectivity': self.connectivity,
            'attention_type': self.attention_type,
            'ffn_type': self.ffn_type,
            'physics_mode': self.physics_mode,
            'num_heads': self.num_heads
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def __repr__(self):
        return f"Genome(d={self.depth}, w={self.width}, conn={self.connectivity}, attn={self.attention_type}, ffn={self.ffn_type}, phys={self.physics_mode}, heads={self.num_heads})"


class DynamicArchitecture(nn.Module):
    """Neural network built from genome at runtime"""
    def __init__(self, genome: ArchitectureGenome, vocab_size=32000):
        super().__init__()
        self.genome = genome
        self.vocab_size = vocab_size
        
        self.build_from_genome()
        
    def build_from_genome(self):
        g = self.genome
        self.embedding = nn.Embedding(self.vocab_size, g.width)
        self.pos_emb = nn.Parameter(torch.randn(512, g.width) * 0.02)
        
        self.layers = nn.ModuleList()
        for i in range(g.depth):
            layer = self._build_layer(i, g)
            self.layers.append(layer)
            
        self.norm = nn.LayerNorm(g.width)
        self.output = nn.Linear(g.width, self.vocab_size)
        
    def _build_layer(self, idx, genome):
        width = genome.width
        heads = genome.num_heads
        
        if genome.attention_type == 'standard':
            attn = nn.MultiheadAttention(width, heads, batch_first=True)
        elif genome.attention_type == 'linear':
            attn = LinearAttention(width, heads)
        elif genome.attention_type == 'sparse':
            attn = SparseAttention(width, heads)
        elif genome.attention_type == 'neuralese':
            attn = LatentAttention(width, heads)
        else:
            attn = nn.MultiheadAttention(width, heads, batch_first=True)
            
        if genome.ffn_type == 'standard':
            ffn = nn.Sequential(
                nn.Linear(width, width * 4),
                nn.GELU(),
                nn.Linear(width * 4, width)
            )
        elif genome.ffn_type == 'gated':
            ffn = GatedFFN(width)
        elif genome.ffn_type == 'moe':
            ffn = MoEFFN(width)
        elif genome.ffn_type == 'reversible':
            ffn = ReversibleFFN(width)
        else:
            ffn = nn.Sequential(
                nn.Linear(width, width * 4),
                nn.GELU(),
                nn.Linear(width * 4, width)
            )
            
        if genome.physics_mode == 'reversible':
            physics = ReversiblePhysics(width)
        elif genome.physics_mode == 'sparse':
            physics = SparsePhysics(width)
        elif genome.physics_mode == 'adiabatic':
            physics = AdiabaticPhysics(width)
        elif genome.physics_mode == 'spiking':
            physics = SpikingPhysics(width)
        elif genome.physics_mode == 'hybrid':
            physics = HybridPhysics(width)
        else:
            physics = nn.Identity()
            
        return DynamicLayer(attn, ffn, physics, genome.connectivity, width)
    
    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        x = self.embedding(input_ids) + self.pos_emb[:seq_len]
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        logits = self.output(x)
        return logits


class DynamicLayer(nn.Module):
    def __init__(self, attention, ffn, physics, connectivity, width):
        super().__init__()
        self.attention = attention
        self.ffn = ffn
        self.physics = physics
        self.connectivity = connectivity
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        
    def forward(self, x):
        if self.connectivity == 'residual':
            x = x + self.attention(self.norm1(x))[0] if isinstance(self.attention, nn.MultiheadAttention) else x + self.attention(self.norm1(x))
        elif self.connectivity == 'dense':
            attn_out = self.attention(self.norm1(x))[0] if isinstance(self.attention, nn.MultiheadAttention) else self.attention(self.norm1(x))
            x = torch.cat([x, attn_out], dim=-1)
            x = nn.Linear(x.size(-1), x.size(-1) // 2).to(x.device)(x)
        elif self.connectivity == 'skip':
            attn_out = self.attention(self.norm1(x))[0] if isinstance(self.attention, nn.MultiheadAttention) else self.attention(self.norm1(x))
            x = x + attn_out
        else:
            x = self.attention(self.norm1(x))[0] if isinstance(self.attention, nn.MultiheadAttention) else self.attention(self.norm1(x))
            
        x = x + self.ffn(self.norm2(x))
        x = self.physics(x)
        return x


# Attention variants
class LinearAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)
        
    def forward(self, x):
        batch, seq, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, seq, self.heads, self.head_dim).transpose(1, 2) for t in qkv]
        kv = torch.matmul(k.transpose(-2, -1), v)
        out = torch.matmul(q, kv)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.out(out)

class SparseAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        
    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out

class LatentAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        
    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out


# FFN variants
class GatedFFN(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = nn.Linear(width, width * 4)
        self.up = nn.Linear(width, width * 4)
        self.down = nn.Linear(width * 4, width)
        
    def forward(self, x):
        return self.down(F.gelu(self.gate(x)) * self.up(x))

class MoEFFN(nn.Module):
    def __init__(self, width, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(width, num_experts)
        
    def forward(self, x):
        weights = F.softmax(self.router(x), dim=-1)
        out = sum(w.unsqueeze(-1) * exp(x) for w, exp in zip(weights.unbind(-1), self.experts))
        return out

class ReversibleFFN(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width)
        )
        
    def forward(self, x):
        return self.net(x)


# Physics variants
class ReversiblePhysics(nn.Module):
    def __init__(self, width):
        super().__init__()
        Q = torch.randn(width, width)
        Q, _ = torch.linalg.qr(Q)
        self.register_buffer('orthogonal', Q)
        
    def forward(self, x):
        return x @ self.orthogonal

class SparsePhysics(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.sparse_proj = nn.Linear(width, width)
        
    def forward(self, x):
        return F.softshrink(self.sparse_proj(x), 0.1)

class AdiabaticPhysics(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.phase = nn.Parameter(torch.ones(4) * 0.25)
        
    def forward(self, x):
        phase = F.softmax(self.phase, dim=0)
        return phase[0] * x + phase[1] * x + phase[2] * x + phase[3] * x

class SpikingPhysics(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(1.0))
        self.decay = nn.Parameter(torch.tensor(0.9))
        
    def forward(self, x):
        spike = (x > self.threshold).float()
        return spike * x

class HybridPhysics(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.rev = ReversiblePhysics(width)
        self.sparse = SparsePhysics(width)
        self.adiabatic = AdiabaticPhysics(width)
        self.weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        w = F.softmax(self.weights, dim=0)
        return w[0] * self.rev(x) + w[1] * self.sparse(x) + w[2] * self.adiabatic(x)


# Architecture-Mutating Merge-Chain RSI
class ArchMergeChainRSI:
    def __init__(self, population_size=8, elite_fraction=0.25):
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.generation = 0
        self.soup_weights = {}
        self.soup_genomes = []
        
    def initialize_population(self):
        population = []
        for i in range(self.population_size):
            genome = ArchitectureGenome(
                depth=random.randint(2, 6),
                width=random.choice([256, 384, 512, 640, 768]),
                connectivity=random.choice(['residual', 'dense', 'skip']),
                attention_type=random.choice(['standard', 'linear', 'sparse']),
                ffn_type=random.choice(['standard', 'gated', 'moe']),
                physics_mode=random.choice(['none', 'reversible', 'sparse', 'hybrid']),
                num_heads=random.choice([4, 8])
            )
            model = DynamicArchitecture(genome)
            population.append({
                'genome': genome,
                'model': model,
                'fitness': 0.0,
                'id': f"gen0_agent_{i}"
            })
        return population
    
    def evaluate_population(self, population, task_fn):
        for agent in population:
            agent['fitness'] = task_fn(agent['model'])
        population.sort(key=lambda a: a['fitness'], reverse=True)
        return population
    
    def select_elites(self, population):
        n_elite = max(1, int(self.population_size * self.elite_fraction))
        return population[:n_elite]
    
    def merge_chain_weights(self, elites):
        layer_soups = {}
        
        for elite in elites:
            model = elite['model']
            for i, layer in enumerate(model.layers):
                layer_key = f'layer_{i}_{type(layer.attention).__name__}_{type(layer.ffn).__name__}'
                if layer_key not in layer_soups:
                    layer_soups[layer_key] = []
                layer_soups[layer_key].append({
                    'state_dict': layer.state_dict(),
                    'fitness': elite['fitness']
                })
        
        merged_soups = {}
        for key, states in layer_soups.items():
            if len(states) > 1:
                total_fit = sum(s['fitness'] for s in states)
                merged = {}
                for param_name in states[0]['state_dict']:
                    merged[param_name] = sum(
                        s['state_dict'][param_name] * s['fitness'] / total_fit 
                        for s in states
                    )
                merged_soups[key] = merged
                
        return merged_soups
    
    def crossover_architectures(self, elites):
        new_genomes = []
        
        for elite in elites:
            new_genomes.append(elite['genome'])
            
        while len(new_genomes) < self.population_size:
            parent1 = random.choice(elites)['genome']
            parent2 = random.choice(elites)['genome']
            child = parent1.crossover(parent2)
            child = child.mutate(0.3)
            new_genomes.append(child)
            
        return new_genomes
    
    def build_next_generation(self, new_genomes, weight_soups):
        population = []
        for i, genome in enumerate(new_genomes):
            model = DynamicArchitecture(genome)
            
            for layer_idx, layer in enumerate(model.layers):
                layer_key = f'layer_{layer_idx}_{type(layer.attention).__name__}_{type(layer.ffn).__name__}'
                if layer_key in weight_soups:
                    try:
                        layer.load_state_dict(weight_soups[layer_key], strict=False)
                    except:
                        pass
                        
            population.append({
                'genome': genome,
                'model': model,
                'fitness': 0.0,
                'id': f"gen{self.generation}_agent_{i}"
            })
            
        return population
    
    def run_generation(self, population, task_fn):
        self.generation += 1
        print(f"\n=== Generation {self.generation} ===")
        
        population = self.evaluate_population(population, task_fn)
        best = population[0]
        print(f"Best fitness: {best['fitness']:.4f} | Genome: {best['genome']}")
        
        elites = self.select_elites(population)
        weight_soups = self.merge_chain_weights(elites)
        new_genomes = self.crossover_architectures(elites)
        population = self.build_next_generation(new_genomes, weight_soups)
        
        return population, best


# Test task
def dummy_task(model):
    total_params = sum(p.numel() for p in model.parameters())
    optimal = 50_000_000
    fitness = 1.0 - abs(total_params - optimal) / optimal
    return max(0.1, min(1.0, fitness + random.uniform(-0.1, 0.1)))


if __name__ == "__main__":
    print("Testing Architecture-Mutating Merge-Chain RSI...")
    
    rsi = ArchMergeChainRSI(population_size=6, elite_fraction=0.33)
    population = rsi.initialize_population()
    
    print(f"Initial population:")
    for agent in population:
        params = sum(p.numel() for p in agent['model'].parameters())
        print(f"  {agent['id']}: {agent['genome']} | Params: {params:,}")
    
    for gen in range(3):
        population, best = rsi.run_generation(population, dummy_task)
        
        print(f"  Population genomes:")
        for agent in population:
            params = sum(p.numel() for p in agent['model'].parameters())
            print(f"    {agent['id']}: {agent['genome']} | Params: {params:,}")
    
    print("\nArchitecture-Mutating Merge-Chain RSI prototype working!")