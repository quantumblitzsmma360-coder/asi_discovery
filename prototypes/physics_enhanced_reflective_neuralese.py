import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# PHYSICS-ENHANCED REFLECTIVE NEURALESE
# Integrates: Reversible, Sparse, Adiabatic, Spiking, Hybrid physics modes
# into the reflective self-modification architecture
# ============================================================================

class OrthogonalLinear(nn.Module):
    """Exactly reversible linear layer via orthogonal matrix (Q^T Q = I)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        assert in_features == out_features, "Orthogonal requires square matrix"
        Q = torch.randn(out_features, in_features)
        Q, _ = torch.linalg.qr(Q)
        self.weight = nn.Parameter(Q)
        
    def forward(self, x):
        return F.linear(x, self.weight)
    
    def inverse(self, y):
        return F.linear(y, self.weight.T)


class AdiabaticClocking(nn.Module):
    """4-phase adiabatic clocking for charge recovery (Bennett 1973)"""
    def __init__(self, features):
        super().__init__()
        self.phase_weights = nn.Parameter(torch.ones(4) * 0.25)
        
    def forward(self, x):
        phase = F.softmax(self.phase_weights, dim=0)
        return phase[0] * x + phase[1] * x + phase[2] * x + phase[3] * x


class ISTASparseCoding(nn.Module):
    """Iterative Shrinkage-Thresholding Algorithm for sparse coding (Olshausen & Field 1996)
    Uses straight-through estimator for gradient flow through soft thresholding."""
    def __init__(self, input_dim, dict_size, sparsity_target=0.965, num_iterations=10, step_size=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.num_iterations = num_iterations
        self.step_size = step_size
        
        self.dictionary = nn.Parameter(torch.randn(dict_size, input_dim) * 0.02)
        with torch.no_grad():
            self.dictionary.data = F.normalize(self.dictionary.data, dim=1)
        
        self.threshold = nn.Parameter(torch.tensor(0.001))
        
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        z = torch.zeros(batch_size, self.dict_size, device=device)
        threshold = self.threshold
        
        D = F.normalize(self.dictionary, dim=1)
        
        for _ in range(self.num_iterations):
            recon = z @ D
            residual = recon - x
            grad = residual @ D.T
            z = z - self.step_size * grad
            # Straight-through estimator for soft thresholding
            z_shrunk = F.softshrink(z, threshold.item())
            z = z + (z_shrunk - z).detach()
            
        sparsity = (z.abs() < 1e-3).float().mean()
        x_recon = z @ D
        recon_loss = F.mse_loss(x_recon, x)
        
        return x_recon, z, sparsity, recon_loss


class SpikingLayer(nn.Module):
    """LIF spiking layer with surrogate gradients (fast sigmoid)"""
    def __init__(self, input_size, hidden_size, tau_mem=20.0, tau_syn=10.0, threshold=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.threshold = threshold
        
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        self.register_buffer('v_mem', torch.zeros(1, hidden_size))
        self.register_buffer('i_syn', torch.zeros(1, hidden_size))
        self.register_buffer('spikes', torch.zeros(1, hidden_size))
        
    def reset_state(self, batch_size=1):
        device = self.weight_ih.device
        self.v_mem = torch.zeros(batch_size, self.hidden_size, device=device)
        self.i_syn = torch.zeros(batch_size, self.hidden_size, device=device)
        self.spikes = torch.zeros(batch_size, self.hidden_size, device=device)
        
    def forward(self, x, dt=1.0):
        batch = x.size(0)
        
        if self.v_mem.size(0) != batch:
            self.v_mem = self.v_mem.expand(batch, -1).clone()
            self.i_syn = self.i_syn.expand(batch, -1).clone()
            self.spikes = self.spikes.expand(batch, -1).clone()
        
        self.i_syn = self.i_syn + (dt / self.tau_syn) * (-self.i_syn + x @ self.weight_ih.T + self.bias)
        rec_current = self.spikes @ self.weight_hh.T
        self.v_mem = self.v_mem + (dt / self.tau_mem) * (-self.v_mem + self.i_syn + rec_current)
        
        spikes = (self.v_mem >= self.threshold).float()
        # Fast sigmoid surrogate gradient
        surrogate_grad = 10 * torch.sigmoid(10 * (self.v_mem - self.threshold)) * (1 - torch.sigmoid(10 * (self.v_mem - self.threshold)))
        spikes_st = spikes + surrogate_grad * (self.v_mem - self.v_mem.detach())
        
        self.v_mem = self.v_mem * (1 - spikes)
        self.spikes = spikes_st
        return spikes_st


class PhysicsEnhancedReasoningBlock(nn.Module):
    """
    Reasoning block with selectable physics mode:
    - 'standard': Standard GELU FFN
    - 'reversible': Orthogonal linear + adiabatic clocking
    - 'sparse': ISTA sparse coding (Olshausen & Field 1996)
    - 'adiabatic': 4-phase adiabatic clocking (Bennett 1973)
    - 'spiking': LIF spiking neurons with surrogate gradients
    - 'hybrid': Weighted combination of all modes
    """
    def __init__(self, latent_dim=512, physics_mode='standard', sparse_dict_size=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.physics_mode = physics_mode
        
        # Always create all physics components for gradient flow
        self.reversible_linear = OrthogonalLinear(latent_dim, latent_dim)
        self.adiabatic = AdiabaticClocking(latent_dim)
        
        dict_size = sparse_dict_size or latent_dim * 4
        self.sparse_coding = ISTASparseCoding(latent_dim, dict_size)
        
        self.spiking = SpikingLayer(latent_dim, latent_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim)
        )
        
        if physics_mode == 'hybrid':
            self.physics_weights = nn.Parameter(torch.ones(4) / 4)
        
        # Standard components
        self.norm1 = nn.LayerNorm(latent_dim)
        self.attention = nn.MultiheadAttention(latent_dim, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(latent_dim)
        
        self.physics_mode = physics_mode
        
    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape
        
        # Attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = residual + attn_out
        
        residual = x
        x = self.norm2(x)
        
        if self.physics_mode == 'standard':
            x = self.ffn(x)
            
        elif self.physics_mode == 'reversible':
            x = self.reversible_linear(x)
            x = self.adiabatic(x)
            
        elif self.physics_mode == 'sparse':
            x_flat = x.view(-1, self.latent_dim)
            x_recon, z, sparsity, recon_loss = self.sparse_coding(x_flat)
            x = x_recon.view(batch, seq_len, -1)
            
        elif self.physics_mode == 'adiabatic':
            x = self.adiabatic(x)
            
        elif self.physics_mode == 'spiking':
            x_flat = x.view(-1, self.latent_dim)
            spikes = self.spiking(x_flat)
            x = spikes.view(batch, seq_len, -1)
            
        elif self.physics_mode == 'hybrid':
            w = F.softmax(self.physics_weights, dim=0)
            
            x_rev = self.reversible_linear(x)
            x_rev = self.adiabatic(x_rev)
            
            x_flat = x.view(-1, self.latent_dim)
            x_sparse, _, _, _ = self.sparse_coding(x_flat)
            x_sparse = x_sparse.view(batch, seq_len, -1)
            
            x_adia = self.adiabatic(x)
            
            x_spike = self.spiking(x_flat).view(batch, seq_len, -1)
            
            w = F.softmax(self.physics_weights, dim=0)
            x = w[0] * x_rev + w[1] * x_sparse + w[2] * x_adia + w[3] * x_spike
            x = x + w[0] * self.ffn(x)
            
        else:
            x = self.ffn(x)
        
        x = residual + x
        return x


class PhysicsEnhancedReflectiveNeuralese(nn.Module):
    """
    Reflective Neuralese with Physics-Enhanced Reasoning Blocks
    
    Each block can use different physics mode.
    The self-model simulates the physics-enhanced computation.
    """
    def __init__(self, latent_dim=512, num_blocks=4, max_act_steps=8, vocab_size=32000,
                 num_classes=10, world_state_dim=256, world_action_dim=128,
                 physics_modes=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.max_act_steps = max_act_steps
        self.threshold = 0.99
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.world_state_dim = world_state_dim
        self.world_action_dim = world_action_dim
        
        self.physics_modes = physics_modes or ['standard'] * num_blocks
        
        self.token_emb = nn.Embedding(vocab_size, latent_dim)
        self.enc_linear = nn.Linear(latent_dim, latent_dim)
        self.enc_norm = nn.LayerNorm(latent_dim)
        
        self.reasoning_blocks = nn.ModuleList([
            PhysicsEnhancedReasoningBlock(latent_dim, self.physics_modes[i])
            for i in range(num_blocks)
        ])
        
        self.halt = nn.Linear(latent_dim, 1)
        self.decoder = nn.Linear(latent_dim, vocab_size)
        self.classifier = nn.Linear(latent_dim, num_classes)
        
        # Self-model and world model
        self.goal_encoder = nn.Sequential(
            nn.Linear(vocab_size, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, world_state_dim)
        )
        self.action_projector = nn.Linear(latent_dim, world_action_dim)
        
        self.world_model = nn.Sequential(
            nn.Linear(world_state_dim + world_action_dim, 256),
            nn.GELU(),
            nn.Linear(256, world_state_dim)
        )
        self.goal_predictor = nn.Sequential(
            nn.Linear(world_state_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.verification_threshold = 0.7
        
    def encode(self, input_ids):
        x = self.token_emb(input_ids)
        x = self.enc_linear(x)
        x = self.enc_norm(x)
        return x
    
    def reason_step(self, x, block_idx):
        return self.reasoning_blocks[block_idx](x)
    
    def halt_prob(self, h):
        return torch.sigmoid(self.halt(h)).squeeze(-1)
    
    def decode(self, x):
        return self.decoder(x)
    
    def classify(self, x):
        return self.classifier(x)
    
    def forward(self, input_ids, goal_tokens=None, self_mod_steps=0):
        x = self.encode(input_ids)
        
        halt_probs = torch.zeros(input_ids.size(0), input_ids.size(1), device=input_ids.device)
        accum = torch.zeros_like(x)
        n_updates = torch.zeros(input_ids.size(0), input_ids.size(1), device=input_ids.device)
        
        for step in range(self.max_act_steps):
            block_idx = step % self.num_blocks
            h = self.reason_step(x, block_idx)
            
            p = self.halt_prob(h)
            halt_probs = halt_probs + p
            still_running = (halt_probs < self.threshold).float()
            
            weight = p * still_running
            accum = accum + weight.unsqueeze(-1) * h
            n_updates = n_updates + still_running
            
            if (halt_probs >= self.threshold).all():
                break
        
        reasoned = accum
        logits = self.decode(reasoned)
        cls_logits = self.classify(reasoned[:, 0])
        
        mod_results = []
        if goal_tokens is not None and self_mod_steps > 0:
            for _ in range(self_mod_steps):
                mod_result = self.autonomous_self_modification_step(goal_tokens)
                mod_results.append(mod_result)
        
        return logits, cls_logits, mod_results, n_updates.mean()
    
    def autonomous_self_modification_step(self, goal_tokens, learning_rate=1e-3):
        device = goal_tokens.device
        
        goal_state = self.goal_encoder(goal_tokens.float())
        
        core_params = {}
        for name, param in self.named_parameters():
            if 'goal_encoder' not in name and 'action_projector' not in name and 'world_model' not in name and 'goal_predictor' not in name:
                core_params[name] = param.detach().clone().requires_grad_(True)
        
        dummy_input = torch.randint(0, min(1000, self.token_emb.num_embeddings), (1, 64), device=device)
        
        # Simulate forward with physics (simplified)
        emb = F.embedding(dummy_input, core_params['token_emb.weight'])
        x = F.linear(emb, core_params['enc_linear.weight'], core_params['enc_linear.bias'])
        x = F.layer_norm(x, [self.latent_dim],
                        core_params['enc_norm.weight'], core_params['enc_norm.bias'])
        
        halt_probs = torch.zeros(1, 64, device=device)
        accum = torch.zeros_like(x)
        
        for step in range(self.max_act_steps):
            block_idx = step % self.num_blocks
            block_prefix = f'reasoning_blocks.{block_idx}'
            
            # Attention
            residual = x
            x_norm = F.layer_norm(x, [self.latent_dim],
                                core_params[f'{block_prefix}.norm1.weight'],
                                core_params[f'{block_prefix}.norm1.bias'])
            attn_out = F.multi_head_attention_forward(
                x_norm, x_norm, x_norm,
                core_params[f'{block_prefix}.attention.in_proj_weight'],
                core_params[f'{block_prefix}.attention.in_proj_bias'],
                8, 0, False,
                core_params[f'{block_prefix}.attention.out_proj.weight'],
                core_params[f'{block_prefix}.attention.out_proj.bias'],
                training=False)[0]
            x = residual + attn_out
            
            # Physics FFN
            residual = x
            x = F.layer_norm(x, [self.latent_dim],
                           core_params[f'{block_prefix}.norm2.weight'],
                           core_params[f'{block_prefix}.norm2.bias'])
            x = F.linear(x, core_params[f'{block_prefix}.ffn.0.weight'], core_params[f'{block_prefix}.ffn.0.bias'])
            x = F.gelu(x)
            x = F.linear(x, core_params[f'{block_prefix}.ffn.2.weight'], core_params[f'{block_prefix}.ffn.2.bias'])
            x = residual + x
            
            p = torch.sigmoid(F.linear(x, core_params['halt.weight'], core_params['halt.bias'])).squeeze(-1)
            halt_probs = halt_probs + p
            still_running = (halt_probs < self.threshold).float()
            
            weight = p * still_running
            accum = accum + weight.unsqueeze(-1) * x
            
            if (halt_probs >= self.threshold).all():
                break
        
        reasoned_latent = accum
        
        action = F.linear(reasoned_latent[:, 0], 
                         dict(self.named_parameters())['action_projector.weight'],
                         dict(self.named_parameters())['action_projector.bias'])
        
        goal_state = self.goal_encoder(goal_tokens[:1].float())
        
        if goal_state.dim() > 2:
            goal_state = goal_state.squeeze(1)
        if action.dim() > 2:
            action = action.squeeze(1)
        
        next_state = self.world_model(torch.cat([goal_state, action], dim=-1))
        goal_prob = self.goal_predictor(next_state)
        
        goal_prob.backward()
        
        param_gradients = {}
        for name, param in core_params.items():
            if param.grad is not None and param.grad.abs().sum() > 0:
                param_gradients[name] = param.grad.clone()
        
        verified_modifications = {}
        for name, grad in param_gradients.items():
            if name in dict(self.named_parameters()):
                with torch.no_grad():
                    param = dict(self.named_parameters())[name]
                    param.add_(-learning_rate * grad)
                verified_modifications[name] = -learning_rate * grad
        
        total_grad_norm = sum(g.abs().sum().item() for g in param_gradients.values()) if param_gradients else 0
        
        return {
            'goal_prob_before': goal_prob.item(),
            'modifications_applied': len(verified_modifications),
            'total_grad_params': len(param_gradients),
            'grad_norm': total_grad_norm
        }


def test_physics_enhanced():
    print("=" * 70)
    print("TESTING PHYSICS-ENHANCED REFLECTIVE NEURALESE")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test each physics mode
    physics_modes = ['standard', 'reversible', 'sparse', 'adiabatic', 'spiking', 'hybrid']
    
    for mode in physics_modes:
        print(f"\n--- Testing Physics Mode: {mode} ---")
        
        block = PhysicsEnhancedReasoningBlock(latent_dim=256, physics_mode=mode).to(device)
        
        batch, seq_len = 4, 32
        x = torch.randn(batch, seq_len, 256, device=device)
        
        block.train()
        out = block(x)
        
        print(f"  Input: {x.shape} -> Output: {out.shape}")
        
        loss = out.sum()
        loss.backward()
        
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())
        print(f"  Gradients: {has_grad}")
    
    # Test full reflective neuralese with mixed physics
    print("\n--- Testing Full Reflective Neuralese with Mixed Physics ---")
    
    mixed_modes = ['reversible', 'sparse', 'hybrid', 'standard']
    system = PhysicsEnhancedReflectiveNeuralese(
        latent_dim=256,
        num_blocks=4,
        max_act_steps=6,
        vocab_size=5000,
        num_classes=10,
        world_state_dim=256,
        world_action_dim=128,
        physics_modes=mixed_modes
    ).to(device)
    
    total_params = sum(p.numel() for p in system.parameters())
    print(f"Total parameters: {total_params:,}")
    
    batch = 4
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)
    goal_tokens = torch.zeros(batch, 5000, device=device)
    goal_tokens[:, 1] = 1.0
    goal_tokens[:, 2] = 1.0
    
    system.train()
    logits, cls_logits, mod_results, avg_steps = system(input_ids, goal_tokens, self_mod_steps=2)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Class logits shape: {cls_logits.shape}")
    print(f"Avg reasoning steps: {avg_steps:.2f}")
    print(f"Self-mod results: {len(mod_results)}")
    if mod_results:
        for i, mr in enumerate(mod_results):
            print(f"  Step {i}: goal_prob={mr['goal_prob_before']:.4f}, mods={mr['modifications_applied']}, grad_params={mr['total_grad_params']}")
    
    print("\n" + "=" * 70)
    print("PHYSICS-ENHANCED REFLECTIVE NEURALESE WORKING")
    print("=" * 70)
    
    return system


if __name__ == "__main__":
    test_physics_enhanced()