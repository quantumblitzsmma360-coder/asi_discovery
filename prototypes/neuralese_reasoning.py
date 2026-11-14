import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Neuralese: High-Bandwidth Continuous Latent Reasoning
# Core innovation: Latent-to-latent reasoning blocks operating in continuous space
# No discrete tokenization in the reasoning core

class NeuraleseEncoder(nn.Module):
    """Maps discrete tokens to continuous latent space"""
    def __init__(self, vocab_size=32000, latent_dim=1024, max_seq_len=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.token_emb = nn.Embedding(vocab_size, latent_dim)
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, latent_dim) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.token_emb(input_ids) + self.pos_emb[:seq_len]
        return self.proj(x)

class NeuraleseDecoder(nn.Module):
    """Maps continuous latent space back to discrete tokens"""
    def __init__(self, vocab_size=32000, latent_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, vocab_size)
        )
        
    def forward(self, latent):
        return self.proj(latent)

class LatentReasoningBlock(nn.Module):
    """Single latent-to-latent reasoning block - no tokenization"""
    def __init__(self, latent_dim=1024, num_heads=16, ff_mult=4, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        
        self.q_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.k_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.v_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * ff_mult),
            nn.GELU(),
            nn.Linear(latent_dim * ff_mult, latent_dim)
        )
        
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        x = residual + self.dropout(self.out_proj(attn_output))
        
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ff(x))
        
        return x

class AdaptiveComputationTime(nn.Module):
    """Adaptive depth: halting unit decides how many reasoning steps"""
    def __init__(self, latent_dim=1024, max_steps=8, threshold=0.99):
        super().__init__()
        self.halt = nn.Linear(latent_dim, 1)
        self.max_steps = max_steps
        self.threshold = threshold
        
    def forward(self, x, reasoning_blocks):
        batch, seq_len, _ = x.shape
        halt_probs = torch.zeros(batch, seq_len, device=x.device)
        previous_h = torch.zeros_like(x)
        n_updates = torch.zeros(batch, seq_len, device=x.device)
        
        for step in range(self.max_steps):
            h = reasoning_blocks[step % len(reasoning_blocks)](x)
            p = torch.sigmoid(self.halt(h)).squeeze(-1)
            
            halt_probs = halt_probs + p
            still_running = (halt_probs < self.threshold).float()
            
            weight = p * still_running
            previous_h = previous_h + weight.unsqueeze(-1) * h
            n_updates = n_updates + still_running
            
            if (halt_probs >= self.threshold).all():
                break
                
        return previous_h, n_updates.mean()

class NeuraleseReasoner(nn.Module):
    """Complete Neuralese reasoning loop"""
    def __init__(self, vocab_size=32000, latent_dim=1024, num_blocks=4, max_act_steps=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = NeuraleseEncoder(vocab_size, latent_dim)
        self.decoder = NeuraleseDecoder(vocab_size, latent_dim)
        
        self.reasoning_blocks = nn.ModuleList([
            LatentReasoningBlock(latent_dim) for _ in range(num_blocks)
        ])
        
        self.act = AdaptiveComputationTime(latent_dim, max_act_steps)
        self.contrastive_proj = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, input_ids, targets=None):
        latent = self.encoder(input_ids)
        reasoned_latent, avg_steps = self.act(latent, self.reasoning_blocks)
        logits = self.decoder(reasoned_latent)
        contrastive_latent = F.normalize(self.contrastive_proj(reasoned_latent), dim=-1)
        
        outputs = {
            "logits": logits,
            "reasoned_latent": reasoned_latent,
            "contrastive_latent": contrastive_latent,
            "avg_reasoning_steps": avg_steps
        }
        
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            with torch.no_grad():
                encoder_latent = F.normalize(self.contrastive_proj(latent), dim=-1)
            
            sim = torch.matmul(contrastive_latent, encoder_latent.transpose(-2, -1)) / 0.07
            batch, seq_len, _ = sim.shape
            sim_flat = sim.reshape(batch * seq_len, seq_len)
            labels = torch.arange(seq_len, device=sim.device).unsqueeze(0).expand(batch, seq_len).reshape(batch * seq_len)
            contrastive_loss = F.cross_entropy(sim_flat, labels)
            
            outputs["loss"] = ce_loss + 0.1 * contrastive_loss
            outputs["ce_loss"] = ce_loss
            outputs["contrastive_loss"] = contrastive_loss
            
        return outputs

def measure_reasoning_bandwidth(model, seq_len=128, latent_dim=1024):
    """Calculate bits of reasoning per forward pass"""
    latent_bits = seq_len * latent_dim * 32
    token_bits = seq_len * 15
    num_blocks = len(model.reasoning_blocks)
    total_latent_ops = num_blocks * seq_len * latent_dim * latent_dim
    bandwidth_ratio = latent_bits / token_bits
    return {
        "latent_bits_per_pass": latent_bits,
        "token_bits_per_pass": token_bits,
        "bandwidth_ratio": bandwidth_ratio,
        "reasoning_blocks": num_blocks,
        "total_latent_flops": total_latent_ops
    }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = NeuraleseReasoner(vocab_size=32000, latent_dim=1024, num_blocks=4, max_act_steps=8).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    bw = measure_reasoning_bandwidth(model)
    print(f"Bandwidth ratio (latent/token): {bw["bandwidth_ratio"]:.1f}x")
    print(f"Latent bits per pass: {bw["latent_bits_per_pass"]:,}")
    print(f"Token bits per pass: {bw["token_bits_per_pass"]:,}")
    
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        out = model(input_ids, targets)
    
    print(f"Logits shape: {out["logits"].shape}")
    print(f"Reasoned latent shape: {out["reasoned_latent"].shape}")
    print(f"Avg reasoning steps: {out["avg_reasoning_steps"]:.2f}")
    if "loss" in out:
        print(f"Loss: {out["loss"].item():.4f} (CE: {out["ce_loss"].item():.4f}, Contrastive: {out["contrastive_loss"].item():.4f})")
    
    print("\nNeuralese prototype working!")