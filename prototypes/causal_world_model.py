import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# EXP-004: Causal World Model via Intervention Training
# Hypothesis: Training on interventional data (do(X)=x) enables causal reasoning
# Mechanism: Synthetic environment with interventions -> learn SCM -> do-calculus for planning

class CausalEnvironment:
    """Synthetic environment with known causal structure for intervention training"""
    def __init__(self, num_vars=5, noise_scale=0.1):
        self.num_vars = num_vars
        self.noise_scale = noise_scale
        
        # True causal graph (DAG) - upper triangular for acyclicity
        self.causal_matrix = torch.triu(torch.randn(num_vars, num_vars) * 0.5, diagonal=1)
        for i in range(num_vars - 1):
            self.causal_matrix[i, i+1] = 1.5 + torch.randn(1).item() * 0.2
        self.causal_matrix = self.causal_matrix.clamp(-2, 2)
        
    def sample_observational(self, batch_size):
        X = torch.zeros(batch_size, self.num_vars)
        for i in range(self.num_vars):
            parents = self.causal_matrix[:i, i]
            if parents.abs().sum() > 0:
                parent_vals = X[:, :i]
                X[:, i] = (parent_vals * parents).sum(dim=1) + torch.randn(batch_size) * self.noise_scale
            else:
                X[:, i] = torch.randn(batch_size) * self.noise_scale
        return X
    
    def intervene(self, batch_size, interventions):
        X = torch.zeros(batch_size, self.num_vars)
        for i in range(self.num_vars):
            if i in interventions:
                X[:, i] = interventions[i]
            else:
                parents = self.causal_matrix[:i, i]
                if parents.abs().sum() > 0:
                    parent_vals = X[:, :i]
                    X[:, i] = (parent_vals * parents).sum(dim=1) + torch.randn(batch_size) * self.noise_scale
                else:
                    X[:, i] = torch.randn(batch_size) * self.noise_scale
        return X
    
    def counterfactual(self, factual_X, interventions):
        noise = torch.zeros_like(factual_X)
        for i in range(self.num_vars):
            parents = self.causal_matrix[:i, i]
            if parents.abs().sum() > 0:
                parent_vals = factual_X[:, :i]
                noise[:, i] = factual_X[:, i] - (parent_vals * parents).sum(dim=1)
            else:
                noise[:, i] = factual_X[:, i]
        
        X_cf = torch.zeros_like(factual_X)
        for i in range(self.num_vars):
            if i in interventions:
                X_cf[:, i] = interventions[i]
            else:
                parents = self.causal_matrix[:i, i]
                if parents.abs().sum() > 0:
                    parent_vals = X_cf[:, :i]
                    X_cf[:, i] = (parent_vals * parents).sum(dim=1) + noise[:, i]
                else:
                    X_cf[:, i] = noise[:, i]
        return X_cf


class CausalWorldModel(nn.Module):
    """Neural network that learns structural causal model from interventional data"""
    def __init__(self, num_vars=5, hidden_dim=128):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_vars)
        ])
        
        self.mechanisms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * num_vars, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_vars)
        ])
        
        self.intervention_encoder = nn.Sequential(
            nn.Linear(num_vars, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.causal_logits = nn.Parameter(torch.randn(num_vars, num_vars) * 0.1)
        
    def get_causal_mask(self, temperature=1.0):
        mask = torch.triu(torch.sigmoid(self.causal_logits / temperature), diagonal=1)
        return mask
    
    def forward(self, X, intervention_mask=None):
        batch = X.size(0)
        
        encoded = []
        for i in range(self.num_vars):
            enc = self.encoders[i](X[:, i:i+1])
            encoded.append(enc)
        encoded = torch.stack(encoded, dim=1)
        
        if intervention_mask is None:
            intervention_mask = torch.zeros(batch, self.num_vars, device=X.device)
        intv_emb = self.intervention_encoder(intervention_mask)
        
        causal_mask = self.get_causal_mask()
        
        predictions = []
        for i in range(self.num_vars):
            parents = (causal_mask[:i, i] > 0.1).nonzero(as_tuple=True)[0]
            
            if len(parents) > 0:
                parent_enc = encoded[:, parents]
                parent_flat = parent_enc.view(batch, -1)
            else:
                parent_flat = torch.zeros(batch, self.hidden_dim, device=X.device)
            
            mechanism_input = torch.cat([parent_flat, intv_emb], dim=-1)
            if mechanism_input.size(-1) < self.mechanisms[i][0].in_features:
                pad_size = self.mechanisms[i][0].in_features - mechanism_input.size(-1)
                mechanism_input = F.pad(mechanism_input, (0, pad_size))
            
            pred = self.mechanisms[i](mechanism_input)
            predictions.append(pred)
            
        predictions = torch.cat(predictions, dim=-1)
        
        return {
            'predictions': predictions,
            'causal_mask': causal_mask,
            'encoded': encoded
        }
    
    def intervene(self, X, interventions):
        batch = X.size(0)
        intervention_mask = torch.zeros(batch, self.num_vars, device=X.device)
        X_intv = X.clone()
        
        for var_idx, value in interventions.items():
            intervention_mask[:, var_idx] = 1
            X_intv[:, var_idx] = value
            
        return self.forward(X_intv, intervention_mask)
    
    def counterfactual(self, factual_X, interventions):
        return self.intervene(factual_X, interventions)


class CausalTrainer:
    def __init__(self, env, model, lr=1e-3):
        self.env = env
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
    def train_step(self, batch_size=256, interventional_ratio=0.5):
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        metrics = {}
        
        obs_batch = int(batch_size * (1 - interventional_ratio))
        if obs_batch > 0:
            X_obs = self.env.sample_observational(obs_batch)
            out_obs = self.model(X_obs)
            loss_obs = self.loss_fn(out_obs['predictions'], X_obs)
            total_loss += loss_obs
            metrics['obs_loss'] = loss_obs.item()
        
        intv_batch = batch_size - obs_batch
        if intv_batch > 0:
            interventions_list = []
            for _ in range(intv_batch):
                n_intv = random.randint(1, self.env.num_vars - 1)
                intv_vars = random.sample(range(self.env.num_vars), n_intv)
                interventions = {v: random.uniform(-2, 2) for v in intv_vars}
                interventions_list.append(interventions)
            
            X_intv = []
            intv_masks = []
            for interventions in interventions_list:
                var_idx = random.choice(list(interventions.keys()))
                value = interventions[var_idx]
                x = self.env.intervene(1, {var_idx: value})
                X_intv.append(x)
                mask = torch.zeros(self.env.num_vars)
                mask[var_idx] = 1
                intv_masks.append(mask)
            
            X_intv = torch.cat(X_intv, dim=0)
            intv_masks = torch.stack(intv_masks)
            
            out_intv = self.model(X_intv, intv_masks)
            loss_intv = self.loss_fn(out_intv['predictions'], X_intv)
            total_loss += loss_intv
            metrics['intv_loss'] = loss_intv.item()
        
        causal_mask = self.model.get_causal_mask()
        sparsity_loss = causal_mask.abs().mean()
        acyclic_loss = torch.trace(torch.matrix_exp(causal_mask * causal_mask)) - self.env.num_vars
        
        total_loss += 0.01 * sparsity_loss + 0.1 * acyclic_loss
        metrics['sparsity'] = sparsity_loss.item()
        metrics['acyclic'] = acyclic_loss.item()
        
        total_loss.backward()
        self.optimizer.step()
        
        metrics['total_loss'] = total_loss.item()
        return metrics
    
    def evaluate(self, num_samples=1000):
        self.model.eval()
        with torch.no_grad():
            X_obs = self.env.sample_observational(num_samples)
            out_obs = self.model(X_obs)
            obs_mse = F.mse_loss(out_obs['predictions'], X_obs).item()
            
            intv_errors = []
            for _ in range(100):
                var_idx = random.randint(0, self.env.num_vars - 1)
                value = random.uniform(-2, 2)
                X_true = self.env.intervene(10, {var_idx: value})
                X_obs_batch = self.env.sample_observational(10)
                
                out_cf = self.model.intervene(X_obs_batch, {var_idx: value})
                pred = out_cf['predictions']
                
                error = F.mse_loss(pred, X_true).item()
                intv_errors.append(error)
            
            avg_intv_error = np.mean(intv_errors)
            
            learned_mask = self.model.get_causal_mask().detach().cpu().numpy()
            true_mask = (self.env.causal_matrix.abs() > 0.1).float().cpu().numpy()
            true_mask = np.triu(true_mask, k=1)
            
            pred_mask = (learned_mask > 0.3).astype(float)
            structural_hamming = np.abs(pred_mask - true_mask).sum()
            
            return {
                'obs_mse': obs_mse,
                'intv_mse': avg_intv_error,
                'structural_hamming': structural_hamming,
                'learned_causal_mask': learned_mask
            }


if __name__ == "__main__":
    print("Testing Causal World Model via Intervention Training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    num_vars = 6
    env = CausalEnvironment(num_vars=num_vars, noise_scale=0.1)
    model = CausalWorldModel(num_vars=num_vars, hidden_dim=64).to(device)
    trainer = CausalTrainer(env, model, lr=1e-3)
    
    print(f"True causal matrix:")
    print(env.causal_matrix.numpy().round(2))
    
    print("\nTraining...")
    for epoch in range(50):
        metrics = trainer.train_step(batch_size=128, interventional_ratio=0.5)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {metrics}")
    
    print("\nEvaluating...")
    eval_results = trainer.evaluate(num_samples=500)
    print(f"Observational MSE: {eval_results['obs_mse']:.6f}")
    print(f"Interventional MSE: {eval_results['intv_mse']:.6f}")
    print(f"Structural Hamming Distance: {eval_results['structural_hamming']:.0f}")
    print(f"\nLearned causal mask:")
    print(eval_results['learned_causal_mask'].round(2))
    
    print("\nCounterfactual test:")
    X_factual = env.sample_observational(5)
    cf_result = model.counterfactual(X_factual, {0: 1.5, 2: -1.0})
    print(f"Factual X[0]: {X_factual[:3, :3]}")
    print(f"Counterfactual X[0] (do(X0=1.5, X2=-1.0)): {cf_result['predictions'][:3, :3]}")
    
    print("\nCausal World Model prototype working!")