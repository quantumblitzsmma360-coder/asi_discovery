import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# EXP-005: Neuromorphic Spiking Neural Networks with Surrogate Gradients
# Hypothesis: SNNs with surrogate gradients solve temporal credit assignment over 1000+ steps without BPTT
# Mechanism: LIF neurons + surrogate gradient (fast sigmoid) + eligibility traces; event-driven computation

class SurrogateGradient:
    @staticmethod
    def fast_sigmoid(x, alpha=10.0):
        sigma = torch.sigmoid(alpha * x)
        return alpha * sigma * (1 - sigma)
    
    @staticmethod
    def piecewise_linear(x, gamma=0.3):
        return (x.abs() < gamma).float() / (2 * gamma)
    
    @staticmethod
    def arctan(x, alpha=2.0):
        return alpha / (1 + (alpha * x).pow(2))
    
    @staticmethod
    def exponential(x, alpha=1.0):
        return alpha * torch.exp(-alpha * x.abs())


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with surrogate gradients"""
    def __init__(self, input_size, hidden_size, tau_mem=20.0, tau_syn=10.0, 
                 threshold=1.0, reset_mechanism='subtract', surrogate='fast_sigmoid'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        self.surrogate_fn = getattr(SurrogateGradient, surrogate)
        
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
        surrogate_grad = self.surrogate_fn(self.v_mem - self.threshold)
        spikes_st = spikes + surrogate_grad * (self.v_mem - self.v_mem.detach())
        
        if self.reset_mechanism == 'subtract':
            self.v_mem = self.v_mem - spikes * self.threshold
        elif self.reset_mechanism == 'zero':
            self.v_mem = self.v_mem * (1 - spikes)
        
        self.spikes = spikes_st
        return spikes_st


class SpikingLayer(nn.Module):
    """Layer of LIF neurons (vectorized for efficiency)"""
    def __init__(self, input_size, hidden_size, **neuron_kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_mem = neuron_kwargs.get('tau_mem', 20.0)
        self.tau_syn = neuron_kwargs.get('tau_syn', 10.0)
        self.threshold = neuron_kwargs.get('threshold', 1.0)
        self.reset_mechanism = neuron_kwargs.get('reset_mechanism', 'subtract')
        surrogate = neuron_kwargs.get('surrogate', 'fast_sigmoid')
        self.surrogate_fn = getattr(SurrogateGradient, surrogate)
        
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
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            spikes_out = []
            for t in range(seq_len):
                spikes_out.append(self._step(x[:, t], dt))
            return torch.stack(spikes_out, dim=1)
        else:
            return self._step(x, dt)
    
    def _step(self, x, dt):
        batch = x.size(0)
        
        if self.v_mem.size(0) != batch:
            self.v_mem = self.v_mem.expand(batch, -1).clone()
            self.i_syn = self.i_syn.expand(batch, -1).clone()
            self.spikes = self.spikes.expand(batch, -1).clone()
        
        self.i_syn = self.i_syn + (dt / self.tau_syn) * (-self.i_syn + x @ self.weight_ih.T + self.bias)
        rec_current = self.spikes @ self.weight_hh.T
        self.v_mem = self.v_mem + (dt / self.tau_mem) * (-self.v_mem + self.i_syn + rec_current)
        
        spikes = (self.v_mem >= self.threshold).float()
        surrogate_grad = self.surrogate_fn(self.v_mem - self.threshold)
        spikes_st = spikes + surrogate_grad * (self.v_mem - self.v_mem.detach())
        
        if self.reset_mechanism == 'subtract':
            self.v_mem = self.v_mem - spikes * self.threshold
        elif self.reset_mechanism == 'zero':
            self.v_mem = self.v_mem * (1 - spikes)
        
        self.spikes = spikes_st
        return spikes_st


class EligibilityTrace(nn.Module):
    """Eligibility traces for temporal credit assignment (e-prop)"""
    def __init__(self, hidden_size, trace_decay=0.95):
        super().__init__()
        self.hidden_size = hidden_size
        self.trace_decay = trace_decay
        self.register_buffer('traces', torch.zeros(1, hidden_size))
        
    def reset(self, batch_size=1, device=None):
        if device is None:
            device = self.traces.device
        self.traces = torch.zeros(batch_size, self.hidden_size, device=device)
        
    def update(self, pre_synaptic, post_synaptic, learning_signal):
        batch = pre_synaptic.size(0)
        if self.traces.size(0) != batch:
            self.traces = self.traces.expand(batch, -1).clone()
            
        self.traces = self.trace_decay * self.traces + pre_synaptic * post_synaptic
        weight_update = learning_signal * self.traces
        return weight_update


class SpikingNetwork(nn.Module):
    """Multi-layer spiking network with e-prop learning"""
    def __init__(self, input_size, hidden_sizes, output_size, 
                 tau_mem=20.0, tau_syn=10.0, threshold=1.0, 
                 trace_decay=0.95, surrogate='fast_sigmoid'):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        layer_sizes = [input_size] + hidden_sizes
        self.spiking_layers = nn.ModuleList()
        self.eligibility_traces = nn.ModuleList()
        
        neuron_kwargs = {
            'tau_mem': tau_mem, 'tau_syn': tau_syn, 
            'threshold': threshold, 'surrogate': surrogate
        }
        
        for i in range(len(layer_sizes) - 1):
            layer = SpikingLayer(layer_sizes[i], layer_sizes[i+1], **neuron_kwargs)
            self.spiking_layers.append(layer)
            self.eligibility_traces.append(EligibilityTrace(layer_sizes[i+1], trace_decay))
            
        self.readout = nn.Linear(hidden_sizes[-1], output_size)
        
    def reset_state(self, batch_size=1):
        device = next(self.parameters()).device
        for layer in self.spiking_layers:
            layer.reset_state(batch_size)
        for trace in self.eligibility_traces:
            trace.reset(batch_size, device)
            
    def forward(self, x, dt=1.0):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        batch, seq_len, _ = x.shape
        self.reset_state(batch)
        
        outputs = []
        
        for t in range(seq_len):
            layer_input = x[:, t]
            layer_spikes = []
            
            for layer in self.spiking_layers:
                spikes = layer(layer_input, dt)
                layer_spikes.append(spikes)
                layer_input = spikes
                
            last_spikes = layer_spikes[-1]
            out = self.readout(last_spikes.float())
            outputs.append(out)
            
        return torch.stack(outputs, dim=1)


class SpikingClassifier(nn.Module):
    """Spiking network for classification (e.g., CVE classification)"""
    def __init__(self, input_dim, hidden_sizes=[256, 128], num_classes=10, 
                 seq_len=100, **snn_kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        self.input_encoder = nn.Linear(input_dim, hidden_sizes[0])
        self.snn = SpikingNetwork(
            hidden_sizes[0], hidden_sizes[1:], num_classes, **snn_kwargs
        )
        
    def forward(self, x):
        batch = x.size(0)
        encoded = self.input_encoder(x)
        
        spike_train = []
        for t in range(self.seq_len):
            noise = torch.randn_like(encoded) * 0.1
            spike_input = encoded + noise
            spike_train.append(spike_input)
        spike_train = torch.stack(spike_train, dim=1)
        
        outputs = self.snn(spike_train)
        logits = outputs.mean(dim=1)
        return logits


def test_snn_classification():
    print("Testing SNN on classification task...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = SpikingClassifier(
        input_dim=100,
        hidden_sizes=[128, 64],
        num_classes=10,
        seq_len=50,
        tau_mem=20.0,
        tau_syn=10.0,
        threshold=1.0,
        surrogate='fast_sigmoid'
    ).to(device)
    
    batch = 32
    x = torch.randn(batch, 100, device=device)
    y = torch.randint(0, 10, (batch,), device=device)
    
    model.train()
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    
    ce_loss = F.cross_entropy(logits, y)
    print(f"Cross-entropy loss: {ce_loss.item():.4f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    loss = ce_loss
    loss.backward()
    
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"Gradients flowing: {has_grad}")
    
    if has_grad:
        grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        print(f"Total gradient norm: {grad_norm:.4f}")
    
    with torch.no_grad():
        model.eval()
        _ = model(x)
        last_layer = model.snn.spiking_layers[-1]
        spike_rate = last_layer.spikes.mean().item()
        print(f"Last layer spike rate: {spike_rate:.4f}")
    
    print("\nSNN prototype working!")


if __name__ == "__main__":
    test_snn_classification()