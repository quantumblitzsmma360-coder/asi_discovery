FROM nvidia/cuda:11.8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TOR_PROXY=socks5://127.0.0.1:9052

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    tor curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Tor
RUN mkdir -p /etc/tor && echo "SOCKSPort 9052
ControlPort 9051
RunAsDaemon 1" > /etc/tor/torrc

WORKDIR /app

# Copy source
COPY reflective_neuralese.py /app/
COPY causal_world_model.py /app/
COPY physics_enhanced_reflective_neuralese.py /app/
COPY reflective_merge_chain_rsi.py /app/

# Install Python deps
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    numpy pandas scikit-learn tqdm huggingface_hub

# Start Tor and run
CMD service tor start && sleep 10 && python3 -c "
import os, torch
os.environ['TOR_PROXY'] = 'socks5://127.0.0.1:9052'
from reflective_neuralese import ReflectiveNeuralese

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

model = ReflectiveNeuralese(latent_dim=256, num_blocks=3).to(device)
goal = torch.zeros(1, 5000, device=device)
goal[0, 1] = 1.0; goal[0, 2] = 1.0

model.train()
for i in range(1000):
    logits, cls, mods, _ = model(torch.randint(0, 1000, (4, 32), device=device), goal, self_mod_steps=1)
    if i % 100 == 0:
        print(f'Step {i}: {mods}')
