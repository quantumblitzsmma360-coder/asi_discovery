# EXP-001: Dynamic Architecture via Differentiable Topology (H1)

## Hypothesis
Architecture can be represented as a continuous, differentiable topology that evolves during forward pass via learned gating and routing. Each layer has a learned "existence probability" and "connection weights" to other layers. The forward pass computes a soft architecture. Gradients flow through architecture parameters.

## Falsifiable Prediction
On variable-complexity tasks, dynamic architecture achieves same accuracy with 50% fewer FLOPs on average vs fixed architecture of equivalent max capacity.

## Prototype Design

### Core Components
1. **Layer Existence Gates**: Each potential layer has a learned gate g_i in [0,1] (sigmoid of parameter)
2. **Connection Routing**: Learnable adjacency matrix A_ij for connections between layers
3. **Dynamic Depth**: Effective depth = sum(g_i). Varies per input.
4. **Differentiable Forward**: Soft execution: x = sum_i g_i * Layer_i(x) + sum_{i,j} A_ij * Layer_j(Layer_i(x))

### Architecture
- Max 12 layers (potential)
- Each layer: 512-dim, FFN + residual
- Gates initialized to favor shallow (bias=-2 -> ~0.12)
- Connections initialized as residual-like (identity + noise)
- Total params: ~12 * (512*2048*2) = ~25M (but only active subset used)

### Training
- Loss = task_loss + lambda_flops * expected_flops
- lambda_flops annealed from 0 to 1e-6
- Expected FLOPs = sum_i g_i * flops_per_layer + sum_{i,j} A_ij * flops_per_connection
- Baseline: Fixed 6-layer transformer (same max capacity)

### Test Tasks
1. Variable-depth reasoning: Synthetic tasks requiring 1-8 reasoning steps
2. Adaptive computation: Input complexity predictor
3. FLOPs vs accuracy tradeoff curve

## Implementation
