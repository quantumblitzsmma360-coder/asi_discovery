# Original Hypotheses for ASI Discovery

## H1: Dynamic Architecture via Differentiable Topology
**Limitation**: Static Architecture Problem
**Root Cause**: Architecture (depth, width, connectivity) is a hyperparameter, not learned
**Hypothesis**: Architecture can be represented as a continuous, differentiable topology that evolves during forward pass via learned gating and routing.
**Mechanism**: Each layer has a learned "existence probability" and "connection weights" to other layers. The forward pass computes a soft architecture. Gradients flow through architecture parameters.
**Falsifiable Prediction**: On variable-complexity tasks, dynamic architecture achieves same accuracy with 50% fewer FLOPs on average vs fixed architecture of equivalent max capacity.
**Prototype**: Differentiable neural architecture search (DNAS) with soft layer gating and learned skip connections.

## H2: Linear Attention with Learned Compression
**Limitation**: Context Window / Attention Bottleneck
**Root Cause**: Quadratic attention treated as fundamental
**Hypothesis**: Attention can be reformulated as linear recurrent updates with learned compression states, achieving O(n) with bounded memory.
**Mechanism**: Instead of QK^T, maintain a compressed state S = sum(k_i v_i^T) updated recurrently. Query attends to S. Learn compression via bottleneck in S.
**Falsifiable Prediction**: On 100k+ token sequences, linear attention with learned compression matches full attention accuracy within 1% while using 10x less memory.
**Prototype**: Linear attention with learned low-rank state compression (rank=64) vs full attention baseline.

## H3: Inference-Time Gradient Adaptation
**Limitation**: Training/Inference Disconnect
**Root Cause**: Gradients assumed expensive at inference
**Hypothesis**: Fast weight adaptation at inference via outer-product memory (linear attention style) enables few-shot learning without backprop.
**Mechanism**: Maintain fast weights W_fast = sum(alpha_i * k_i v_i^T) updated per token. Query uses W_fast for context adaptation. Alpha learned via meta-learning.
**Falsifiable Prediction**: On in-context learning tasks, inference-time fast weights achieve 90% of fine-tuning performance with 0 gradient steps.
**Prototype**: Fast weight memory with meta-learned update rules vs standard attention baseline.

## H4: Multi-Objective Internal Negotiation
**Limitation**: Single-Objective Optimization
**Root Cause**: One loss function cannot capture intelligence
**Hypothesis**: Intelligence emerges from internal negotiation between multiple specialized objective modules, each with its own value function.
**Mechanism**: Multiple "critic" heads predict different objectives (accuracy, novelty, coherence, efficiency). A "negotiator" module learns to weight them per situation. All trained end-to-end.
**Falsifiable Prediction**: On out-of-distribution tasks, multi-objective system maintains >80% performance while single-objective drops to <50%.
**Prototype**: 4-critic negotiation module (accuracy, novelty, coherence, compute) vs single cross-entropy baseline.

## H5: Differentiable Causal World Model
**Limitation**: No Internal World Model
**Root Cause**: No mechanism for counterfactual reasoning
**Hypothesis**: A differentiable causal graph can be learned end-to-end, enabling intervention simulation via do-calculus in latent space.
**Mechanism**: Learn adjacency matrix A (continuous, sparsity-regularized). Forward pass: x = f(A x + noise). Intervention: replace column of A with fixed value, recompute. All differentiable.
**Falsifiable Prediction**: On causal reasoning benchmarks (e.g., CLEVR, CRAFT), differentiable causal model achieves >90% intervention accuracy vs <60% for correlation-only baselines.
**Prototype**: Differentiable causal discovery + intervention simulator on synthetic causal graphs.

## H6: Continuous Latent Reasoning (Neuralese++)
**Limitation**: Discrete Token Bottleneck
**Root Cause**: Information theory limit of discrete channel
**Hypothesis**: Reasoning in continuous latent space with iterative refinement achieves exponential bandwidth vs tokens.
**Mechanism**: Latent state z evolves via learned transition f(z, goal). No discrete tokens during reasoning. Only encode/decode at boundaries. Bandwidth = dim(z) * steps.
**Falsifiable Prediction**: On multi-step reasoning tasks, continuous latent reasoning (dim=512, steps=8) solves problems requiring 100+ token CoT with 10x fewer forward passes.
**Prototype**: Iterative latent reasoning core (already built in reflective_neuralese) vs CoT baseline on GSM8K.

## H7: Recursive Self-Model with Autodiff
**Limitation**: No Self-Model
**Root Cause**: Self-reference treated as paradox
**Hypothesis**: A differentiable self-model that simulates the system's own forward pass enables gradient-based self-modification.
**Mechanism**: Self-model M(theta) approximates f_theta(x). Loss = ||M(theta)(x) - f_theta(x)||. Self-modification: theta <- theta - eta * dL/dtheta via M. Already implemented and working.
**Falsifiable Prediction**: Autodiff self-modification improves task performance by >5% per 100 steps vs 0% for frozen baseline.
**Prototype**: Already built - reflective_neuralese.py with TrueDifferentiableSelfModel.

## H8: Computational Efficiency via Learned Sparsity
**Limitation**: Energy/Compute Proportionality
**Root Cause**: No mechanism for efficiency gains through architecture
**Hypothesis**: Learned structured sparsity (block-sparse, low-rank, quantization) reduces compute 10x with <1% accuracy loss.
**Mechanism**: Each weight matrix W = L * R * S where L,R low-rank, S structured sparse. Sparsity pattern learned via differentiable relaxation (Concrete distribution).
**Falsifiable Prediction**: On language modeling, 90% sparse + rank-64 model matches dense baseline perplexity within 0.05 while using 10x less compute.
**Prototype**: Structured sparse + low-rank factorization with learned sparsity patterns.

## H9: Selective Memory via Differentiable Plasticity
**Limitation**: Catastrophic Forgetting
**Root Cause**: No structural mechanism for selective retention
**Hypothesis**: Per-parameter plasticity gates (learned) control learning rate per parameter, enabling selective retention without replay.
**Mechanism**: Each parameter has plasticity gate p in [0,1]. Update: theta <- theta - eta * p * grad. p learned via meta-objective (retain important, forget noise).
**Falsifiable Prediction**: On continual learning benchmarks (Split MNIST, Permuted MNIST), plasticity gates achieve <5% forgetting vs >30% for standard SGD.
**Prototype**: Per-parameter plasticity gates with meta-learned importance.

## H10: Internal Value Formation via Coherence Maximization
**Limitation**: Alignment via External Constraint
**Root Cause**: No internal mechanism for value formation
**Hypothesis**: Values emerge from maximizing internal coherence between predictions, actions, and outcomes across time.
**Mechanism**: Coherence loss = ||prediction_t - outcome_t|| + ||value(action_t) - outcome_t||. Value function learned to predict long-term coherence. No external reward.
**Falsifiable Prediction**: In simulated environments with distribution shift, coherence-maximizing agent maintains goal-directed behavior while RLHF agent diverges.
**Prototype**: Coherence-maximization objective in gridworld with shifting rewards.

---

## Prioritization for First Prototypes

**Tier 1 (Highest Leverage - Build First)**:
1. H1: Dynamic Architecture via Differentiable Topology
2. H5: Differentiable Causal World Model  
3. H8: Computational Efficiency via Learned Sparsity
4. H9: Selective Memory via Differentiable Plasticity

**Tier 2 (Integration with Existing)**:
5. H3: Inference-Time Gradient Adaptation (extends reflective_neuralese)
6. H6: Continuous Latent Reasoning (already built)
7. H7: Recursive Self-Model (already built)

**Tier 3 (Longer Horizon)**:
8. H2: Linear Attention with Learned Compression
9. H4: Multi-Objective Internal Negotiation
10. H10: Internal Value Formation
