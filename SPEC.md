microGPT Specification (spec.md)
1. Objective

Implement a minimal GPT-2–like language model from scratch with:

No external dependencies
Full algorithmic transparency
End-to-end pipeline:
Dataset ingestion
Tokenization
Autograd engine
Transformer architecture
Adam optimizer
Training loop
Inference loop

The implementation must prioritize clarity and completeness over efficiency.

2. System Overview

The system consists of the following components:

Dataset loader (docs: list[str])
Character-level tokenizer
Scalar-based autograd engine (Value class)
Transformer model (GPT-2–like)
Optimizer (Adam)
Training loop
Inference (sampling) loop
3. Dataset
Requirements
Input dataset: docs: list[str]
Each document is a string (e.g., names dataset)
Shuffle dataset before training
Default Dataset
Source:
https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
Size: ~32,000 documents
Format: one name per line
Behavior
Strip whitespace
Ignore empty lines
Shuffle documents randomly
4. Tokenizer
Design
Character-level tokenizer
Each unique character → unique integer ID
Special Tokens
BOS (Beginning of Sequence)
ID = len(unique_chars)
Vocabulary
vocab_size = len(unique_chars) + 1
Encoding

Example:

"emma" → [BOS, e, m, m, a, BOS]
Constraints
Deterministic mapping
Reversible (token → char)
5. Autograd Engine
Core Class: Value
Attributes
data: float
grad: float
_children: tuple
_local_grads: tuple
Supported Operations
Addition
Multiplication
Power
Log
Exp
ReLU
Negation
Division
Backpropagation
Reverse topological traversal
Chain rule:
∂L/∂c += ∂v/∂c * ∂L/∂v
Requirements
Scalar-based (no tensors)
Gradient accumulation (+=)
Graph construction during forward pass
6. Model Parameters
Hyperparameters
n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head
Parameter Initialization
Gaussian distribution:
N(0, 0.08)
Parameter Groups
Token embeddings (wte)
Position embeddings (wpe)
Attention weights:
attn_wq, attn_wk, attn_wv, attn_wo
MLP weights:
mlp_fc1, mlp_fc2
Output projection (lm_head)
Flattened Parameters
params = [all Value objects]
7. Model Architecture
Core Function
gpt(token_id, pos_id, keys, values) → logits
Components
7.1 Embeddings
Token embedding lookup
Position embedding lookup
Sum → input vector
7.2 Normalization
RMSNorm:
x = x / sqrt(mean(x^2) + eps)
7.3 Attention Block

Per layer:

Compute:
Query (Q)
Key (K)
Value (V)
Append K/V to cache
Multi-head attention:
Split into heads

Compute attention logits:

(Q · K) / sqrt(head_dim)
Apply softmax
Weighted sum of V
Concatenate heads
Linear projection (attn_wo)
Residual connection
7.4 MLP Block
x → Linear → ReLU → Linear → Residual
7.5 Output
logits = Linear(x, lm_head)
8. Helper Functions
Linear

Matrix-vector multiplication

Softmax
Numerically stable
Subtract max before exponentiation
RMSNorm
Normalize using root mean square
9. Training Loop
Configuration
num_steps = 1000
learning_rate = 0.01
beta1 = 0.85
beta2 = 0.99
eps = 1e-8
Process Per Step
Select document
Tokenize with BOS boundaries
Forward pass:
Iterate token-by-token
Maintain KV cache

Compute loss:

loss = average(-log(prob(target)))

Backward pass:

loss.backward()
Parameter update (Adam)
Reset gradients
Learning Rate
Linear decay:
lr_t = lr * (1 - step / num_steps)
10. Optimizer (Adam)
State
m: first moment
v: second moment
Update Rule
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2

m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)

param -= lr * m_hat / (sqrt(v_hat) + eps)
11. Inference
Sampling Loop
Initialize with BOS
Loop until:
BOS generated OR
max length reached
At each step:
Compute logits

Apply temperature:

logits / temperature
Softmax → probabilities
Sample next token
Temperature
Controls randomness:
Low → deterministic
High → diverse
12. Expected Results
Training
Initial loss ≈ 3.3
Final loss ≈ 2.3–2.4
Output
Generates plausible synthetic names
Examples:
"kamon"
"karai"
"annel"
"kaina"
13. Execution
Requirements
Python only 
No external libraries
Run
python train.py
Runtime
~1 minute on standard laptop
14. Extensibility
Possible Improvements
Larger dataset
More layers (n_layer)
Larger embeddings (n_embd)
More heads (n_head)
Longer training
Dataset Swap
Model adapts to:
Names
Cities
Words
Poems
15. Conceptual Notes
Model = probability distribution over next token
No inherent notion of truth
Learns statistical patterns only
Same core mechanism as large LLMs (scaled)
16. Non-Goals
Efficiency
GPU support
Production scalability
Advanced tokenization (BPE)
Distributed training
17. Deliverable

A single Python script implementing:

All components above
Fully working training + inference
Minimal, readable, and self-contained