# MicroGPT in Go

A minimal GPT-2-like language model implementation in Go, using Apple MLX for hardware acceleration on Apple Silicon (M1-M4).

## Overview

This is a Go re-architecture of the microgpt Python implementation, designed to:

- Preserve the algorithmic essence exactly
- Improve execution model, structure, and performance
- Replace scalar autograd with tensor-based computation using MLX
- Run efficiently on macOS Apple Silicon (Metal via MLX)

## Architecture

```
microgpt/
├── cmd/
│   ├── train/          # Training CLI entrypoint
│   └── infer/          # Inference CLI entrypoint
├── dataset/            # Dataset loading and shuffling
├── tokenizer/          # Character-level tokenizer
├── tensor/             # MLX-backed tensor wrapper
├── model/              # GPT model (attention, MLP, embeddings)
├── optimizer/          # Adam optimizer
├── train/              # Training loop
└── inference/          # Sampling loop
```

## Requirements

- Go 1.22 or later
- Apple Silicon Mac (M1-M4)
- MLX library

## Installation

```bash
# Download dependencies
go mod tidy

# Build all binaries
make build
```

## Usage

### Training

```bash
# Train with default parameters
make train

# Train with custom parameters
./bin/train -steps 1000 -lr 0.01 -embed 16 -heads 4 -layers 1 -block 16
```

### Inference

```bash
# Generate samples
make infer

# Interactive mode
make infer-interactive

# Custom parameters
./bin/infer -samples 20 -temp 0.5 -seed 42
```

## Model Configuration

| Parameter   | Default | Description                    |
|-------------|---------|--------------------------------|
| embed       | 16      | Embedding dimension (n_embd)   |
| heads       | 4       | Number of attention heads      |
| layers      | 1       | Number of transformer layers   |
| block       | 16      | Context block size             |
| lr          | 0.01    | Learning rate                  |
| steps       | 1000    | Training steps                 |

## Training Parameters

| Parameter | Default | Description                    |
|-----------|---------|--------------------------------|
| beta1     | 0.85    | Adam first moment decay        |
| beta2     | 0.99    | Adam second moment decay       |
| temp      | 0.5     | Sampling temperature           |
| seed      | 42      | Random seed                    |

## Dataset

By default, the model trains on the names dataset:
- Source: https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt
- Size: ~32,000 names
- Format: one name per line

The dataset is automatically downloaded on first run.

## Model Architecture

The model follows GPT-2 architecture with:

1. **Embeddings**: Token + Position embeddings
2. **Normalization**: RMSNorm (no biases)
3. **Attention**: Multi-head self-attention with KV caching
4. **MLP**: Two-layer feed-forward with ReLU activation
5. **Output**: Linear projection to vocabulary

### Differences from standard GPT-2

- Uses RMSNorm instead of LayerNorm
- No bias terms in linear layers
- ReLU instead of GeLU in MLP
- Character-level tokenization

## Expected Results

### Training
- Initial loss: ~3.3
- Final loss (1000 steps): ~2.3-2.4

### Generated Samples
```
sample  1: kamon
sample  2: karai
sample  3: annel
sample  4: kaina
...
```

## Design Philosophy

This implementation follows UNIX philosophy:

- **Small packages**: Each package has a single responsibility
- **Clear interfaces**: Explicit contracts between components
- **Minimal magic**: No hidden behavior or complex abstractions
- **Debuggability**: Easy to trace and understand

## Performance Notes for M4

The M4 chip features:

- Enhanced Neural Engine for ML workloads
- Improved memory bandwidth
- Advanced Metal GPU architecture

MLX leverages these capabilities through:

- Unified memory architecture
- Metal-accelerated tensor operations
- Efficient automatic differentiation

## License

MIT

## References

- Original microgpt: https://github.com/karpathy/microgpt
- MLX: https://ml-explore.github.io/mlx/
- MakeMore: https://github.com/karpathy/makemore
