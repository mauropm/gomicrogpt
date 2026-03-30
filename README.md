# MicroGPT in Go

A minimal GPT-2-like language model implementation in Go with **MLX acceleration** for Apple Silicon (M1-M4).

## Overview

This is a Go re-architecture of the microgpt Python implementation, featuring:

- **GPU Acceleration**: MLX-backed tensor operations on Apple Silicon
- **Automatic Differentiation**: Built-in backpropagation for training
- **Pure Go**: Clean, idiomatic Go codebase
- **Hardware Optimized**: Leverages M1-M4 Neural Engine and Metal GPU

## Requirements

### Hardware
- **macOS** on Apple Silicon (M1, M2, M3, or M4)
- macOS 13.0 or later recommended

### Software
```bash
# Install MLX via Homebrew
brew install mlx

# Install Go 1.22 or later
brew install go@1.22

# Verify installation
mlx --version
go version
```

### Build Dependencies
- Xcode Command Line Tools: `xcode-select --install`
- CGO enabled (default on macOS)

## Installation

```bash
# Clone and enter repository
cd microgpt

# Download Go dependencies
go mod tidy

# Build all binaries
make build
```

## Quick Start

### Training

```bash
# Train with default parameters (1000 steps)
make train

# Or run directly with custom parameters
./bin/train -steps 1000 -lr 0.01 -embed 16 -heads 4 -layers 1 -block 16
```

### Inference

```bash
# Generate 20 samples
make infer

# Interactive mode
make infer-interactive

# Custom parameters
./bin/infer -samples 20 -temp 0.5 -seed 42
```

## Configuration

### Model Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `-embed` | 16 | Embedding dimension (n_embd) |
| `-heads` | 4 | Number of attention heads (n_head) |
| `-layers` | 1 | Number of transformer layers (n_layer) |
| `-block` | 16 | Context block size |

### Training Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `-steps` | 1000 | Training iterations |
| `-lr` | 0.01 | Learning rate |
| `-beta1` | 0.85 | Adam first moment decay |
| `-beta2` | 0.99 | Adam second moment decay |
| `-temp` | 0.5 | Sampling temperature |
| `-seed` | 42 | Random seed |
| `-verbose` | false | Enable verbose output with backend info |

### Full Training Example

```bash
./bin/train \
    -steps 2000 \
    -lr 0.005 \
    -beta1 0.9 \
    -beta2 0.999 \
    -embed 32 \
    -heads 4 \
    -layers 2 \
    -block 32 \
    -temp 0.7 \
    -seed 123
```

### Verbose Mode

Use `-verbose` to see detailed backend information:

```bash
./bin/train -steps 100 -verbose
```

Example output:
```
=== MicroGPT Training ===

Backend: Pure Go (CPU - arm64)
MLX enabled: false

Loading dataset...
...
```

When MLX is installed and built with CGO:
```
Backend: MLX (Metal GPU) - Apple Silicon
MLX enabled: true
```

## Dataset

By default, trains on the names dataset:
- **Source**: https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt
- **Size**: ~32,000 names
- **Format**: One name per line

The dataset is automatically downloaded on first run and cached as `input.txt`.

### Custom Dataset

```bash
# Use your own dataset
./bin/train -data /path/to/your/dataset.txt
```

### Inference Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `-samples` | 20 | Number of samples to generate |
| `-temp` | 0.5 | Sampling temperature (0.1-2.0) |
| `-seed` | 42 | Random seed |
| `-interactive` | false | Run in interactive mode |
| `-verbose` | false | Enable verbose output |

## Expected Results

### Training Progress
```
=== MicroGPT Training ===

Loading dataset...
Loaded 32033 documents
Vocabulary size: 27 (26 chars + 1 BOS)
Model created: 4192 parameters

Training...
step    1 / 1000 | loss 3.2847
step    2 / 1000 | loss 3.1923
...
step 1000 / 1000 | loss 2.3156
```

### Generated Samples
```
--- Inference (generated samples) ---
sample  1: kamon
sample  2: karai
sample  3: annel
sample  4: kaina
sample  5: marie
...
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

### Package Structure

```
microgpt/
├── cmd/
│   ├── train/          # Training CLI
│   └── infer/          # Inference CLI
├── mlx/                # MLX CGO bindings
├── tensor/             # MLX-backed tensor operations
├── model/              # GPT architecture
├── optimizer/          # Adam optimizer
├── train/              # Training loop
├── inference/          # Text generation
├── tokenizer/          # Character tokenizer
└── dataset/            # Data loading
```

### Model Architecture

```
Input ──► Embeddings ──► [Transformer Layer]×N ──► Output
                          │
                          ├── Multi-Head Attention
                          │   ├── Query/Key/Value
                          │   └── KV Cache (inference)
                          │
                          └── MLP (FC-ReLU-FC)
```

## Performance

### M4 Performance Characteristics

| Metric | Value |
|--------|-------|
| Memory Bandwidth | 120 GB/s (M4) |
| GPU Cores | Up to 10 |
| Neural Engine | 16 cores |

### Optimization Tips

1. **Use GPU**: MLX automatically uses Metal GPU
2. **Batch Size**: Currently single-sequence; batching planned
3. **Precision**: FP32 default; FP16 support planned
4. **Memory**: Unified memory allows larger models

## Development

### Building

```bash
# Full build
make build

# Build specific binary
make build-train
make build-infer

# Clean build
make clean && make build
```

### Testing

```bash
# Run tests
make test

# With coverage
make coverage
```

### Formatting

```bash
# Format code
make fmt

# Lint
make lint
```

## Troubleshooting

### CGO Errors

```bash
# Ensure CGO is enabled
export CGO_ENABLED=1

# Check MLX installation
brew list mlx

# Reinstall MLX if needed
brew reinstall mlx
```

### Build Errors

```bash
# Clean and rebuild
make clean
go clean -cache
make build

# Check Go version
go version  # Should be 1.22+
```

### Runtime Errors

```bash
# Check Metal support
system_profiler SPDisplaysDataType | grep Metal

# Verify MLX device
# (MLX automatically selects GPU on Apple Silicon)
```

## API Reference

### Tensor Operations

```go
import "github.com/microgpt/go/tensor"

// Create tensors
a := tensor.Gaussian([2, 3], 0, 0.02)
b := tensor.Zeros([3, 4])
c := tensor.Ones([4])

// Element-wise operations
d := a.Add(b)
e := a.Mul(b)
f := a.Relu()

// Matrix operations
g := a.MatMul(b)

// Reduction
sum := a.Sum()
mean := a.Mean()

// Automatic differentiation
loss, grads := tensor.ValueAndGrad(fn, params)
```

### Model Usage

```go
import "github.com/microgpt/go/model"

// Create model
cfg := model.Config{
    VocabSize: 27,
    EmbedDim:  16,
    NumHeads:  4,
    NumLayers: 1,
    BlockSize: 16,
}
m := model.New(cfg)

// Forward pass
cache := model.NewKVCache(cfg.NumLayers)
logits := m.Forward(tokenID, posID, cache)

// Get parameters
params := m.Params()
```

## Comparison with Python Version

| Feature | Python | Go + MLX |
|---------|--------|----------|
| Autograd | Scalar Value | Tensor-level |
| Acceleration | CPU | GPU (Metal) |
| Memory | Python GC | Go GC + MLX |
| Performance | ~1 min/train | Faster with GPU |
| Dependencies | None | MLX library |

## Future Enhancements

- [ ] Batched training
- [ ] Model checkpointing (save/load weights)
- [ ] Mixed precision (FP16/BF16)
- [ ] BPE tokenization
- [ ] Multi-GPU support
- [ ] WandB/MLflow integration
- [ ] Export to ONNX

## References

- [Original microgpt](https://github.com/karpathy/microgpt)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MakeMore](https://github.com/karpathy/makemore)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please read ARCHITECTURE.md before submitting PRs.
