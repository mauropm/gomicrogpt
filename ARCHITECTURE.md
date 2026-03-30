# MicroGPT Architecture

A minimal GPT-2-like language model implementation in Go with MLX acceleration for Apple Silicon.

## Overview

This project re-architectures the microgpt Python implementation into Go, leveraging Apple's MLX library for GPU-accelerated tensor operations on M1-M4 chips.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   cmd/train     │  │   cmd/infer     │  │   (future CLI)  │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼────────────────────┼────────────────────┼───────────┘
            │                    │                    │
┌───────────┴────────────────────┴────────────────────┴───────────┐
│                        Domain Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │ dataset/ │  │tokenizer/│  │  model/  │  │   inference/    │  │
│  └──────────┘  └──────────┘  └────┬─────┘  └────────┬────────┘  │
│  ┌──────────┐  ┌──────────┐      │        │         │          │
│  │  train/  │  │optimizer/│      │        │         │          │
│  └──────────┘  └──────────┘      │        │         │          │
└──────────────────────────────────┼────────┼─────────┼──────────┘
                                   │        │         │
┌──────────────────────────────────┴────────┴─────────┴──────────┐
│                      Core Abstraction Layer                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      tensor/                              │   │
│  │  - MLX-backed tensor operations                           │   │
│  │  - Automatic differentiation (backpropagation)            │   │
│  │  - GPU acceleration via Metal                             │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────┐
│                    Hardware Abstraction Layer                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                        mlx/                               │   │
│  │  - CGO bindings to MLX C API                              │   │
│  │  - C++ wrapper for MLX C++ library                        │   │
│  │  - Device management (CPU/GPU)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────┐
│                      Hardware Layer                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Apple Silicon (M1/M2/M3/M4)                     │   │
│  │  - Metal GPU                                              │   │
│  │  - Unified Memory Architecture                            │   │
│  │  - Neural Engine                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Package Responsibilities

### `mlx/` - MLX Bindings

**Purpose**: CGO bindings for Apple's MLX library.

**Files**:
- `mlx.go` - Go wrapper for MLX C API
- `mlx_wrapper.h` - C header for MLX C++ library
- `mlx_wrapper.cpp` - C++ implementation wrapping MLX

**Key Types**:
```go
type Array struct {
    ptr *C.mlx_array_t
}

type Device int  // CPU, GPU
type Dtype int   // Float32, Float16, etc.
```

**Key Functions**:
```go
func Zeros(shape []int32, dtype Dtype) *Array
func Ones(shape []int32, dtype Dtype) *Array
func RandomNormal(shape []int32, dtype Dtype) *Array
func MatMul(a, b *Array) *Array
func Add(a, b *Array) *Array
// ... and more
```

### `tensor/` - Tensor Operations

**Purpose**: High-level tensor API with automatic differentiation.

**Key Types**:
```go
type Tensor struct {
    data     *mlx.Array
    grad     *mlx.Array
    children []*Tensor
    op       opType
    requiresGrad bool
}
```

**Key Operations**:
- Element-wise: `Add`, `Sub`, `Mul`, `Div`, `Pow`
- Unary: `Neg`, `Sqrt`, `Log`, `Exp`, `Relu`
- Reduction: `Sum`, `Mean`, `Max`
- Linear Algebra: `MatMul`
- Manipulation: `Reshape`, `Transpose`, `Slice`, `Concat`, `Stack`
- Normalization: `RMSNorm`, `Softmax`

**Automatic Differentiation**:
```go
// Compute value and gradients
loss, grads := tensor.ValueAndGrad(
    func() *tensor.Tensor {
        return model.Forward(x, y)
    },
    params,
)
```

### `dataset/` - Data Loading

**Purpose**: Load and shuffle training documents.

**Key Types**:
```go
type Dataset struct {
    docs []string
}

type Iterator struct {
    dataset *Dataset
    index   int
}
```

**Features**:
- Load from file or URL
- Automatic download of default dataset
- Random shuffling
- Iterator with wraparound

### `tokenizer/` - Tokenization

**Purpose**: Character-level tokenization.

**Key Types**:
```go
type Tokenizer struct {
    chars     []rne
    charToID  map[rune]int
    idToChar  map[int]rune
    bosID     int
    vocabSize int
}
```

**Features**:
- Character-level encoding
- BOS (Beginning of Sequence) token
- Reversible encoding/decoding

### `model/` - Neural Network

**Purpose**: GPT-2-like transformer architecture.

**Key Types**:
```go
type Config struct {
    VocabSize int
    EmbedDim  int
    NumHeads  int
    NumLayers int
    BlockSize int
    HeadDim   int
}

type GPT struct {
    cfg      Config
    wte      *tensor.Tensor  // Token embeddings
    wpe      *tensor.Tensor  // Position embeddings
    lmHead   *tensor.Tensor  // Output projection
    attnWQ   []*tensor.Tensor
    attnWK   []*tensor.Tensor
    attnWV   []*tensor.Tensor
    attnWO   []*tensor.Tensor
    mlpFC1   []*tensor.Tensor
    mlpFC2   []*tensor.Tensor
}

type KVCache struct {
    Keys   [][]*tensor.Tensor
    Values [][]*tensor.Tensor
}
```

**Architecture**:
```
Input Token ──┬── Token Embedding ──┐
              │                     │
Input Pos  ───┴── Position Embedding│
                                    │
                                    ▼
                              ┌─────────────┐
                              │   RMSNorm   │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │  Attention  │◄─── KV Cache
                              │  (Multi-Head)│
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │  Residual + │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │   RMSNorm   │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │    MLP      │
                              │ (FC-ReLU-FC)│
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │  Residual + │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │   RMSNorm   │
                              └──────┬──────┘
                                     │
                                     ▼
                              Output Logits
```

### `optimizer/` - Optimization

**Purpose**: Adam optimizer with learning rate scheduling.

**Key Types**:
```go
type AdamConfig struct {
    LearningRate float64
    Beta1        float64
    Beta2        float64
    Eps          float64
}

type Adam struct {
    cfg       AdamConfig
    m         []*tensor.Tensor  // First moment
    v         []*tensor.Tensor  // Second moment
    step      int
}
```

**Update Rule**:
```
m = β₁·m + (1-β₁)·∇L
v = β₂·v + (1-β₂)·∇L²
m̂ = m / (1-β₁ᵗ)
v̂ = v / (1-β₂ᵗ)
θ = θ - α·m̂ / (√v̂ + ε)
```

### `train/` - Training Loop

**Purpose**: Orchestrate training process.

**Key Types**:
```go
type Config struct {
    NumSteps     int
    LearningRate float64
    Beta1        float64
    Beta2        float64
    Temperature  float64
    Seed         int64
}

type Trainer struct {
    cfg       Config
    model     *model.GPT
    tokenizer *tokenizer.Tokenizer
    dataset   *dataset.Dataset
    optimizer *optimizer.Adam
}
```

**Training Loop**:
```
for step in 0..numSteps:
    1. Get document from dataset
    2. Tokenize with BOS boundaries
    3. Forward pass (compute loss)
    4. Backward pass (compute gradients)
    5. Optimizer step (update parameters)
    6. Zero gradients
    7. Log progress
```

### `inference/` - Text Generation

**Purpose**: Generate text from trained model.

**Key Types**:
```go
type Config struct {
    Temperature float64
    MaxLen      int
    Seed        int64
}

type Generator struct {
    cfg       Config
    model     *model.GPT
    tokenizer *tokenizer.Tokenizer
    rng       *rand.Rand
}
```

**Generation Process**:
```
1. Start with BOS token
2. Forward pass to get logits
3. Apply temperature scaling
4. Softmax to get probabilities
5. Sample next token
6. Repeat until EOS or max length
```

## Data Flow

### Training
```
Dataset ──► Tokenizer ──► Tokens ──► Model.Forward() ──► Logits
                                                      │
                                                      ▼
Loss ◄── CrossEntropy ◄── Softmax ◄──────────────────┘
 │
 ▼
Backprop ──► Gradients ──► Adam.Update() ──► Parameters
```

### Inference
```
BOS Token ──► Model.Forward() ──► Logits ──► Temperature
                                              │
                                              ▼
Sample ◄── Categorical ◄── Softmax ◄──────────┘
 │
 ▼
Next Token ──► (repeat until EOS)
```

## Memory Management

### MLX Array Lifecycle
```go
// Create
arr := mlx.Zeros(shape, dtype)

// Use
result := mlx.MatMul(a, b)

// Explicit cleanup (optional, GC handles it)
arr.Free()

// Go finalizer ensures cleanup
runtime.SetFinalizer(arr, (*Array).Free)
```

### Tensor Gradient Tracking
```go
// Enable gradient tracking
param.EnableGrad()

// Forward pass creates computation graph
loss := model.Forward(x)

// Backward pass populates gradients
_, grads := tensor.ValueAndGrad(fn, params)

// Zero gradients after optimizer step
param.ZeroGrad()
```

## Performance Considerations

### M4-Specific Optimizations

1. **Unified Memory**: MLX leverages unified memory architecture
   - No CPU-GPU data transfer overhead
   - Large models fit in shared memory

2. **Metal GPU**: All tensor ops run on GPU
   - Matrix multiplication: Highly optimized
   - Element-wise ops: Parallel execution

3. **Neural Engine**: Future integration point
   - Quantized inference
   - Specialized operations

### Batching

Current implementation processes single sequences. Future improvements:
```go
// Batched forward pass
func (m *GPT) ForwardBatch(tokens []int, positions []int) *tensor.Tensor
```

### Lazy Evaluation

MLX uses lazy evaluation:
```go
// Operations build computation graph
a := tensor.Zeros([1000, 1000])
b := tensor.Zeros([1000, 1000])
c := a.MatMul(b)  // Not computed yet

// Force evaluation
tensor.Eval(c)  // Now computed on GPU
```

## Error Handling

### CGO Error Handling
```go
// Check MLX status
status := C.mlx_get_last_error()
if status != C.MLX_SUCCESS {
    return fmt.Errorf("MLX error: %s", C.mlx_get_error_string(status))
}
```

### Go Error Propagation
```go
func operation() (*Tensor, error) {
    if invalidInput {
        return nil, fmt.Errorf("invalid input shape")
    }
    return result, nil
}
```

## Testing Strategy

### Unit Tests
```go
func TestTensorAdd(t *testing.T) {
    a := tensor.FromList([]float64{1, 2, 3})
    b := tensor.FromList([]float64{4, 5, 6})
    c := a.Add(b)
    
    expected := []float64{5, 7, 9}
    // Assert equality
}
```

### Integration Tests
```go
func TestTrainingStep(t *testing.T) {
    model := model.New(config)
    trainer := train.NewTrainer(cfg, model, tok, ds)
    
    loss := trainer.TrainStep(doc, 0)
    assert.Less(t, loss, initialLoss)
}
```

## Build System

### Makefile Targets
```makefile
build          # Build all binaries
build-train    # Build training binary
build-infer    # Build inference binary
train          # Run training
infer          # Run inference
test           # Run tests
clean          # Remove build artifacts
```

### CGO Configuration
```bash
# Required environment
export CGO_ENABLED=1
export CGO_CFLAGS="-I/opt/homebrew/include"
export CGO_LDFLAGS="-L/opt/homebrew/lib -lmlx"
```

## Future Enhancements

1. **Batched Training**: Process multiple sequences in parallel
2. **Gradient Checkpointing**: Reduce memory usage for long sequences
3. **Mixed Precision**: FP16/BF16 for faster computation
4. **Model Checkpointing**: Save/load trained weights
5. **Distributed Training**: Multi-GPU support
6. **Tokenization**: BPE/WordPiece support
7. **Evaluation**: Perplexity metrics, validation loops

## Dependencies

| Package | Purpose |
|---------|---------|
| MLX | GPU-accelerated tensor operations |
| CGO | C/C++ interop |
| runtime | Memory management, finalizers |
| sync | Thread-safe tensor operations |
| unsafe | Low-level memory access |

## License

MIT License - See LICENSE file for details.
