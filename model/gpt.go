// Package model implements a minimal GPT-2-like transformer architecture.
package model

import (
	"math"

	"github.com/microgpt/go/tensor"
)

// Config holds the model hyperparameters.
type Config struct {
	VocabSize  int // vocabulary size
	EmbedDim   int // embedding dimension (n_embd)
	NumHeads   int // number of attention heads (n_head)
	NumLayers  int // number of transformer layers (n_layer)
	BlockSize  int // maximum context length (block_size)
	HeadDim    int // dimension per attention head (derived: EmbedDim / NumHeads)
}

// NewConfig creates a config with default GPT-like hyperparameters.
func NewConfig(vocabSize int) Config {
	cfg := Config{
		VocabSize: vocabSize,
		EmbedDim:  16,
		NumHeads:  4,
		NumLayers: 1,
		BlockSize: 16,
	}
	cfg.HeadDim = cfg.EmbedDim / cfg.NumHeads
	return cfg
}

// KVCache stores key/value tensors for each layer during autoregressive generation.
type KVCache struct {
	Keys   [][]*tensor.Tensor // [layer][position][head_dim]
	Values [][]*tensor.Tensor // [layer][position][head_dim]
}

// NewKVCache creates a new KV cache for the given number of layers.
func NewKVCache(numLayers int) *KVCache {
	return &KVCache{
		Keys:   make([][]*tensor.Tensor, numLayers),
		Values: make([][]*tensor.Tensor, numLayers),
	}
}

// Reset clears the cache.
func (c *KVCache) Reset() {
	for i := range c.Keys {
		c.Keys[i] = nil
		c.Values[i] = nil
	}
}

// Append adds new key/value tensors to the cache.
func (c *KVCache) Append(layer int, k, v *tensor.Tensor) {
	c.Keys[layer] = append(c.Keys[layer], k)
	c.Values[layer] = append(c.Values[layer], v)
}

// GPT is the main transformer model.
type GPT struct {
	cfg        Config
	wte        *tensor.Tensor // token embeddings [vocab_size, embed_dim]
	wpe        *tensor.Tensor // position embeddings [block_size, embed_dim]
	lmHead     *tensor.Tensor // output projection [embed_dim, vocab_size]
	attnWQ     []*tensor.Tensor // [num_layers][embed_dim, embed_dim]
	attnWK     []*tensor.Tensor // [num_layers][embed_dim, embed_dim]
	attnWV     []*tensor.Tensor // [num_layers][embed_dim, embed_dim]
	attnWO     []*tensor.Tensor // [num_layers][embed_dim, embed_dim]
	mlpFC1     []*tensor.Tensor // [num_layers][4*embed_dim, embed_dim]
	mlpFC2     []*tensor.Tensor // [num_layers][embed_dim, 4*embed_dim]
}

// New creates a new GPT model with Gaussian-initialized parameters.
func New(cfg Config) *GPT {
	std := 0.08

	// Token and position embeddings
	wte := tensor.Gaussian([]int{cfg.VocabSize, cfg.EmbedDim}, 0, std)
	wpe := tensor.Gaussian([]int{cfg.BlockSize, cfg.EmbedDim}, 0, std)

	// Output projection (stored as [vocab_size, embed_dim], transposed for matmul)
	lmHead := tensor.Gaussian([]int{cfg.VocabSize, cfg.EmbedDim}, 0, std)

	// Initialize per-layer parameters
	attnWQ := make([]*tensor.Tensor, cfg.NumLayers)
	attnWK := make([]*tensor.Tensor, cfg.NumLayers)
	attnWV := make([]*tensor.Tensor, cfg.NumLayers)
	attnWO := make([]*tensor.Tensor, cfg.NumLayers)
	mlpFC1 := make([]*tensor.Tensor, cfg.NumLayers)
	mlpFC2 := make([]*tensor.Tensor, cfg.NumLayers)

	for i := 0; i < cfg.NumLayers; i++ {
		attnWQ[i] = tensor.Gaussian([]int{cfg.EmbedDim, cfg.EmbedDim}, 0, std)
		attnWK[i] = tensor.Gaussian([]int{cfg.EmbedDim, cfg.EmbedDim}, 0, std)
		attnWV[i] = tensor.Gaussian([]int{cfg.EmbedDim, cfg.EmbedDim}, 0, std)
		attnWO[i] = tensor.Gaussian([]int{cfg.EmbedDim, cfg.EmbedDim}, 0, std)
		mlpFC1[i] = tensor.Gaussian([]int{4 * cfg.EmbedDim, cfg.EmbedDim}, 0, std)
		mlpFC2[i] = tensor.Gaussian([]int{cfg.EmbedDim, 4 * cfg.EmbedDim}, 0, std)
	}

	return &GPT{
		cfg:      cfg,
		wte:      wte,
		wpe:      wpe,
		lmHead:   lmHead,
		attnWQ:   attnWQ,
		attnWK:   attnWK,
		attnWV:   attnWV,
		attnWO:   attnWO,
		mlpFC1:   mlpFC1,
		mlpFC2:   mlpFC2,
	}
}

// Config returns the model configuration.
func (m *GPT) Config() Config {
	return m.cfg
}

// Forward performs a forward pass through the model.
// tokenID: input token ID
// posID: position ID
// cache: KV cache for autoregressive generation
// Returns: logits tensor of shape [vocab_size]
func (m *GPT) Forward(tokenID, posID int, cache *KVCache) *tensor.Tensor {
	// Token and position embeddings
	tokEmb := m.wte.Slice([]int{tokenID, 0}, []int{tokenID + 1, m.cfg.EmbedDim})
	tokEmb = tokEmb.Reshape(m.cfg.EmbedDim)

	posEmb := m.wpe.Slice([]int{posID, 0}, []int{posID + 1, m.cfg.EmbedDim})
	posEmb = posEmb.Reshape(m.cfg.EmbedDim)

	// Combined embedding
	x := tokEmb.Add(posEmb)

	// Initial normalization
	x = rmsNorm(x)

	// Transformer layers
	for li := 0; li < m.cfg.NumLayers; li++ {
		x = m.attentionBlock(x, li, cache)
		x = m.mlpBlock(x, li)
	}

	// Output projection
	logits := tensor.Linear(x, m.lmHead)

	return logits
}

// attentionBlock applies multi-head self-attention.
func (m *GPT) attentionBlock(x *tensor.Tensor, layerIdx int, cache *KVCache) *tensor.Tensor {
	xResidual := x
	x = rmsNorm(x)

	// Compute Q, K, V
	q := tensor.Linear(x, m.attnWQ[layerIdx])
	k := tensor.Linear(x, m.attnWK[layerIdx])
	v := tensor.Linear(x, m.attnWV[layerIdx])

	// Store in cache
	cache.Append(layerIdx, k, v)

	// Multi-head attention
	xAttn := m.multiHeadAttention(q, cache.Keys[layerIdx], cache.Values[layerIdx], layerIdx)

	// Output projection
	x = tensor.Linear(xAttn, m.attnWO[layerIdx])

	// Residual connection
	return x.Add(xResidual)
}

// multiHeadAttention computes multi-head self-attention.
func (m *GPT) multiHeadAttention(q *tensor.Tensor, keys, values []*tensor.Tensor, layerIdx int) *tensor.Tensor {
	headDim := m.cfg.HeadDim
	numHeads := m.cfg.NumHeads

	// Process each head
	headOutputs := make([]*tensor.Tensor, numHeads)

	for h := 0; h < numHeads; h++ {
		hs := h * headDim
		he := hs + headDim

		// Slice query for this head
		qHead := q.Slice([]int{hs}, []int{he})

		// Slice keys and values for this head
		kHeads := make([]*tensor.Tensor, len(keys))
		vHeads := make([]*tensor.Tensor, len(values))
		for t := range keys {
			kHeads[t] = keys[t].Slice([]int{hs}, []int{he})
			vHeads[t] = values[t].Slice([]int{hs}, []int{he})
		}

		// Compute attention logits: (q · k) / sqrt(head_dim)
		attnLogits := make([]*tensor.Tensor, len(kHeads))
		for t := range kHeads {
			// Dot product: sum(q_head * k_head)
			product := qHead.Mul(kHeads[t])
			dotProduct := product.Sum()
			// Scale by 1/sqrt(head_dim)
			scale := 1.0 / math.Sqrt(float64(headDim))
			attnLogits[t] = dotProduct.Mul(tensor.FromList(scale))
		}

		// Stack logits and apply softmax
		logitsStack := tensor.StackTensors(attnLogits, 0)
		weights := tensor.Softmax(logitsStack)

		// Weighted sum of values
		headOut := tensor.ZerosTensor([]int{headDim})
		for t := range vHeads {
			weightSlice := weights.Slice([]int{t}, []int{t + 1}).Reshape(1)
			scaledV := vHeads[t].Mul(weightSlice)
			headOut = headOut.Add(scaledV)
		}

		headOutputs[h] = headOut
	}

	// Concatenate all heads
	return tensor.ConcatTensors(headOutputs, 0)
}

// mlpBlock applies the MLP (feed-forward) block.
func (m *GPT) mlpBlock(x *tensor.Tensor, layerIdx int) *tensor.Tensor {
	xResidual := x
	x = rmsNorm(x)

	// FC1 + ReLU
	x = tensor.Linear(x, m.mlpFC1[layerIdx])
	x = x.Relu()

	// FC2
	x = tensor.Linear(x, m.mlpFC2[layerIdx])

	// Residual connection
	return x.Add(xResidual)
}

// rmsNorm applies RMS normalization to a tensor.
func rmsNorm(x *tensor.Tensor) *tensor.Tensor {
	// x / sqrt(mean(x^2) + eps)
	// Compute mean of squares
	x2 := x.Mul(x)
	sum := x2.Sum()
	ms := sum.Mul(tensor.FromList(1.0 / float64(x2.Shape()[0])))
	scale := ms.Add(tensor.FromList(1e-5)).Rsqrt()
	// Broadcast scale to match x shape and multiply
	return x.Mul(scale)
}

// Params returns all model parameters as a flat list for optimization.
func (m *GPT) Params() []*tensor.Tensor {
	var params []*tensor.Tensor

	// Add embeddings
	params = append(params, m.wte, m.wpe, m.lmHead)

	// Add layer parameters
	for i := 0; i < m.cfg.NumLayers; i++ {
		params = append(params,
			m.attnWQ[i], m.attnWK[i], m.attnWV[i], m.attnWO[i],
			m.mlpFC1[i], m.mlpFC2[i],
		)
	}

	return params
}

// NumParams returns the total number of parameters.
func (m *GPT) NumParams() int {
	count := 0
	for _, p := range m.Params() {
		shape := p.Shape()
		prod := 1
		for _, s := range shape {
			prod *= s
		}
		count += prod
	}
	return count
}

// GetParamByIndex retrieves a parameter by its flattened index.
// This is useful for the optimizer which works with flattened parameters.
func (m *GPT) GetParamByIndex(idx int) *tensor.Tensor {
	params := m.Params()
	if idx < 0 || idx >= len(params) {
		panic("parameter index out of range")
	}
	return params[idx]
}

// ZeroGrad resets gradients for all parameters.
func (m *GPT) ZeroGrad() {
	for _, p := range m.Params() {
		p.ZeroGrad()
	}
}

// Softmax applies softmax to a logits tensor.
func Softmax(logits *tensor.Tensor) *tensor.Tensor {
	return tensor.Softmax(logits)
}

// CrossEntropyLoss computes -log(prob[target]).
func CrossEntropyLoss(logits *tensor.Tensor, target int) *tensor.Tensor {
	probs := Softmax(logits)
	// Get probability of target class
	targetProb := probs.Slice([]int{target}, []int{target + 1})
	// Negative log likelihood
	return targetProb.Log().Neg()
}
