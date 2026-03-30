// Package inference implements the sampling loop for GPT generation.
package inference

import (
	"math/rand"
	"time"

	"github.com/microgpt/go/model"
	"github.com/microgpt/go/tensor"
	"github.com/microgpt/go/tokenizer"
)

// Config holds inference configuration.
type Config struct {
	Temperature float64 // sampling temperature (0, 1]
	MaxLen      int     // maximum generation length
	Seed        int64   // random seed for reproducibility
}

// DefaultConfig returns default inference configuration.
func DefaultConfig() Config {
	return Config{
		Temperature: 0.5,
		MaxLen:      16,
		Seed:        42,
	}
}

// Generator handles text generation from a trained model.
type Generator struct {
	cfg       Config
	model     *model.GPT
	tokenizer *tokenizer.Tokenizer
	rng       *rand.Rand
}

// NewGenerator creates a new generator.
func NewGenerator(cfg Config, m *model.GPT, tok *tokenizer.Tokenizer) *Generator {
	src := rand.NewSource(cfg.Seed)
	if cfg.Seed == 0 {
		src = rand.NewSource(time.Now().UnixNano())
	}

	return &Generator{
		cfg:       cfg,
		model:     m,
		tokenizer: tok,
		rng:       rand.New(src),
	}
}

// Generate produces a new sequence of tokens.
// Returns the generated text as a string.
func (g *Generator) Generate() string {
	cache := model.NewKVCache(g.model.Config().NumLayers)
	tokenID := g.tokenizer.BOS()
	sample := make([]int, 0)

	blockSize := g.model.Config().BlockSize
	maxLen := min(g.cfg.MaxLen, blockSize)

	for posID := 0; posID < maxLen; posID++ {
		// Forward pass
		logits := g.model.Forward(tokenID, posID, cache)

		// Apply temperature scaling
		if g.cfg.Temperature != 1.0 {
			logits = logits.Mul(tensor.FromList(1.0 / g.cfg.Temperature))
		}

		// Softmax to get probabilities
		probs := model.Softmax(logits)

		// Sample from distribution
		tokenID = g.sampleFromProbs(probs)

		// Check for EOS (BOS token acts as EOS too)
		if tokenID == g.tokenizer.BOS() {
			break
		}

		sample = append(sample, tokenID)
	}

	return g.tokenizer.Decode(sample)
}

// GenerateMultiple produces multiple samples.
func (g *Generator) GenerateMultiple(n int) []string {
	samples := make([]string, n)
	for i := 0; i < n; i++ {
		samples[i] = g.Generate()
	}
	return samples
}

// GenerateWithPrompt generates continuation from a prompt.
func (g *Generator) GenerateWithPrompt(prompt string) string {
	cache := model.NewKVCache(g.model.Config().NumLayers)

	// Process prompt tokens (without the final BOS)
	promptTokens := g.tokenizer.EncodeWithoutEndBOS(prompt)
	blockSize := g.model.Config().BlockSize

	// Run through prompt to build up KV cache
	var lastTokenID int
	for posID, tokenID := range promptTokens {
		_ = g.model.Forward(tokenID, posID, cache)
		lastTokenID = tokenID
	}

	// Continue generation from where prompt left off
	startPos := len(promptTokens)
	sample := make([]int, 0)
	maxLen := min(g.cfg.MaxLen, blockSize-startPos)

	for posID := 0; posID < maxLen; posID++ {
		logits := g.model.Forward(lastTokenID, startPos+posID, cache)

		if g.cfg.Temperature != 1.0 {
			logits = logits.Mul(tensor.FromList(1.0 / g.cfg.Temperature))
		}

		probs := model.Softmax(logits)
		lastTokenID = g.sampleFromProbs(probs)

		if lastTokenID == g.tokenizer.BOS() {
			break
		}

		sample = append(sample, lastTokenID)
	}

	return prompt + g.tokenizer.Decode(sample)
}

// sampleFromProbs samples a token ID from a probability distribution.
func (g *Generator) sampleFromProbs(probs *tensor.Tensor) int {
	// Convert probabilities to Go slice
	probList, err := probs.Data().ToList()
	if err != nil {
		// Fallback: return most likely token
		return 0
	}

	// Handle different list structures
	var probSlice []float64
	switch v := probList.(type) {
	case []interface{}:
		probSlice = make([]float64, len(v))
		for i, val := range v {
			if f, ok := val.(float64); ok {
				probSlice[i] = f
			}
		}
	case []float64:
		probSlice = v
	}

	if len(probSlice) == 0 {
		return 0
	}

	// Sample from categorical distribution
	r := g.rng.Float64()
	cumsum := 0.0
	for i, p := range probSlice {
		cumsum += p
		if r <= cumsum {
			return i
		}
	}

	// Fallback: return last token
	return len(probSlice) - 1
}

// SetTemperature updates the sampling temperature.
func (g *Generator) SetTemperature(temp float64) {
	g.cfg.Temperature = temp
}

// SetSeed updates the random seed.
func (g *Generator) SetSeed(seed int64) {
	g.rng = rand.New(rand.NewSource(seed))
}

// GreedyDecode generates using greedy decoding (temperature -> 0).
func (g *Generator) GreedyDecode() string {
	cache := model.NewKVCache(g.model.Config().NumLayers)
	tokenID := g.tokenizer.BOS()
	sample := make([]int, 0)

	blockSize := g.model.Config().BlockSize

	for posID := 0; posID < blockSize; posID++ {
		logits := g.model.Forward(tokenID, posID, cache)

		// Find argmax
		tokenID = g.argmax(logits)

		if tokenID == g.tokenizer.BOS() {
			break
		}

		sample = append(sample, tokenID)
	}

	return g.tokenizer.Decode(sample)
}

// argmax returns the index of the maximum value.
func (g *Generator) argmax(t *tensor.Tensor) int {
	list, err := t.Data().ToList()
	if err != nil {
		return 0
	}

	var values []float64
	switch v := list.(type) {
	case []interface{}:
		values = make([]float64, len(v))
		for i, val := range v {
			if f, ok := val.(float64); ok {
				values[i] = f
			}
		}
	case []float64:
		values = v
	}

	if len(values) == 0 {
		return 0
	}

	maxIdx := 0
	maxVal := values[0]
	for i, v := range values {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}

	return maxIdx
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
