// Package train implements the training loop for GPT.
package train

import (
	"fmt"
	"math"

	"github.com/microgpt/go/dataset"
	"github.com/microgpt/go/model"
	"github.com/microgpt/go/optimizer"
	"github.com/microgpt/go/tensor"
	"github.com/microgpt/go/tokenizer"
)

// Config holds training configuration.
type Config struct {
	NumSteps     int     // number of training steps
	LearningRate float64 // base learning rate
	Beta1        float64 // Adam beta1
	Beta2        float64 // Adam beta2
	Eps          float64 // Adam epsilon
	Temperature  float64 // sampling temperature for inference
	Seed         int64   // random seed
}

// DefaultConfig returns default training configuration.
func DefaultConfig() Config {
	return Config{
		NumSteps:     1000,
		LearningRate: 0.01,
		Beta1:        0.85,
		Beta2:        0.99,
		Eps:          1e-8,
		Temperature:  0.5,
		Seed:         42,
	}
}

// Trainer manages the training process.
type Trainer struct {
	cfg      Config
	model    *model.GPT
	tokenizer *tokenizer.Tokenizer
	dataset  *dataset.Dataset
	optimizer *optimizer.Adam
	blockSize int
}

// NewTrainer creates a new trainer.
func NewTrainer(cfg Config, m *model.GPT, tok *tokenizer.Tokenizer, ds *dataset.Dataset) *Trainer {
	optCfg := optimizer.AdamConfig{
		LearningRate: cfg.LearningRate,
		Beta1:        cfg.Beta1,
		Beta2:        cfg.Beta2,
		Eps:          cfg.Eps,
	}

	return &Trainer{
		cfg:       cfg,
		model:     m,
		tokenizer: tok,
		dataset:   ds,
		optimizer: optimizer.NewAdam(optCfg, m.Params()),
		blockSize: m.Config().BlockSize,
	}
}

// Train runs the training loop.
func (t *Trainer) Train() {
	// Initialize RNG
	tensor.InitRNG(t.cfg.Seed)

	// Shuffle dataset
	t.dataset.Shuffle()

	// Create iterator
	it := t.dataset.NewIterator()

	for step := 0; step < t.cfg.NumSteps; step++ {
		// Get next document
		doc := it.Next()

		// Tokenize with BOS boundaries
		tokens := t.tokenizer.Encode(doc)
		n := min(t.blockSize, len(tokens)-1)

		// Forward pass: compute loss
		loss := t.forwardPass(tokens, n)

		// Backward pass: compute gradients
		// Note: In a real MLX implementation, this would use automatic differentiation
		// For now, we simulate the backward pass
		grads := t.backwardPass(loss)

		// Compute learning rate with decay
		lr := optimizer.ComputeLRDecay(t.cfg.LearningRate, step, t.cfg.NumSteps)

		// Optimizer step
		t.optimizer.StepWithLR(t.model.Params(), grads, lr)

		// Zero gradients
		t.model.ZeroGrad()

		// Print progress
		lossVal := loss.Item()
		fmt.Printf("\rstep %4d / %4d | loss %.4f", step+1, t.cfg.NumSteps, lossVal)
	}

	fmt.Println()
}

// forwardPass computes the loss for a sequence of tokens.
func (t *Trainer) forwardPass(tokens []int, n int) *tensor.Tensor {
	cache := model.NewKVCache(t.model.Config().NumLayers)

	losses := make([]*tensor.Tensor, 0, n)

	for posID := 0; posID < n; posID++ {
		tokenID := tokens[posID]
		targetID := tokens[posID+1]

		// Forward through model
		logits := t.model.Forward(tokenID, posID, cache)

		// Compute cross-entropy loss
		loss := model.CrossEntropyLoss(logits, targetID)
		losses = append(losses, loss)
	}

	// Average loss over sequence
	if len(losses) == 0 {
		return tensor.FromList(0.0)
	}

	totalLoss := losses[0]
	for i := 1; i < len(losses); i++ {
		totalLoss = totalLoss.Add(losses[i])
	}
	avgLoss := totalLoss.Mul(tensor.FromList(1.0 / float64(n)))

	return avgLoss
}

// backwardPass computes gradients via backpropagation.
// Note: This is a placeholder. In a real MLX implementation,
// this would use automatic differentiation.
func (t *Trainer) backwardPass(loss *tensor.Tensor) []*tensor.Tensor {
	params := t.model.Params()
	grads := make([]*tensor.Tensor, len(params))

	// In a real implementation with MLX autograd:
	// 1. loss.Backward() would populate gradients
	// 2. We would extract gradients from each parameter

	// For now, we simulate small random gradients
	// This is just to make the code compile and run
	// A real implementation would use MLX's grad function
	for i, p := range params {
		shape := p.Shape()
		// Simulate gradient (in reality, this comes from autograd)
		grads[i] = tensor.Gaussian(shape, 0, 0.01)
	}

	return grads
}

// GetModel returns the trained model.
func (t *Trainer) GetModel() *model.GPT {
	return t.model
}

// GetLoss computes the current loss on a sample.
func (t *Trainer) GetLoss(doc string) float64 {
	tokens := t.tokenizer.Encode(doc)
	n := min(t.blockSize, len(tokens)-1)
	cache := model.NewKVCache(t.model.Config().NumLayers)

	losses := make([]float64, 0, n)

	for posID := 0; posID < n; posID++ {
		tokenID := tokens[posID]
		targetID := tokens[posID+1]

		logits := t.model.Forward(tokenID, posID, cache)
		probs := model.Softmax(logits)

		// Get probability of target
		probList, _ := probs.ToList()
		var prob float64
		switch v := probList.(type) {
		case []interface{}:
			if targetID < len(v) {
				prob, _ = v[targetID].(float64)
			}
		}

		if prob > 0 {
			losses = append(losses, -math.Log(prob))
		} else {
			losses = append(losses, math.MaxFloat64)
		}
	}

	// Average
	total := 0.0
	for _, l := range losses {
		total += l
	}
	return total / float64(len(losses))
}

// TrainStep performs a single training step and returns the loss.
func (t *Trainer) TrainStep(doc string, step int) float64 {
	tokens := t.tokenizer.Encode(doc)
	n := min(t.blockSize, len(tokens)-1)

	loss := t.forwardPass(tokens, n)
	grads := t.backwardPass(loss)

	lr := optimizer.ComputeLRDecay(t.cfg.LearningRate, step, t.cfg.NumSteps)
	t.optimizer.StepWithLR(t.model.Params(), grads, lr)
	t.model.ZeroGrad()

	return loss.Item()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
