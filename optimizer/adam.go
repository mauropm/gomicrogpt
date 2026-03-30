// Package optimizer implements the Adam optimization algorithm.
package optimizer

import (
	"math"

	"github.com/microgpt/go/tensor"
)

// AdamConfig holds Adam optimizer hyperparameters.
type AdamConfig struct {
	LearningRate float64 // base learning rate
	Beta1        float64 // first moment decay rate
	Beta2        float64 // second moment decay rate
	Eps          float64 // epsilon for numerical stability
}

// DefaultAdamConfig returns default Adam hyperparameters.
func DefaultAdamConfig() AdamConfig {
	return AdamConfig{
		LearningRate: 0.01,
		Beta1:        0.85,
		Beta2:        0.99,
		Eps:          1e-8,
	}
}

// Adam implements the Adam optimizer with per-parameter moment buffers.
type Adam struct {
	cfg       AdamConfig
	m         []*tensor.Tensor // first moment estimates
	v         []*tensor.Tensor // second moment estimates
	step      int              // optimization step counter
	numParams int              // number of parameters
}

// NewAdam creates a new Adam optimizer for the given parameters.
func NewAdam(cfg AdamConfig, params []*tensor.Tensor) *Adam {
	// Initialize moment buffers to zeros
	m := make([]*tensor.Tensor, len(params))
	v := make([]*tensor.Tensor, len(params))

	for i, p := range params {
		shape := p.Shape()
		m[i] = tensor.ZerosTensor(shape)
		v[i] = tensor.ZerosTensor(shape)
	}

	return &Adam{
		cfg:       cfg,
		m:         m,
		v:         v,
		step:      0,
		numParams: len(params),
	}
}

// Step performs one optimization step.
// params: parameters to update
// grads: gradients for each parameter
// step: global training step (for learning rate scheduling)
func (opt *Adam) Step(params, grads []*tensor.Tensor, globalStep int) {
	opt.step++
	t := float64(opt.step)

	// Compute bias correction factors
	beta1Pow := math.Pow(opt.cfg.Beta1, t)
	beta2Pow := math.Pow(opt.cfg.Beta2, t)
	biasCorrection1 := 1.0 - beta1Pow
	biasCorrection2 := 1.0 - beta2Pow

	// Linear learning rate decay
	lr := opt.cfg.LearningRate * (1.0 - float64(globalStep)/1000.0)
	if lr < 0 {
		lr = 0
	}

	// Update each parameter
	for i := range params {
		if grads[i] == nil {
			continue
		}

		// m = beta1 * m + (1 - beta1) * grad
		opt.m[i] = opt.m[i].Mul(tensor.FromList(opt.cfg.Beta1)).Add(
			grads[i].Mul(tensor.FromList(1.0-opt.cfg.Beta1)),
		)

		// v = beta2 * v + (1 - beta2) * grad^2
		gradSquared := grads[i].Mul(grads[i])
		opt.v[i] = opt.v[i].Mul(tensor.FromList(opt.cfg.Beta2)).Add(
			gradSquared.Mul(tensor.FromList(1.0-opt.cfg.Beta2)),
		)

		// Bias-corrected estimates
		mHat := opt.m[i].Div(tensor.FromList(biasCorrection1))
		vHat := opt.v[i].Div(tensor.FromList(biasCorrection2))

		// param -= lr * m_hat / (sqrt(v_hat) + eps)
		update := mHat.Div(vHat.Sqrt().Add(tensor.FromList(opt.cfg.Eps))).Mul(tensor.FromList(lr))
		params[i] = params[i].Sub(update)
	}
}

// StepWithLR performs one optimization step with explicit learning rate.
// This allows for external learning rate scheduling.
func (opt *Adam) StepWithLR(params, grads []*tensor.Tensor, lr float64) {
	opt.step++
	t := float64(opt.step)

	// Compute bias correction factors
	beta1Pow := math.Pow(opt.cfg.Beta1, t)
	beta2Pow := math.Pow(opt.cfg.Beta2, t)
	biasCorrection1 := 1.0 - beta1Pow
	biasCorrection2 := 1.0 - beta2Pow

	// Update each parameter
	for i := range params {
		if grads[i] == nil {
			continue
		}

		// m = beta1 * m + (1 - beta1) * grad
		opt.m[i] = opt.m[i].Mul(tensor.FromList(opt.cfg.Beta1)).Add(
			grads[i].Mul(tensor.FromList(1.0-opt.cfg.Beta1)),
		)

		// v = beta2 * v + (1 - beta2) * grad^2
		gradSquared := grads[i].Mul(grads[i])
		opt.v[i] = opt.v[i].Mul(tensor.FromList(opt.cfg.Beta2)).Add(
			gradSquared.Mul(tensor.FromList(1.0-opt.cfg.Beta2)),
		)

		// Bias-corrected estimates
		mHat := opt.m[i].Div(tensor.FromList(biasCorrection1))
		vHat := opt.v[i].Div(tensor.FromList(biasCorrection2))

		// param -= lr * m_hat / (sqrt(v_hat) + eps)
		update := mHat.Div(vHat.Sqrt().Add(tensor.FromList(opt.cfg.Eps))).Mul(tensor.FromList(lr))
		params[i] = params[i].Sub(update)
	}
}

// Reset clears the optimizer state (moment buffers).
func (opt *Adam) Reset() {
	for i := range opt.m {
		shape := opt.m[i].Shape()
		opt.m[i] = tensor.ZerosTensor(shape)
		opt.v[i] = tensor.ZerosTensor(shape)
	}
	opt.step = 0
}

// GetStep returns the current step count.
func (opt *Adam) GetStep() int {
	return opt.step
}

// SetStep sets the step count (useful for resuming training).
func (opt *Adam) SetStep(step int) {
	opt.step = step
}

// GetLR returns the current learning rate (after decay).
func (opt *Adam) GetLR(globalStep int) float64 {
	lr := opt.cfg.LearningRate * (1.0 - float64(globalStep)/1000.0)
	if lr < 0 {
		return 0
	}
	return lr
}

// GetMoment1 returns the first moment buffer for parameter i.
func (opt *Adam) GetMoment1(i int) *tensor.Tensor {
	if i < 0 || i >= len(opt.m) {
		return nil
	}
	return opt.m[i]
}

// GetMoment2 returns the second moment buffer for parameter i.
func (opt *Adam) GetMoment2(i int) *tensor.Tensor {
	if i < 0 || i >= len(opt.v) {
		return nil
	}
	return opt.v[i]
}

// ComputeLRDecay computes the learning rate with linear decay.
func ComputeLRDecay(baseLR float64, step, totalSteps int) float64 {
	lr := baseLR * (1.0 - float64(step)/float64(totalSteps))
	if lr < 0 {
		return 0
	}
	return lr
}

// GradientClip clips gradients by norm.
func GradientClip(grads []*tensor.Tensor, maxNorm float64) []*tensor.Tensor {
	// Compute global norm
	totalNormSq := 0.0
	for _, g := range grads {
		if g != nil {
			// L2 norm of gradient
			g2 := g.Mul(g).Sum()
			totalNormSq += g2.Item()
		}
	}
	totalNorm := math.Sqrt(totalNormSq)

	// Clip if necessary
	if totalNorm > maxNorm {
		scale := maxNorm / totalNorm
		clipped := make([]*tensor.Tensor, len(grads))
		for i, g := range grads {
			if g != nil {
				clipped[i] = g.Mul(tensor.FromList(scale))
			}
		}
		return clipped
	}

	return grads
}
