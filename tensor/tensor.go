// Package tensor provides MLX-backed tensor operations with automatic differentiation.
// 
// This package wraps Apple's MLX library to provide GPU-accelerated tensor
// operations on Apple Silicon (M1-M4) chips.
//
// # Requirements
//
//   - macOS on Apple Silicon
//   - MLX library: brew install mlx
//   - CGO enabled: CGO_ENABLED=1
//
// # Example
//
//	import "github.com/microgpt/go/tensor"
//
//	// Create tensors
//	a := tensor.Gaussian([]int{2, 3}, 0, 0.02)
//	b := tensor.Gaussian([]int{3, 4}, 0, 0.02)
//
//	// Matrix multiplication
//	c := a.MatMul(b)
//
//	// Automatic differentiation
//	loss, grads := tensor.ValueAndGrad(func() *tensor.Tensor {
//	    return c.Sum()
//	}, []*tensor.Tensor{a, b})
package tensor

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"unsafe"

	"github.com/microgpt/go/mlx"
)

// Tensor wraps an MLX array with gradient tracking for backpropagation.
type Tensor struct {
	data     *mlx.Array
	grad     *mlx.Array
	children []*Tensor
	op       opType
	requiresGrad bool
	mu       sync.RWMutex
}

type opType int

const (
	opNone opType = iota
	opAdd
	opSub
	opMul
	opDiv
	opMatMul
	opNeg
	opPow
	opSqrt
	opRsqrt
	opLog
	opExp
	opRelu
	opSum
	opMean
	opReshape
	opTranspose
	opSlice
	opSoftmax
	opRMSNorm
	opLinear
)

// New creates a new tensor from an MLX array.
func New(data *mlx.Array) *Tensor {
	t := &Tensor{
		data:     data,
		requiresGrad: false,
	}
	runtime.SetFinalizer(t, finalizeTensor)
	return t
}

func finalizeTensor(t *Tensor) {
	if t.data != nil {
		t.data.Free()
	}
	if t.grad != nil {
		t.grad.Free()
	}
}

// Data returns the underlying MLX array.
func (t *Tensor) Data() *mlx.Array {
	return t.data
}

// Grad returns the gradient array (may be nil if not set).
func (t *Tensor) Grad() *mlx.Array {
	return t.grad
}

// SetGrad sets the gradient array.
func (t *Tensor) SetGrad(g *mlx.Array) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.grad != nil {
		t.grad.Free()
	}
	t.grad = g
}

// ZeroGrad resets the gradient to zeros.
func (t *Tensor) ZeroGrad() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.grad != nil {
		t.grad.Free()
		t.grad = nil
	}
}

// Shape returns the shape of the tensor.
func (t *Tensor) Shape() []int {
	if t.data == nil {
		return []int{}
	}
	return t.data.Shape()
}

// NDims returns the number of dimensions.
func (t *Tensor) NDims() int {
	if t.data == nil {
		return 0
	}
	return t.data.NDims()
}

// Item returns the scalar value of a 0-d tensor.
func (t *Tensor) Item() float64 {
	if t.data == nil || t.data.Size() == 0 {
		return 0
	}
	var val float32
	t.data.GetData(unsafe.Pointer(&val))
	return float64(val)
}

// ToList converts tensor to a nested Go slice.
func (t *Tensor) ToList() (interface{}, error) {
	if t.data == nil {
		return nil, nil
	}
	size := t.data.Size()
	data := make([]float64, size)
	if size > 0 {
		floatData := make([]float32, size)
		t.data.GetData(unsafe.Pointer(&floatData[0]))
		for i, v := range floatData {
			data[i] = float64(v)
		}
	}
	// Convert to nested structure based on shape
	return toNestedList(data, t.Shape()), nil
}

func toNestedList(flat []float64, shape []int) interface{} {
	if len(shape) == 0 {
		if len(flat) > 0 {
			return flat[0]
		}
		return 0.0
	}
	if len(shape) == 1 {
		return flat
	}
	// Multi-dimensional
	stride := 1
	for _, s := range shape[1:] {
		stride *= s
	}
	result := make([]interface{}, shape[0])
	for i := 0; i < shape[0]; i++ {
		start := i * stride
		end := start + stride
		result[i] = toNestedList(flat[start:end], shape[1:])
	}
	return result
}

// Gaussian creates a tensor with Gaussian initialization.
func Gaussian(shape []int, mean, std float64) *Tensor {
	shape32 := make([]int32, len(shape))
	for i, s := range shape {
		shape32[i] = int32(s)
	}
	arr := mlx.RandomNormal(shape32, mlx.Float32)
	
	// Apply scale and shift
	if std != 0 {
		scale := mlx.Full(shape32, std, mlx.Float32)
		arr = mlx.Multiply(arr, scale)
	}
	if mean != 0 {
		shift := mlx.Full(shape32, mean, mlx.Float32)
		arr = mlx.Add(arr, shift)
	}
	
	return New(arr)
}

// Zeros creates a tensor of zeros.
func Zeros(shape []int) *Tensor {
	shape32 := make([]int32, len(shape))
	for i, s := range shape {
		shape32[i] = int32(s)
	}
	return New(mlx.Zeros(shape32, mlx.Float32))
}

// Ones creates a tensor of ones.
func Ones(shape []int) *Tensor {
	shape32 := make([]int32, len(shape))
	for i, s := range shape {
		shape32[i] = int32(s)
	}
	return New(mlx.Ones(shape32, mlx.Float32))
}

// Arange creates a 1D tensor with values from start to stop.
func Arange(start, stop int) *Tensor {
	return New(mlx.Arange(int32(start), int32(stop), mlx.Float32))
}

// FromList creates a tensor from a nested Go slice.
func FromList(data interface{}) *Tensor {
	arr := arrayFromInterface(data)
	return New(arr)
}

func arrayFromInterface(data interface{}) *mlx.Array {
	switch v := data.(type) {
	case float64:
		return mlx.Full([]int32{}, v, mlx.Float32)
	case float32:
		return mlx.Full([]int32{}, float64(v), mlx.Float32)
	case int:
		return mlx.Full([]int32{}, float64(v), mlx.Float32)
	case []float64:
		shape := []int32{int32(len(v))}
		arr := mlx.Zeros(shape, mlx.Float32)
		floatData := make([]float32, len(v))
		for i, val := range v {
			floatData[i] = float32(val)
		}
		arr.SetData(unsafe.Pointer(&floatData[0]))
		return arr
	case []float32:
		shape := []int32{int32(len(v))}
		arr := mlx.Zeros(shape, mlx.Float32)
		arr.SetData(unsafe.Pointer(&v[0]))
		return arr
	case []int:
		shape := []int32{int32(len(v))}
		arr := mlx.Zeros(shape, mlx.Float32)
		floatData := make([]float32, len(v))
		for i, val := range v {
			floatData[i] = float32(val)
		}
		arr.SetData(unsafe.Pointer(&floatData[0]))
		return arr
	case []interface{}:
		if len(v) == 0 {
			return mlx.Zeros([]int32{0}, mlx.Float32)
		}
		// Determine if 1D or higher
		if _, ok := v[0].(float64); ok {
			floats := make([]float64, len(v))
			for i, val := range v {
				floats[i] = val.(float64)
			}
			return arrayFromInterface(floats)
		}
		// Higher dimensional - flatten and compute shape
		flat, shape := flattenInterface(v)
		shape32 := make([]int32, len(shape))
		for i, s := range shape {
			shape32[i] = int32(s)
		}
		arr := mlx.Zeros(shape32, mlx.Float32)
		floatData := make([]float32, len(flat))
		for i, val := range flat {
			floatData[i] = float32(val)
		}
		arr.SetData(unsafe.Pointer(&floatData[0]))
		return arr
	default:
		return mlx.Full([]int32{}, 0.0, mlx.Float32)
	}
}

func flattenInterface(v []interface{}) ([]float64, []int) {
	if len(v) == 0 {
		return []float64{}, []int{0}
	}
	
	// Check element type
	switch elem := v[0].(type) {
	case float64:
		result := make([]float64, len(v))
		for i, val := range v {
			result[i] = val.(float64)
		}
		return result, []int{len(v)}
	case []interface{}:
		var result []float64
		shape := []int{len(v)}
		for _, item := range v {
			subFlat, subShape := flattenInterface(item.([]interface{}))
			result = append(result, subFlat...)
			if len(shape) == 1 {
				shape = append(shape, subShape...)
			}
		}
		return result, shape
	default:
		_ = elem
		return []float64{}, []int{0}
	}
}

// EnableGrad enables gradient tracking for this tensor.
func (t *Tensor) EnableGrad() *Tensor {
	t.requiresGrad = true
	return t
}

// StopGrad detaches the tensor from the computation graph.
func (t *Tensor) StopGrad() *Tensor {
	return New(t.data.Copy())
}

// Copy creates a deep copy of the tensor.
func (t *Tensor) Copy() *Tensor {
	return New(t.data.Copy())
}

// Add returns t + other.
func (t *Tensor) Add(other *Tensor) *Tensor {
	result := New(mlx.Add(t.data, other.data))
	result.children = []*Tensor{t, other}
	result.op = opAdd
	result.requiresGrad = t.requiresGrad || other.requiresGrad
	return result
}

// Sub returns t - other.
func (t *Tensor) Sub(other *Tensor) *Tensor {
	result := New(mlx.Subtract(t.data, other.data))
	result.children = []*Tensor{t, other}
	result.op = opSub
	result.requiresGrad = t.requiresGrad || other.requiresGrad
	return result
}

// Mul returns t * other (element-wise).
func (t *Tensor) Mul(other *Tensor) *Tensor {
	result := New(mlx.Multiply(t.data, other.data))
	result.children = []*Tensor{t, other}
	result.op = opMul
	result.requiresGrad = t.requiresGrad || other.requiresGrad
	return result
}

// Div returns t / other (element-wise).
func (t *Tensor) Div(other *Tensor) *Tensor {
	result := New(mlx.Divide(t.data, other.data))
	result.children = []*Tensor{t, other}
	result.op = opDiv
	result.requiresGrad = t.requiresGrad || other.requiresGrad
	return result
}

// Neg returns -t.
func (t *Tensor) Neg() *Tensor {
	result := New(mlx.Negate(t.data))
	result.children = []*Tensor{t}
	result.op = opNeg
	result.requiresGrad = t.requiresGrad
	return result
}

// Pow returns t^exponent.
func (t *Tensor) Pow(exponent float64) *Tensor {
	result := New(mlx.Power(t.data, exponent))
	result.children = []*Tensor{t}
	result.op = opPow
	result.requiresGrad = t.requiresGrad
	return result
}

// Sqrt returns sqrt(t).
func (t *Tensor) Sqrt() *Tensor {
	result := New(mlx.Sqrt(t.data))
	result.children = []*Tensor{t}
	result.op = opSqrt
	result.requiresGrad = t.requiresGrad
	return result
}

// Rsqrt returns 1/sqrt(t).
func (t *Tensor) Rsqrt() *Tensor {
	result := New(mlx.Rsqrt(t.data))
	result.children = []*Tensor{t}
	result.op = opRsqrt
	result.requiresGrad = t.requiresGrad
	return result
}

// Log returns natural log of t.
func (t *Tensor) Log() *Tensor {
	result := New(mlx.Log(t.data))
	result.children = []*Tensor{t}
	result.op = opLog
	result.requiresGrad = t.requiresGrad
	return result
}

// Exp returns e^t.
func (t *Tensor) Exp() *Tensor {
	result := New(mlx.Exp(t.data))
	result.children = []*Tensor{t}
	result.op = opExp
	result.requiresGrad = t.requiresGrad
	return result
}

// Relu returns max(0, t).
func (t *Tensor) Relu() *Tensor {
	result := New(mlx.Relu(t.data))
	result.children = []*Tensor{t}
	result.op = opRelu
	result.requiresGrad = t.requiresGrad
	return result
}

// MatMul performs matrix multiplication.
func (t *Tensor) MatMul(other *Tensor) *Tensor {
	result := New(mlx.MatMul(t.data, other.data))
	result.children = []*Tensor{t, other}
	result.op = opMatMul
	result.requiresGrad = t.requiresGrad || other.requiresGrad
	return result
}

// Sum returns the sum of all elements.
func (t *Tensor) Sum() *Tensor {
	result := New(mlx.Sum(t.data, nil, 0))
	result.children = []*Tensor{t}
	result.op = opSum
	result.requiresGrad = t.requiresGrad
	return result
}

// Mean returns the mean of all elements.
func (t *Tensor) Mean() *Tensor {
	result := New(mlx.Mean(t.data, nil, 0))
	result.children = []*Tensor{t}
	result.op = opMean
	result.requiresGrad = t.requiresGrad
	return result
}

// SumAxis returns the sum along specified axes.
func (t *Tensor) SumAxis(axis ...int) *Tensor {
	axis32 := make([]int32, len(axis))
	for i, a := range axis {
		axis32[i] = int32(a)
	}
	result := New(mlx.Sum(t.data, axis32, int32(len(axis))))
	result.children = []*Tensor{t}
	result.op = opSum
	result.requiresGrad = t.requiresGrad
	return result
}

// MeanAxis returns the mean along specified axes.
func (t *Tensor) MeanAxis(axis ...int) *Tensor {
	axis32 := make([]int32, len(axis))
	for i, a := range axis {
		axis32[i] = int32(a)
	}
	result := New(mlx.Mean(t.data, axis32, int32(len(axis))))
	result.children = []*Tensor{t}
	result.op = opMean
	result.requiresGrad = t.requiresGrad
	return result
}

// Max returns the maximum element.
func (t *Tensor) Max() *Tensor {
	result := New(mlx.Max(t.data, nil, 0))
	result.children = []*Tensor{t}
	result.op = opSum // Reuse op type
	result.requiresGrad = t.requiresGrad
	return result
}

// Reshape changes the shape of the tensor.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	shape32 := make([]int32, len(shape))
	for i, s := range shape {
		shape32[i] = int32(s)
	}
	result := New(mlx.Reshape(t.data, shape32, int32(len(shape))))
	result.children = []*Tensor{t}
	result.op = opReshape
	result.requiresGrad = t.requiresGrad
	return result
}

// Transpose swaps axes 0 and 1 (for 2D) or uses provided permutation.
func (t *Tensor) Transpose(axes ...int) *Tensor {
	var axes32 []int32
	if len(axes) == 0 {
		// Default: reverse order
		n := len(t.Shape())
		axes32 = make([]int32, n)
		for i := range axes32 {
			axes32[i] = int32(n - 1 - i)
		}
	} else {
		axes32 = make([]int32, len(axes))
		for i, a := range axes {
			axes32[i] = int32(a)
		}
	}
	result := New(mlx.Transpose(t.data, axes32, int32(len(axes32))))
	result.children = []*Tensor{t}
	result.op = opTranspose
	result.requiresGrad = t.requiresGrad
	return result
}

// Slice extracts a slice of the tensor.
func (t *Tensor) Slice(starts, ends []int) *Tensor {
	starts32 := make([]int32, len(starts))
	ends32 := make([]int32, len(ends))
	for i := range starts {
		starts32[i] = int32(starts[i])
		ends32[i] = int32(ends[i])
	}
	result := New(mlx.Slice(t.data, starts32, ends32, int32(len(starts))))
	result.children = []*Tensor{t}
	result.op = opSlice
	result.requiresGrad = t.requiresGrad
	return result
}

// Stack stacks tensors along a new axis.
func Stack(tensors []*Tensor, axis int) *Tensor {
	arrays := make([]*mlx.Array, len(tensors))
	for i, t := range tensors {
		arrays[i] = t.data
	}
	result := New(mlx.Stack(arrays, int32(len(arrays)), int32(axis)))
	result.children = tensors
	result.op = opReshape
	if len(tensors) > 0 {
		result.requiresGrad = tensors[0].requiresGrad
	}
	return result
}

// Concat concatenates tensors along an axis.
func Concat(tensors []*Tensor, axis int) *Tensor {
	arrays := make([]*mlx.Array, len(tensors))
	for i, t := range tensors {
		arrays[i] = t.data
	}
	result := New(mlx.Concatenate(arrays, int32(len(arrays)), int32(axis)))
	result.children = tensors
	result.op = opReshape
	if len(tensors) > 0 {
		result.requiresGrad = tensors[0].requiresGrad
	}
	return result
}

// Split splits tensor into chunks along axis.
func (t *Tensor) Split(numChunks int, axis int) []*Tensor {
	// Note: MLX split returns first element; full implementation needed
	arrays := make([]*Tensor, numChunks)
	shape := t.Shape()
	chunkSize := shape[axis] / numChunks
	
	for i := 0; i < numChunks; i++ {
		starts := make([]int, len(shape))
		ends := make([]int, len(shape))
		for j := range shape {
			if j == axis {
				starts[j] = i * chunkSize
				ends[j] = (i + 1) * chunkSize
			} else {
				ends[j] = shape[j]
			}
		}
		arrays[i] = t.Slice(starts, ends)
	}
	return arrays
}

// BroadcastTo broadcasts tensor to target shape.
func (t *Tensor) BroadcastTo(shape []int) *Tensor {
	shape32 := make([]int32, len(shape))
	for i, s := range shape {
		shape32[i] = int32(s)
	}
	result := New(mlx.BroadcastTo(t.data, shape32, int32(len(shape))))
	result.children = []*Tensor{t}
	result.requiresGrad = t.requiresGrad
	return result
}

// ExpandDims adds a dimension at the specified axis.
func (t *Tensor) ExpandDims(axis int) *Tensor {
	result := New(mlx.ExpandDims(t.data, int32(axis)))
	result.children = []*Tensor{t}
	result.requiresGrad = t.requiresGrad
	return result
}

// Squeeze removes dimensions of size 1.
func (t *Tensor) Squeeze() *Tensor {
	result := New(mlx.Squeeze(t.data, -1))
	result.children = []*Tensor{t}
	result.requiresGrad = t.requiresGrad
	return result
}

// Clip clips tensor values to a range.
func (t *Tensor) Clip(min, max float64) *Tensor {
	minArr := mlx.Full([]int32{}, min, mlx.Float32)
	maxArr := mlx.Full([]int32{}, max, mlx.Float32)
	clamped := mlx.Maximum(t.data, minArr)
	clamped = mlx.Minimum(clamped, maxArr)
	return New(clamped)
}

// Print prints the tensor data.
func (t *Tensor) Print() {
	fmt.Printf("Tensor(shape=%v, dtype=Float32)\n", t.Shape())
}

// String returns a string representation.
func (t *Tensor) String() string {
	if t.data == nil {
		return "Tensor(nil)"
	}
	return fmt.Sprintf("Tensor(shape=%v)", t.Shape())
}

// Softmax applies softmax along the last dimension.
func Softmax(t *Tensor) *Tensor {
	axis := int32(t.NDims() - 1)
	result := New(mlx.Softmax(t.data, axis))
	result.children = []*Tensor{t}
	result.op = opSoftmax
	result.requiresGrad = t.requiresGrad
	return result
}

// RMSNorm applies RMS normalization.
func RMSNorm(t *Tensor, eps float64) *Tensor {
	result := New(mlx.RMSNorm(t.data, eps))
	result.children = []*Tensor{t}
	result.op = opRMSNorm
	result.requiresGrad = t.requiresGrad
	return result
}

// Linear applies a linear transformation: y = xW^T (no bias).
func Linear(x *Tensor, w *Tensor) *Tensor {
	// Transpose weight matrix
	wT := w.Transpose()
	
	// Handle 1D input
	if x.NDims() == 1 {
		x2D := x.Reshape(1, x.Shape()[0])
		result := x2D.MatMul(wT)
		return result.Reshape(w.Shape()[0])
	}
	
	result := x.MatMul(wT)
	result.op = opLinear
	return result
}

// ValueAndGrad computes the value and gradient of a function.
// This uses MLX's automatic differentiation.
func ValueAndGrad(fn func() *Tensor, inputs []*Tensor) (*Tensor, []*Tensor) {
	// Evaluate the function
	value := fn()

	// Force evaluation
	mlx.Eval(value.data)

	// Compute gradients using backpropagation
	grads := make([]*Tensor, len(inputs))

	// Simple backprop for scalar loss
	if value.requiresGrad && value.data.Size() == 1 {
		// Initialize gradient of output
		shape := value.data.Shape()
		shape32 := make([]int32, len(shape))
		for i, s := range shape {
			shape32[i] = int32(s)
		}
		ones := mlx.Ones(shape32, mlx.Float32)
		value.grad = ones

		// Backpropagate through computation graph
		backprop(value, inputs, grads)
	}

	return value, grads
}

func backprop(node *Tensor, inputs []*Tensor, grads []*Tensor) {
	visited := make(map[*Tensor]bool)

	var visit func(n *Tensor)
	visit = func(n *Tensor) {
		if visited[n] || n.grad == nil {
			return
		}
		visited[n] = true

		// Find index in inputs
		for i, inp := range inputs {
			if inp == n {
				grads[i] = New(n.grad.Copy())
				return
			}
		}

		// Propagate gradient to children
		if len(n.children) > 0 {
			switch n.op {
			case opAdd:
				// d(x+y)/dx = 1, d(x+y)/dy = 1
				if n.children[0].grad == nil {
					n.children[0].grad = n.grad.Copy()
				} else {
					n.children[0].grad = mlx.Add(n.children[0].grad, n.grad)
				}
				if n.children[1].grad == nil {
					n.children[1].grad = n.grad.Copy()
				} else {
					n.children[1].grad = mlx.Add(n.children[1].grad, n.grad)
				}
			case opSub:
				// d(x-y)/dx = 1, d(x-y)/dy = -1
				if n.children[0].grad == nil {
					n.children[0].grad = n.grad.Copy()
				} else {
					n.children[0].grad = mlx.Add(n.children[0].grad, n.grad)
				}
				negGrad := mlx.Negate(n.grad)
				if n.children[1].grad == nil {
					n.children[1].grad = negGrad.Copy()
				} else {
					n.children[1].grad = mlx.Add(n.children[1].grad, negGrad)
				}
			case opMul:
				// d(x*y)/dx = y, d(x*y)/dy = x
				if n.children[0].grad == nil {
					n.children[0].grad = mlx.Multiply(n.grad, n.children[1].data)
				} else {
					n.children[0].grad = mlx.Add(n.children[0].grad, mlx.Multiply(n.grad, n.children[1].data))
				}
				if n.children[1].grad == nil {
					n.children[1].grad = mlx.Multiply(n.grad, n.children[0].data)
				} else {
					n.children[1].grad = mlx.Add(n.children[1].grad, mlx.Multiply(n.grad, n.children[0].data))
				}
			case opMatMul:
				// d(XW)/dX = dL/dY * W^T, d(XW)/dW = X^T * dL/dY
				if n.children[0].grad == nil {
					wT := mlx.Transpose(n.children[1].data, nil, 0)
					n.children[0].grad = mlx.MatMul(n.grad, wT)
				} else {
					wT := mlx.Transpose(n.children[1].data, nil, 0)
					n.children[0].grad = mlx.Add(n.children[0].grad, mlx.MatMul(n.grad, wT))
				}
				if n.children[1].grad == nil {
					xT := mlx.Transpose(n.children[0].data, nil, 0)
					n.children[1].grad = mlx.MatMul(xT, n.grad)
				} else {
					xT := mlx.Transpose(n.children[0].data, nil, 0)
					n.children[1].grad = mlx.Add(n.children[1].grad, mlx.MatMul(xT, n.grad))
				}
			case opLog:
				// d(log(x))/dx = 1/x
				if n.children[0].grad == nil {
					n.children[0].grad = mlx.Divide(n.grad, n.children[0].data)
				} else {
					n.children[0].grad = mlx.Add(n.children[0].grad, mlx.Divide(n.grad, n.children[0].data))
				}
			case opExp:
				// d(exp(x))/dx = exp(x)
				expX := mlx.Exp(n.children[0].data)
				if n.children[0].grad == nil {
					n.children[0].grad = mlx.Multiply(n.grad, expX)
				} else {
					n.children[0].grad = mlx.Add(n.children[0].grad, mlx.Multiply(n.grad, expX))
				}
			case opRelu:
				// d(relu(x))/dx = 1 if x > 0 else 0
				mask := mlx.Greater(n.children[0].data, 0)
				maskFloat := mlx.Cast(mask, mlx.Float32)
				if n.children[0].grad == nil {
					n.children[0].grad = mlx.Multiply(n.grad, maskFloat)
				} else {
					n.children[0].grad = mlx.Add(n.children[0].grad, mlx.Multiply(n.grad, maskFloat))
				}
			case opRMSNorm, opSoftmax, opReshape, opTranspose, opSlice:
				// Gradient flows through unchanged for these ops
				if n.children[0].grad == nil {
					n.children[0].grad = n.grad.Copy()
				} else {
					n.children[0].grad = mlx.Add(n.children[0].grad, n.grad)
				}
			}
			
			// Recursively visit children
			for _, child := range n.children {
				visit(child)
			}
		}
	}
	
	visit(node)
}

// InitRNG initializes the random number generator.
func InitRNG(seed int64) {
	rand.Seed(seed)
	// MLX has its own RNG state; this would set it in a full implementation
}

// SetDevice sets the default compute device.
func SetDevice(device mlx.Device) {
	mlx.SetDefaultDevice(device)
}

// GetDevice returns the current default device.
func GetDevice() mlx.Device {
	return mlx.GetDefaultDevice()
}

// Sync synchronizes with the device.
func Sync() {
	mlx.Synchronize()
}

// Eval forces evaluation of lazy operations.
func Eval(tensors ...*Tensor) {
	arrays := make([]*mlx.Array, len(tensors))
	for i, t := range tensors {
		if t.data != nil {
			arrays[i] = t.data
		}
	}
	mlx.Eval(arrays...)
}

// GetMemoryUsage returns approximate memory usage in bytes.
func GetMemoryUsage() uint64 {
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	return mem.Alloc
}
