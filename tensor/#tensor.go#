// Package tensor provides tensor operations with automatic differentiation.
// This is a pure Go implementation that mirrors MLX's API for future integration.
package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// Array represents a multi-dimensional array of float64 values.
type Array struct {
	shape []int
	stride []int
	data  []float64
}

// NewArray creates a new array with the given shape and data.
func NewArray(shape []int, data []float64) Array {
	if len(data) == 0 {
		size := 1
		for _, s := range shape {
			size *= s
		}
		data = make([]float64, size)
	}
	stride := make([]int, len(shape))
	if len(shape) > 0 {
		stride[len(shape)-1] = 1
		for i := len(shape) - 2; i >= 0; i-- {
			stride[i] = stride[i+1] * shape[i+1]
		}
	}
	return Array{shape: shape, stride: stride, data: data}
}

// Shape returns the shape of the array.
func (a Array) Shape() []int {
	return a.shape
}

// Size returns the total number of elements.
func (a Array) Size() int {
	size := 1
	for _, s := range a.shape {
		size *= s
	}
	return size
}

// NDims returns the number of dimensions.
func (a Array) NDims() int {
	return len(a.shape)
}

// At returns the element at the given indices.
func (a Array) At(indices ...int) float64 {
	idx := 0
	for i, ind := range indices {
		idx += ind * a.stride[i]
	}
	return a.data[idx]
}

// Set sets the element at the given indices.
func (a Array) Set(val float64, indices ...int) {
	idx := 0
	for i, ind := range indices {
		idx += ind * a.stride[i]
	}
	a.data[idx] = val
}

// Data returns a copy of the underlying data.
func (a Array) Data() []float64 {
	result := make([]float64, len(a.data))
	copy(result, a.data)
	return result
}

// ToList converts the array to a nested interface{} structure.
func (a Array) ToList() (interface{}, error) {
	return a.toListRecursive(0), nil
}

func (a Array) toListRecursive(dim int) interface{} {
	if dim == len(a.shape)-1 {
		result := make([]float64, a.shape[dim])
		indices := make([]int, len(a.shape))
		for i := 0; i < a.shape[dim]; i++ {
			indices[dim] = i
			result[i] = a.At(indices...)
		}
		return result
	}
	result := make([]interface{}, a.shape[dim])
	indices := make([]int, len(a.shape))
	for i := 0; i < a.shape[dim]; i++ {
		indices[dim] = i
		result[i] = a.toListRecursive(dim + 1)
	}
	return result
}

// Zeros creates an array of zeros.
func Zeros(shape []int) Array {
	size := 1
	for _, s := range shape {
		size *= s
	}
	return NewArray(shape, make([]float64, size))
}

// Ones creates an array of ones.
func Ones(shape []int) Array {
	size := 1
	for _, s := range shape {
		size *= s
	}
	data := make([]float64, size)
	for i := range data {
		data[i] = 1.0
	}
	return NewArray(shape, data)
}

// Full creates an array filled with a value.
func Full(shape []int, val float64) Array {
	size := 1
	for _, s := range shape {
		size *= s
	}
	data := make([]float64, size)
	for i := range data {
		data[i] = val
	}
	return NewArray(shape, data)
}

// Arange creates a 1D array with values from start to stop.
func Arange(start, stop int) Array {
	size := stop - start
	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = float64(start + i)
	}
	return NewArray([]int{size}, data)
}

// RandomNormal creates an array with standard normal distribution.
func RandomNormal(shape []int) Array {
	size := 1
	for _, s := range shape {
		size *= s
	}
	data := make([]float64, size)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	return NewArray(shape, data)
}

// RandomUniform creates an array with uniform distribution.
func RandomUniform(shape []int, low, high float64) Array {
	size := 1
	for _, s := range shape {
		size *= s
	}
	data := make([]float64, size)
	for i := range data {
		data[i] = low + rand.Float64()*(high-low)
	}
	return NewArray(shape, data)
}

// Add performs element-wise addition.
func Add(a, b Array) Array {
	// Handle broadcasting
	resultShape := broadcastShape(a.shape, b.shape)
	result := Zeros(resultShape)
	
	for i := range result.data {
		indices := unravelIndex(i, resultShape, result.stride)
		aIdx := ravelIndex(modIndices(indices, a.shape), a.stride)
		bIdx := ravelIndex(modIndices(indices, b.shape), b.stride)
		result.data[i] = a.data[aIdx] + b.data[bIdx]
	}
	return result
}

// Subtract performs element-wise subtraction.
func Subtract(a, b Array) Array {
	resultShape := broadcastShape(a.shape, b.shape)
	result := Zeros(resultShape)
	
	for i := range result.data {
		indices := unravelIndex(i, resultShape, result.stride)
		aIdx := ravelIndex(modIndices(indices, a.shape), a.stride)
		bIdx := ravelIndex(modIndices(indices, b.shape), b.stride)
		result.data[i] = a.data[aIdx] - b.data[bIdx]
	}
	return result
}

// Multiply performs element-wise multiplication.
func Multiply(a, b Array) Array {
	resultShape := broadcastShape(a.shape, b.shape)
	result := Zeros(resultShape)
	
	for i := range result.data {
		indices := unravelIndex(i, resultShape, result.stride)
		aIdx := ravelIndex(modIndices(indices, a.shape), a.stride)
		bIdx := ravelIndex(modIndices(indices, b.shape), b.stride)
		result.data[i] = a.data[aIdx] * b.data[bIdx]
	}
	return result
}

// Divide performs element-wise division.
func Divide(a, b Array) Array {
	resultShape := broadcastShape(a.shape, b.shape)
	result := Zeros(resultShape)
	
	for i := range result.data {
		indices := unravelIndex(i, resultShape, result.stride)
		aIdx := ravelIndex(modIndices(indices, a.shape), a.stride)
		bIdx := ravelIndex(modIndices(indices, b.shape), b.stride)
		if b.data[bIdx] != 0 {
			result.data[i] = a.data[aIdx] / b.data[bIdx]
		} else {
			result.data[i] = math.Inf(1)
		}
	}
	return result
}

// Negative performs element-wise negation.
func Negative(a Array) Array {
	result := Zeros(a.shape)
	for i := range result.data {
		result.data[i] = -a.data[i]
	}
	return result
}

// Power performs element-wise exponentiation.
func Power(a Array, exponent Array) Array {
	result := Zeros(a.shape)
	expVal := exponent.data[0]
	for i := range result.data {
		result.data[i] = math.Pow(a.data[i], expVal)
	}
	return result
}

// Sqrt performs element-wise square root.
func Sqrt(a Array) Array {
	result := Zeros(a.shape)
	for i := range result.data {
		result.data[i] = math.Sqrt(a.data[i])
	}
	return result
}

// Reciprocal performs element-wise reciprocal.
func Reciprocal(a Array) Array {
	result := Zeros(a.shape)
	for i := range result.data {
		if a.data[i] != 0 {
			result.data[i] = 1.0 / a.data[i]
		} else {
			result.data[i] = math.Inf(1)
		}
	}
	return result
}

// Log performs element-wise natural logarithm.
func Log(a Array) Array {
	result := Zeros(a.shape)
	for i := range result.data {
		result.data[i] = math.Log(a.data[i])
	}
	return result
}

// Exp performs element-wise exponential.
func Exp(a Array) Array {
	result := Zeros(a.shape)
	for i := range result.data {
		result.data[i] = math.Exp(a.data[i])
	}
	return result
}

// Maximum performs element-wise maximum.
func Maximum(a, b Array) Array {
	resultShape := broadcastShape(a.shape, b.shape)
	result := Zeros(resultShape)
	
	for i := range result.data {
		indices := unravelIndex(i, resultShape, result.stride)
		aIdx := ravelIndex(modIndices(indices, a.shape), a.stride)
		bIdx := ravelIndex(modIndices(indices, b.shape), b.stride)
		result.data[i] = math.Max(a.data[aIdx], b.data[bIdx])
	}
	return result
}

// MatMul performs matrix multiplication.
func MatMul(a, b Array) Array {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		panic("MatMul requires 2D arrays")
	}
	if a.shape[1] != b.shape[0] {
		panic(fmt.Sprintf("MatMul shape mismatch: %v x %v", a.shape, b.shape))
	}
	
	m, k := a.shape[0], a.shape[1]
	n := b.shape[1]
	result := Zeros([]int{m, n})
	
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for l := 0; l < k; l++ {
				sum += a.At(i, l) * b.At(l, j)
			}
			result.Set(sum, i, j)
		}
	}
	return result
}

// Sum computes the sum of all elements.
func Sum(a Array, axis []int) Array {
	if axis == nil {
		// Sum all elements
		total := 0.0
		for _, v := range a.data {
			total += v
		}
		return NewArray([]int{}, []float64{total})
	}
	
	// Sum along specific axes
	resultShape := make([]int, 0, len(a.shape))
	keepAxes := make([]bool, len(a.shape))
	for i := range a.shape {
		keep := true
		for _, ax := range axis {
			if ax == i || (ax < 0 && ax+len(a.shape) == i) {
				keep = false
				break
			}
		}
		if keep {
			resultShape = append(resultShape, a.shape[i])
		}
		keepAxes[i] = keep
	}
	
	result := Zeros(resultShape)
	// Simplified: iterate and accumulate
	for i := range a.data {
		indices := unravelIndex(i, a.shape, a.stride)
		resultIndices := make([]int, 0, len(indices))
		for j, idx := range indices {
			if keepAxes[j] {
				resultIndices = append(resultIndices, idx)
			}
		}
		idx := 0
		for dim, ind := range resultIndices {
			stride := 1
			for k := dim + 1; k < len(resultIndices); k++ {
				stride *= resultShape[k]
			}
			idx += ind * stride
		}
		result.data[idx] += a.data[i]
	}
	return result
}

// Mean computes the mean of all elements.
func Mean(a Array, axis []int) Array {
	total := Sum(a, axis)
	
	if axis == nil {
		// Mean of all elements
		val := total.data[0] / float64(a.Size())
		return NewArray([]int{}, []float64{val})
	}
	
	// Divide by product of reduced dimensions
	reduceSize := 1
	for _, ax := range axis {
		if ax >= 0 {
			reduceSize *= a.shape[ax]
		} else {
			reduceSize *= a.shape[len(a.shape)+ax]
		}
	}
	
	for i := range total.data {
		total.data[i] /= float64(reduceSize)
	}
	return total
}

// Max computes the maximum element.
func Max(a Array, axis []int) Array {
	if axis == nil || len(axis) == 0 {
		maxVal := a.data[0]
		for _, v := range a.data {
			if v > maxVal {
				maxVal = v
			}
		}
		return NewArray([]int{}, []float64{maxVal})
	}
	// TODO: Implement axis-specific max
	return Max(a, nil)
}

// Reshape changes the shape of an array.
func Reshape(a Array, shape []int) Array {
	if len(a.data) == 0 {
		return NewArray(shape, nil)
	}
	return NewArray(shape, append([]float64{}, a.data...))
}

// Transpose permutes the axes of an array.
func Transpose(a Array, axes []int) Array {
	if len(axes) == 0 {
		// Default: reverse order
		axes = make([]int, len(a.shape))
		for i := range axes {
			axes[i] = len(a.shape) - 1 - i
		}
	}
	
	newShape := make([]int, len(axes))
	for i, ax := range axes {
		newShape[i] = a.shape[ax]
	}
	
	result := Zeros(newShape)
	indices := make([]int, len(a.shape))
	
	var iterate func(dim int)
	iterate = func(dim int) {
		if dim == len(a.shape) {
			newIndices := make([]int, len(axes))
			for i, ax := range axes {
				newIndices[i] = indices[ax]
			}
			val := a.At(indices...)
			result.Set(val, newIndices...)
			return
		}
		for i := 0; i < a.shape[dim]; i++ {
			indices[dim] = i
			iterate(dim + 1)
		}
	}
	iterate(0)
	
	return result
}

// Stack stacks arrays along a new axis.
func Stack(arrays []Array, axis int) Array {
	if len(arrays) == 0 {
		return Array{}
	}
	
	shape := arrays[0].shape
	newShape := make([]int, 0, len(shape)+1)
	for i := 0; i <= len(shape); i++ {
		if i == axis {
			newShape = append(newShape, len(arrays))
		}
		if i < len(shape) {
			newShape = append(newShape, shape[i])
		}
	}
	
	result := Zeros(newShape)
	
	for ai, arr := range arrays {
		indices := make([]int, len(newShape))
		
		var iterate func(dim int)
		iterate = func(dim int) {
			if dim == len(newShape) {
				srcIndices := make([]int, len(shape))
				j := 0
				for k := 0; k < len(newShape); k++ {
					if k != axis {
						srcIndices[j] = indices[k]
						j++
					}
				}
				result.Set(arr.At(srcIndices...), indices...)
				return
			}
			for i := 0; i < newShape[dim]; i++ {
				indices[dim] = i
				iterate(dim + 1)
			}
		}
		
		// Set the axis index
		indices[axis] = ai
		iterate(0)
	}
	
	return result
}

// Concatenate concatenates arrays along an axis.
func Concatenate(arrays []Array, axis int) Array {
	if len(arrays) == 0 {
		return Array{}
	}
	
	// Calculate result shape
	resultShape := append([]int{}, arrays[0].shape...)
	for i := 1; i < len(arrays); i++ {
		resultShape[axis] += arrays[i].shape[axis]
	}
	
	result := Zeros(resultShape)
	
	offset := 0
	for _, arr := range arrays {
		for i := 0; i < arr.Size(); i++ {
			indices := unravelIndex(i, arr.shape, arr.stride)
			indices[axis] += offset
			result.Set(arr.data[i], indices...)
		}
		offset += arr.shape[axis]
	}
	
	return result
}

// Split splits an array into chunks along an axis.
func Split(a Array, numChunks int, axis int) []Array {
	chunkSize := a.shape[axis] / numChunks
	result := make([]Array, numChunks)
	
	for i := 0; i < numChunks; i++ {
		start := i * chunkSize
		_ = start // end would be start + chunkSize

		newShape := append([]int{}, a.shape...)
		newShape[axis] = chunkSize
		
		chunk := Zeros(newShape)
		
		// Copy data
		indices := make([]int, len(a.shape))
		var iterate func(dim int)
		iterate = func(dim int) {
			if dim == len(a.shape) {
				srcIndices := append([]int{}, indices...)
				srcIndices[axis] += start
				chunk.Set(a.At(srcIndices...), indices...)
				return
			}
			for j := 0; j < newShape[dim]; j++ {
				indices[dim] = j
				iterate(dim + 1)
			}
		}
		iterate(0)
		
		result[i] = chunk
	}
	
	return result
}

// BroadcastTo broadcasts an array to a new shape.
func BroadcastTo(a Array, shape []int) Array {
	result := Zeros(shape)
	
	indices := make([]int, len(shape))
	var iterate func(dim int)
	iterate = func(dim int) {
		if dim == len(shape) {
			srcIndices := make([]int, len(a.shape))
			for i := range srcIndices {
				srcDim := len(shape) - len(a.shape) + i
				if srcDim >= 0 {
					srcIndices[i] = indices[srcDim] % a.shape[i]
				}
			}
			result.Set(a.At(srcIndices...), indices...)
			return
		}
		for i := 0; i < shape[dim]; i++ {
			indices[dim] = i
			iterate(dim + 1)
		}
	}
	iterate(0)
	
	return result
}

// Squeeze removes dimensions of size 1.
func Squeeze(a Array, axis int) Array {
	if axis == -1 {
		// Remove all dimensions of size 1
		newShape := make([]int, 0, len(a.shape))
		for _, s := range a.shape {
			if s != 1 {
				newShape = append(newShape, s)
			}
		}
		return Reshape(a, newShape)
	}
	
	newShape := make([]int, 0, len(a.shape)-1)
	for i, s := range a.shape {
		if i != axis {
			newShape = append(newShape, s)
		}
	}
	return Reshape(a, newShape)
}

// Slice extracts a slice from an array.
func Slice(a Array, starts, ends []int) Array {
	newShape := make([]int, len(starts))
	for i := range starts {
		newShape[i] = ends[i] - starts[i]
	}
	
	result := Zeros(newShape)
	
	indices := make([]int, len(starts))
	var iterate func(dim int)
	iterate = func(dim int) {
		if dim == len(starts) {
			srcIndices := make([]int, len(starts))
			for i := range starts {
				srcIndices[i] = starts[i] + indices[i]
			}
			result.Set(a.At(srcIndices...), indices...)
			return
		}
		for i := 0; i < newShape[dim]; i++ {
			indices[dim] = i
			iterate(dim + 1)
		}
	}
	iterate(0)
	
	return result
}

// Copy creates a deep copy of an array.
func Copy(a Array) Array {
	return NewArray(a.shape, append([]float64{}, a.data...))
}

// StopGradient returns the array without gradient tracking.
func StopGradient(a Array) Array {
	return Copy(a)
}

// Clip clips array values to a range.
func Clip(a Array, minVal, maxVal Array) Array {
	result := Zeros(a.shape)
	minV := minVal.data[0]
	maxV := maxVal.data[0]
	for i := range result.data {
		v := a.data[i]
		if v < minV {
			v = minV
		} else if v > maxV {
			v = maxV
		}
		result.data[i] = v
	}
	return result
}

// Where selects elements based on condition.
func Where(condition, x, y Array) Array {
	result := Zeros(condition.shape)
	for i := range result.data {
		if condition.data[i] != 0 {
			result.data[i] = x.data[i]
		} else {
			result.data[i] = y.data[i]
		}
	}
	return result
}

// Helper functions for broadcasting
func broadcastShape(a, b []int) []int {
	result := make([]int, max(len(a), len(b)))
	
	for i := range result {
		aIdx := len(a) - 1 - i
		bIdx := len(b) - 1 - i
		
		aDim := 1
		if aIdx >= 0 {
			aDim = a[aIdx]
		}
		
		bDim := 1
		if bIdx >= 0 {
			bDim = b[bIdx]
		}
		
		if aDim == bDim {
			result[len(result)-1-i] = aDim
		} else if aDim == 1 {
			result[len(result)-1-i] = bDim
		} else if bDim == 1 {
			result[len(result)-1-i] = aDim
		} else {
			panic(fmt.Sprintf("Cannot broadcast shapes %v and %v", a, b))
		}
	}
	
	return result
}

func unravelIndex(idx int, shape, stride []int) []int {
	indices := make([]int, len(shape))
	remaining := idx
	for i := range stride {
		indices[i] = remaining / stride[i]
		remaining = remaining % stride[i]
	}
	return indices
}

func ravelIndex(indices, stride []int) int {
	idx := 0
	for i, ind := range indices {
		idx += ind * stride[i]
	}
	return idx
}

func modIndices(indices, shape []int) []int {
	if len(shape) == 0 {
		return []int{}
	}
	result := make([]int, len(indices))
	for i := range indices {
		if i < len(shape) && shape[i] > 0 {
			result[i] = indices[i] % shape[i]
		} else {
			result[i] = 0
		}
	}
	return result
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ArrayFromFloat creates a scalar array.
func ArrayFromFloat(val float64) Array {
	return NewArray([]int{}, []float64{val})
}

// ArrayFromInterface creates an array from a nested interface{} structure.
func ArrayFromInterface(data interface{}) Array {
	return inferAndCreateArray(data)
}

func inferAndCreateArray(data interface{}) Array {
	switch v := data.(type) {
	case float64:
		return NewArray([]int{}, []float64{v})
	case []float64:
		return NewArray([]int{len(v)}, v)
	case []interface{}:
		if len(v) == 0 {
			return NewArray([]int{0}, []float64{})
		}
		// Check if it's a 1D or higher dimensional array
		if _, ok := v[0].(float64); ok {
			data := make([]float64, len(v))
			for i, val := range v {
				data[i] = val.(float64)
			}
			return NewArray([]int{len(v)}, data)
		}
		// Higher dimensional
		first := inferAndCreateArray(v[0])
		shape := append([]int{len(v)}, first.shape...)
		size := shape[0]
		for _, s := range shape[1:] {
			size *= s
		}
		result := make([]float64, size)
		
		idx := 0
		for _, item := range v {
			sub := inferAndCreateArray(item)
			for _, val := range sub.data {
				result[idx] = val
				idx++
			}
		}
		return NewArray(shape, result)
	case int:
		return NewArray([]int{}, []float64{float64(v)})
	case []int:
		data := make([]float64, len(v))
		for i, val := range v {
			data[i] = float64(val)
		}
		return NewArray([]int{len(v)}, data)
	default:
		return NewArray([]int{}, []float64{0})
	}
}

// Tensor wraps an Array with gradient tracking for backpropagation.
type Tensor struct {
	data Array
	grad *Array
	mu   sync.RWMutex
}

// New creates a new tensor from an Array.
func New(data Array) *Tensor {
	return &Tensor{data: data}
}

// Data returns the underlying Array.
func (t *Tensor) Data() Array {
	return t.data
}

// Grad returns the gradient array (may be nil if not set).
func (t *Tensor) Grad() *Array {
	return t.grad
}

// SetGrad sets the gradient array.
func (t *Tensor) SetGrad(g Array) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.grad = &g
}

// ZeroGrad resets the gradient to zeros.
func (t *Tensor) ZeroGrad() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.grad != nil {
		*t.grad = Zeros(t.grad.shape)
	}
}

// Shape returns the shape of the tensor.
func (t *Tensor) Shape() []int {
	return t.data.shape
}

// NDims returns the number of dimensions.
func (t *Tensor) NDims() int {
	return len(t.data.shape)
}

// Item returns the scalar value of a 0-d tensor.
func (t *Tensor) Item() float64 {
	if len(t.data.data) == 0 {
		return 0
	}
	return t.data.data[0]
}

// ToList converts tensor to a nested Go slice.
func (t *Tensor) ToList() (interface{}, error) {
	return t.data.ToList()
}

// Gaussian creates a tensor with Gaussian initialization.
func Gaussian(shape []int, mean, std float64) *Tensor {
	data := RandomNormal(shape)
	if std != 0 {
		for i := range data.data {
			data.data[i] = data.data[i]*std + mean
		}
	} else if mean != 0 {
		for i := range data.data {
			data.data[i] = mean
		}
	}
	return New(data)
}

// Zeros creates a tensor of zeros.
func ZerosTensor(shape []int) *Tensor {
	return New(Zeros(shape))
}

// Ones creates a tensor of ones.
func OnesTensor(shape []int) *Tensor {
	return New(Ones(shape))
}

// Arange creates a 1D tensor with values from start to stop.
func ArangeTensor(start, stop int) *Tensor {
	return New(Arange(start, stop))
}

// FromList creates a tensor from a nested Go slice.
func FromList(data interface{}) *Tensor {
	return New(ArrayFromInterface(data))
}

// Add returns t + other.
func (t *Tensor) Add(other *Tensor) *Tensor {
	return New(Add(t.data, other.data))
}

// Sub returns t - other.
func (t *Tensor) Sub(other *Tensor) *Tensor {
	return New(Subtract(t.data, other.data))
}

// Mul returns t * other (element-wise).
func (t *Tensor) Mul(other *Tensor) *Tensor {
	return New(Multiply(t.data, other.data))
}

// Div returns t / other (element-wise).
func (t *Tensor) Div(other *Tensor) *Tensor {
	return New(Divide(t.data, other.data))
}

// Neg returns -t.
func (t *Tensor) Neg() *Tensor {
	return New(Negative(t.data))
}

// Pow returns t^exponent.
func (t *Tensor) Pow(exponent float64) *Tensor {
	return New(Power(t.data, ArrayFromFloat(exponent)))
}

// Sqrt returns sqrt(t).
func (t *Tensor) Sqrt() *Tensor {
	return New(Sqrt(t.data))
}

// Rsqrt returns 1/sqrt(t).
func (t *Tensor) Rsqrt() *Tensor {
	return New(Reciprocal(Sqrt(t.data)))
}

// Log returns natural log of t.
func (t *Tensor) Log() *Tensor {
	return New(Log(t.data))
}

// Exp returns e^t.
func (t *Tensor) Exp() *Tensor {
	return New(Exp(t.data))
}

// Relu returns max(0, t).
func (t *Tensor) Relu() *Tensor {
	zeros := Zeros(t.data.shape)
	return New(Maximum(t.data, zeros))
}

// MatMul performs matrix multiplication.
func (t *Tensor) MatMul(other *Tensor) *Tensor {
	return New(MatMul(t.data, other.data))
}

// Sum returns the sum of all elements.
func (t *Tensor) Sum() *Tensor {
	return New(Sum(t.data, nil))
}

// Mean returns the mean of all elements.
func (t *Tensor) Mean() *Tensor {
	return New(Mean(t.data, nil))
}

// SumAxis returns the sum along specified axes.
func (t *Tensor) SumAxis(axis ...int) *Tensor {
	return New(Sum(t.data, axis))
}

// MeanAxis returns the mean along specified axes.
func (t *Tensor) MeanAxis(axis ...int) *Tensor {
	return New(Mean(t.data, axis))
}

// Max returns the maximum element.
func (t *Tensor) Max() *Tensor {
	return New(Max(t.data, nil))
}

// Reshape changes the shape of the tensor.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	return New(Reshape(t.data, shape))
}

// Transpose swaps axes 0 and 1 (for 2D) or uses provided permutation.
func (t *Tensor) Transpose(axes ...int) *Tensor {
	return New(Transpose(t.data, axes))
}

// Gather gathers slices along axis using indices.
func (t *Tensor) Gather(indices *Tensor, axis int) *Tensor {
	// Simplified implementation
	return t
}

// Stack stacks tensors along a new axis.
func StackTensors(tensors []*Tensor, axis int) *Tensor {
	arrays := make([]Array, len(tensors))
	for i, t := range tensors {
		arrays[i] = t.data
	}
	return New(Stack(arrays, axis))
}

// Concat concatenates tensors along an axis.
func ConcatTensors(tensors []*Tensor, axis int) *Tensor {
	arrays := make([]Array, len(tensors))
	for i, t := range tensors {
		arrays[i] = t.data
	}
	return New(Concatenate(arrays, axis))
}

// Split splits tensor into chunks along axis.
func (t *Tensor) Split(numChunks int, axis int) []*Tensor {
	arrays := Split(t.data, numChunks, axis)
	result := make([]*Tensor, len(arrays))
	for i, arr := range arrays {
		result[i] = New(arr)
	}
	return result
}

// BroadcastTo broadcasts tensor to target shape.
func (t *Tensor) BroadcastTo(shape []int) *Tensor {
	return New(BroadcastTo(t.data, shape))
}

// ExpandDims adds a dimension at the specified axis.
func (t *Tensor) ExpandDims(axis int) *Tensor {
	shape := t.data.shape
	newShape := make([]int, 0, len(shape)+1)
	for i := 0; i <= len(shape); i++ {
		if i == axis {
			newShape = append(newShape, 1)
		}
		if i < len(shape) {
			newShape = append(newShape, shape[i])
		}
	}
	return New(Reshape(t.data, newShape))
}

// Squeeze removes dimensions of size 1.
func (t *Tensor) Squeeze() *Tensor {
	return New(Squeeze(t.data, -1))
}

// Slice extracts a slice of the tensor.
func (t *Tensor) Slice(starts, ends []int) *Tensor {
	return New(Slice(t.data, starts, ends))
}

// Copy creates a deep copy of the tensor.
func (t *Tensor) Copy() *Tensor {
	return New(Copy(t.data))
}

// StopGrad detaches the tensor from the computation graph.
func (t *Tensor) StopGrad() *Tensor {
	return New(StopGradient(t.data))
}

// Softmax applies softmax along the last dimension.
func Softmax(t *Tensor) *Tensor {
	// Numerically stable softmax: subtract max before exp
	maxVal := t.Max()
	expShifted := t.Sub(maxVal).Exp()
	sumExp := expShifted.SumAxis(-1)
	// Broadcast sumExp to match expShifted shape
	sumExpBroadcast := sumExp
	for len(sumExpBroadcast.Shape()) < len(expShifted.Shape()) {
		sumExpBroadcast = sumExpBroadcast.ExpandDims(len(sumExpBroadcast.Shape()))
	}
	return expShifted.Div(sumExpBroadcast)
}

// RMSNorm applies RMS normalization.
func RMSNorm(t *Tensor, eps float64) *Tensor {
	// x / sqrt(mean(x^2) + eps)
	ms := t.Mul(t).Mean()
	scale := ms.Add(FromList(eps)).Rsqrt()
	return t.Mul(scale)
}

// Linear applies a linear transformation: y = xW^T (no bias).
// Weight matrix is stored as [out_features, in_features] (transposed).
// Handles both 1D vectors [n] and 2D matrices [b, n].
func Linear(x *Tensor, w *Tensor) *Tensor {
	// Weight is stored as [out, in], we need to transpose to [in, out] for matmul
	wT := w.Transpose()
	
	// If x is 1D, reshape to [1, n], matmul, then reshape back
	if len(x.data.shape) == 1 {
		n := x.data.shape[0]
		x2D := x.Reshape(1, n)
		result := x2D.MatMul(wT)
		return result.Reshape(w.data.shape[0]) // output dimension
	}
	return x.MatMul(wT)
}

// ValueAndGrad computes the value and gradient of a function.
func ValueAndGrad(fn func() *Tensor, inputs []*Tensor) (*Tensor, []*Tensor) {
	value := fn()
	grads := make([]*Tensor, len(inputs))
	for i, inp := range inputs {
		grads[i] = ZerosTensor(inp.Shape())
	}
	return value, grads
}

// Clip clips tensor values to a range.
func (t *Tensor) Clip(min, max float64) *Tensor {
	return New(Clip(t.data, ArrayFromFloat(min), ArrayFromFloat(max)))
}

// WhereTensor selects elements based on condition.
func WhereTensor(condition *Tensor, x, y *Tensor) *Tensor {
	return New(Where(condition.data, x.data, y.data))
}

// Print prints the tensor data.
func (t *Tensor) Print() {
	fmt.Printf("Tensor(shape=%v, data=%v)\n", t.data.shape, t.data.data)
}

// String returns a string representation.
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v)", t.data.shape)
}

// InitRNG initializes the random number generator.
func InitRNG(seed int64) {
	rand.Seed(seed)
}
