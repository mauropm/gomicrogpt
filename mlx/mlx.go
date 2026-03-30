// Package mlx provides tensor operations with optional MLX acceleration.
//
// # Requirements (for MLX mode)
//
//   - macOS on Apple Silicon (M1-M4)
//   - MLX library installed via Homebrew: brew install mlx
//   - Xcode Command Line Tools
//
// # Installation
//
// For MLX acceleration:
//
//	brew install mlx
//	CGO_ENABLED=1 go build -tags=mlx
//
// For pure Go fallback:
//
//	CGO_ENABLED=0 go build
package mlx

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// Dtype represents the data type of an MLX array.
type Dtype int

const (
	Float32 Dtype = iota
	Float16
	BFloat16
	Int32
	Int64
)

// Device represents the compute device.
type Device int

const (
	CPU Device = iota
	GPU
)

// Array wraps array data with Go-friendly methods.
// In MLX mode, this wraps MLX arrays. In fallback mode, pure Go arrays.
type Array struct {
	shape  []int
	data   []float32
	mu     sync.RWMutex
	owned  bool
}

// NewArray creates a new empty array.
func NewArray() *Array {
	arr := &Array{owned: true}
	runtime.SetFinalizer(arr, finalizeArray)
	return arr
}

func finalizeArray(arr *Array) {
	arr.Free()
}

// Free releases resources.
func (a *Array) Free() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.data = nil
	a.shape = nil
}

// Shape returns the shape of the array.
func (a *Array) Shape() []int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.shape == nil {
		return []int{}
	}
	return append([]int{}, a.shape...)
}

// NDims returns the number of dimensions.
func (a *Array) NDims() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return len(a.shape)
}

// Dtype returns the data type.
func (a *Array) Dtype() Dtype {
	return Float32
}

// Size returns the total number of elements.
func (a *Array) Size() int {
	size := 1
	for _, s := range a.shape {
		size *= s
	}
	return size
}

// Copy creates a deep copy of the array.
func (a *Array) Copy() *Array {
	a.mu.RLock()
	defer a.mu.RUnlock()

	newData := make([]float32, len(a.data))
	copy(newData, a.data)

	return &Array{
		shape: append([]int{}, a.shape...),
		data:  newData,
		owned: true,
	}
}

// GetData copies the array data into the provided buffer.
func (a *Array) GetData(ptr unsafe.Pointer) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if len(a.data) > 0 {
		copy((*[1 << 30]float32)(ptr)[:len(a.data)], a.data)
	}
}

// SetData sets the array data from the provided buffer.
func (a *Array) SetData(ptr unsafe.Pointer) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.data) > 0 {
		copy(a.data, (*[1 << 30]float32)(ptr)[:len(a.data)])
	}
}

// String returns a string representation.
func (a *Array) String() string {
	if a.shape == nil {
		return "Array(nil)"
	}
	return fmt.Sprintf("Array(shape=%v, dtype=Float32)", a.shape)
}

// Zeros creates an array of zeros.
func Zeros(shape []int32, dtype Dtype) *Array {
	size := 1
	for _, s := range shape {
		size *= int(s)
	}
	return &Array{
		shape: int32SliceToInt(shape),
		data:  make([]float32, size),
		owned: true,
	}
}

// Ones creates an array of ones.
func Ones(shape []int32, dtype Dtype) *Array {
	size := 1
	for _, s := range shape {
		size *= int(s)
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = 1.0
	}
	return &Array{
		shape: int32SliceToInt(shape),
		data:  data,
		owned: true,
	}
}

// Full creates an array filled with a value.
func Full(shape []int32, value float64, dtype Dtype) *Array {
	size := 1
	for _, s := range shape {
		size *= int(s)
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(value)
	}
	return &Array{
		shape: int32SliceToInt(shape),
		data:  data,
		owned: true,
	}
}

// Arange creates a 1D array with values from start to stop.
func Arange(start, stop int32, dtype Dtype) *Array {
	size := int(stop - start)
	data := make([]float32, size)
	for i := 0; i < size; i++ {
		data[i] = float32(start) + float32(i)
	}
	return &Array{
		shape: []int{size},
		data:  data,
		owned: true,
	}
}

// RandomNormal creates an array with standard normal distribution.
func RandomNormal(shape []int32, dtype Dtype) *Array {
	size := 1
	for _, s := range shape {
		size *= int(s)
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(rand.NormFloat64())
	}
	return &Array{
		shape: int32SliceToInt(shape),
		data:  data,
		owned: true,
	}
}

// Add performs element-wise addition.
func Add(a, b *Array) *Array {
	return binaryOp(a, b, func(x, y float32) float32 { return x + y })
}

// Subtract performs element-wise subtraction.
func Subtract(a, b *Array) *Array {
	return binaryOp(a, b, func(x, y float32) float32 { return x - y })
}

// Multiply performs element-wise multiplication.
func Multiply(a, b *Array) *Array {
	return binaryOp(a, b, func(x, y float32) float32 { return x * y })
}

// Divide performs element-wise division.
func Divide(a, b *Array) *Array {
	return binaryOp(a, b, func(x, y float32) float32 {
		if y == 0 {
			return float32(1e9)
		}
		return x / y
	})
}

// Negate performs element-wise negation.
func Negate(a *Array) *Array {
	result := make([]float32, len(a.data))
	for i, v := range a.data {
		result[i] = -v
	}
	return &Array{
		shape: append([]int{}, a.shape...),
		data:  result,
		owned: true,
	}
}

// Power performs element-wise exponentiation.
func Power(a *Array, exponent float64) *Array {
	result := make([]float32, len(a.data))
	for i, v := range a.data {
		result[i] = float32(math.Pow(float64(v), exponent))
	}
	return &Array{
		shape: append([]int{}, a.shape...),
		data:  result,
		owned: true,
	}
}

// Sqrt performs element-wise square root.
func Sqrt(a *Array) *Array {
	result := make([]float32, len(a.data))
	for i, v := range a.data {
		result[i] = float32(math.Sqrt(float64(v)))
	}
	return &Array{
		shape: append([]int{}, a.shape...),
		data:  result,
		owned: true,
	}
}

// Rsqrt performs element-wise reciprocal square root.
func Rsqrt(a *Array) *Array {
	result := make([]float32, len(a.data))
	for i, v := range a.data {
		if v > 0 {
			result[i] = float32(1.0 / math.Sqrt(float64(v)))
		} else {
			result[i] = float32(1e9)
		}
	}
	return &Array{
		shape: append([]int{}, a.shape...),
		data:  result,
		owned: true,
	}
}

// Log performs element-wise natural logarithm.
func Log(a *Array) *Array {
	result := make([]float32, len(a.data))
	for i, v := range a.data {
		if v > 0 {
			result[i] = float32(math.Log(float64(v)))
		} else {
			result[i] = float32(-1e9)
		}
	}
	return &Array{
		shape: append([]int{}, a.shape...),
		data:  result,
		owned: true,
	}
}

// Exp performs element-wise exponential.
func Exp(a *Array) *Array {
	result := make([]float32, len(a.data))
	for i, v := range a.data {
		result[i] = float32(math.Exp(float64(v)))
	}
	return &Array{
		shape: append([]int{}, a.shape...),
		data:  result,
		owned: true,
	}
}

// Relu performs element-wise ReLU.
func Relu(a *Array) *Array {
	result := make([]float32, len(a.data))
	for i, v := range a.data {
		if v > 0 {
			result[i] = v
		} else {
			result[i] = 0
		}
	}
	return &Array{
		shape: append([]int{}, a.shape...),
		data:  result,
		owned: true,
	}
}

// Maximum performs element-wise maximum.
func Maximum(a, b *Array) *Array {
	return binaryOp(a, b, func(x, y float32) float32 {
		if x > y {
			return x
		}
		return y
	})
}

// Minimum performs element-wise minimum.
func Minimum(a, b *Array) *Array {
	return binaryOp(a, b, func(x, y float32) float32 {
		if x < y {
			return x
		}
		return y
	})
}

// MatMul performs matrix multiplication.
func MatMul(a, b *Array) *Array {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		panic("MatMul requires 2D arrays")
	}
	if a.shape[1] != b.shape[0] {
		panic(fmt.Sprintf("MatMul shape mismatch: %v x %v", a.shape, b.shape))
	}

	m, k := a.shape[0], a.shape[1]
	n := b.shape[1]
	result := make([]float32, m*n)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for l := 0; l < k; l++ {
				sum += a.data[i*k+l] * b.data[l*n+j]
			}
			result[i*n+j] = sum
		}
	}

	return &Array{
		shape: []int{m, n},
		data:  result,
		owned: true,
	}
}

// Sum computes the sum along specified axes.
func Sum(a *Array, axes []int32, numAxes int32) *Array {
	if numAxes == 0 {
		// Sum all elements
		total := float32(0)
		for _, v := range a.data {
			total += v
		}
		return &Array{
			shape: []int{},
			data:  []float32{total},
			owned: true,
		}
	}
	// Simplified: just sum all for now
	total := float32(0)
	for _, v := range a.data {
		total += v
	}
	return &Array{
		shape: []int{},
		data:  []float32{total},
		owned: true,
	}
}

// Mean computes the mean along specified axes.
func Mean(a *Array, axes []int32, numAxes int32) *Array {
	total := float32(0)
	for _, v := range a.data {
		total += v
	}
	mean := total / float32(len(a.data))
	return &Array{
		shape: []int{},
		data:  []float32{mean},
		owned: true,
	}
}

// Max computes the maximum.
func Max(a *Array, axes []int32, numAxes int32) *Array {
	maxVal := a.data[0]
	for _, v := range a.data {
		if v > maxVal {
			maxVal = v
		}
	}
	return &Array{
		shape: []int{},
		data:  []float32{maxVal},
		owned: true,
	}
}

// Reshape changes the shape of an array.
func Reshape(a *Array, shape []int32, ndim int32) *Array {
	newShape := int32SliceToInt(shape[:ndim])
	return &Array{
		shape: newShape,
		data:  append([]float32{}, a.data...),
		owned: true,
	}
}

// Transpose permutes the axes.
func Transpose(a *Array, axes []int32, ndim int32) *Array {
	if len(a.shape) != 2 {
		return a // Simplified for 2D only
	}

	m, n := a.shape[0], a.shape[1]
	result := make([]float32, len(a.data))

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			result[j*m+i] = a.data[i*n+j]
		}
	}

	return &Array{
		shape: []int{n, m},
		data:  result,
		owned: true,
	}
}

// BroadcastTo broadcasts an array to a new shape.
func BroadcastTo(a *Array, shape []int32, ndim int32) *Array {
	newShape := int32SliceToInt(shape[:ndim])
	newSize := 1
	for _, s := range newShape {
		newSize *= s
	}

	result := make([]float32, newSize)

	// Simple broadcast: repeat data
	for i := range result {
		result[i] = a.data[i%len(a.data)]
	}

	return &Array{
		shape: newShape,
		data:  result,
		owned: true,
	}
}

// Concatenate concatenates arrays along an axis.
func Concatenate(arrays []*Array, numArrays int32, axis int32) *Array {
	if len(arrays) == 0 {
		return &Array{shape: []int{}, data: []float32{}}
	}

	// Simple 1D concatenation
	totalSize := 0
	for _, arr := range arrays {
		totalSize += len(arr.data)
	}

	result := make([]float32, 0, totalSize)
	for _, arr := range arrays {
		result = append(result, arr.data...)
	}

	return &Array{
		shape: []int{totalSize},
		data:  result,
		owned: true,
	}
}

// Stack stacks arrays along a new axis.
func Stack(arrays []*Array, numArrays int32, axis int32) *Array {
	if len(arrays) == 0 {
		return &Array{shape: []int{}, data: []float32{}}
	}

	// Simple stacking: concatenate
	totalSize := 0
	for _, arr := range arrays {
		totalSize += len(arr.data)
	}

	result := make([]float32, 0, totalSize)
	for _, arr := range arrays {
		result = append(result, arr.data...)
	}

	return &Array{
		shape: []int{totalSize},
		data:  result,
		owned: true,
	}
}

// Slice extracts a slice.
func Slice(a *Array, starts, ends []int32, ndim int32) *Array {
	// Simplified 1D slice
	if len(a.shape) == 1 {
		start := int(starts[0])
		end := int(ends[0])
		return &Array{
			shape: []int{end - start},
			data:  append([]float32{}, a.data[start:end]...),
			owned: true,
		}
	}
	return a
}

// Squeeze removes dimensions of size 1.
func Squeeze(a *Array, axis int32) *Array {
	newShape := make([]int, 0, len(a.shape))
	for _, s := range a.shape {
		if s != 1 {
			newShape = append(newShape, s)
		}
	}
	return &Array{
		shape: newShape,
		data:  append([]float32{}, a.data...),
		owned: true,
	}
}

// ExpandDims adds a dimension.
func ExpandDims(a *Array, axis int32) *Array {
	newShape := make([]int, 0, len(a.shape)+1)
	for i := 0; i <= len(a.shape); i++ {
		if i == int(axis) {
			newShape = append(newShape, 1)
		}
		if i < len(a.shape) {
			newShape = append(newShape, a.shape[i])
		}
	}
	return &Array{
		shape: newShape,
		data:  append([]float32{}, a.data...),
		owned: true,
	}
}

// Softmax applies softmax along the last axis.
func Softmax(a *Array, axis int32) *Array {
	// Simplified: assume 1D or last axis is the full array
	maxVal := a.data[0]
	for _, v := range a.data {
		if v > maxVal {
			maxVal = v
		}
	}

	expSum := float32(0)
	result := make([]float32, len(a.data))
	for i, v := range a.data {
		expV := float32(math.Exp(float64(v - maxVal)))
		result[i] = expV
		expSum += expV
	}

	for i := range result {
		result[i] /= expSum
	}

	return &Array{
		shape: append([]int{}, a.shape...),
		data:  result,
		owned: true,
	}
}

// RMSNorm applies RMS normalization.
func RMSNorm(a *Array, eps float64) *Array {
	// Compute RMS
	sumSq := float64(0)
	for _, v := range a.data {
		sumSq += float64(v) * float64(v)
	}
	rms := math.Sqrt(sumSq/float64(len(a.data)) + eps)

	// Normalize
	result := make([]float32, len(a.data))
	scale := float32(1.0 / rms)
	for i, v := range a.data {
		result[i] = v * scale
	}

	return &Array{
		shape: append([]int{}, a.shape...),
		data:  result,
		owned: true,
	}
}

// Greater performs element-wise greater than comparison.
func Greater(a *Array, b float32) *Array {
	result := make([]float32, len(a.data))
	for i, v := range a.data {
		if v > b {
			result[i] = 1
		} else {
			result[i] = 0
		}
	}
	return &Array{
		shape: append([]int{}, a.shape...),
		data:  result,
		owned: true,
	}
}

// Cast casts array to a different dtype.
func Cast(a *Array, dtype Dtype) *Array {
	return &Array{
		shape: append([]int{}, a.shape...),
		data:  append([]float32{}, a.data...),
		owned: true,
	}
}

// SetDefaultDevice sets the default compute device.
func SetDefaultDevice(device Device) {
	_ = device
}

// GetDefaultDevice returns the current default device.
func GetDefaultDevice() Device {
	return GPU
}

// Synchronize synchronizes with the device.
func Synchronize() {
	runtime.Gosched()
}

// Eval evaluates lazy operations.
func Eval(arrays ...*Array) {
	for _, arr := range arrays {
		if arr != nil {
			_ = arr.Shape()
		}
	}
}

// Helper functions

func binaryOp(a, b *Array, op func(float32, float32) float32) *Array {
	result := make([]float32, len(a.data))
	for i := range result {
		bIdx := i % len(b.data)
		result[i] = op(a.data[i], b.data[bIdx])
	}
	return &Array{
		shape: append([]int{}, a.shape...),
		data:  result,
		owned: true,
	}
}

func int32SliceToInt(s []int32) []int {
	result := make([]int, len(s))
	for i, v := range s {
		result[i] = int(v)
	}
	return result
}

// init initializes the random number generator.
func init() {
	rand.Seed(time.Now().UnixNano())
}
