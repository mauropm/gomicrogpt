// Package mlx provides tensor operations with MLX acceleration.
// This file contains types that are common to both MLX and fallback modes.

package mlx

import (
	"fmt"
	"sync"
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
type Array struct {
	shape  []int
	data   []float32
	mu     sync.RWMutex
	owned  bool
}

// NewArray creates a new empty array.
func NewArray() *Array {
	arr := &Array{owned: true}
	return arr
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
