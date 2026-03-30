//go:build mlx && cgo
// +build mlx,cgo

package mlx

import (
	"fmt"
	"runtime"
)

// UseMLX indicates whether MLX acceleration is enabled.
// This is set at compile time via build tags.
const UseMLX = true

// IsUsingMLX returns true if MLX acceleration is active.
func IsUsingMLX() bool {
	return UseMLX
}

// GetBackendInfo returns information about the current backend.
func GetBackendInfo() string {
	if UseMLX {
		return fmt.Sprintf("MLX (Metal GPU) - %s", getChipInfo())
	}
	return fmt.Sprintf("Pure Go (CPU - %s)", runtime.GOARCH)
}

// getChipInfo returns information about the Apple Silicon chip.
func getChipInfo() string {
	// This would query system info in a full implementation
	// For now, return generic info
	return "Apple Silicon"
}
