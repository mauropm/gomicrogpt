//go:build mlx && cgo
// +build mlx,cgo

// Package mlx provides MLX build mode for GPU acceleration status.
// This file is only compiled when building with -tags=mlx and CGO enabled.
//
// Note: MLX (https://github.com/ml-explore/mlx) primarily provides Python bindings.
// The C API is not distributed as a standalone library via Homebrew.
// 
// This build enables MLX mode status reporting. The actual tensor operations
// use the optimized pure Go implementation in mlx_fallback_mlx.go which provides
// equivalent functionality.
//
// For true GPU acceleration via Metal, future work would include:
// - CGO wrappers to MLX Python C API
// - Direct Metal shader implementations
// - Integration with MLX Swift bindings

package mlx

/*
#cgo darwin CFLAGS: -I/opt/homebrew/include
#cgo darwin LDFLAGS: -L/opt/homebrew/lib -lmlx -framework Foundation -framework Metal -framework Accelerate

#include <stdlib.h>

// MLX Python library doesn't expose a standalone C API.
// When MLX C API becomes available, include headers here:
// #include <mlx/c/array.h>
*/
import "C"

// This file enables MLX build mode.
// UseMLX, IsUsingMLX, and GetBackendInfo are defined in mlx_fallback_mlx.go
// The tensor operations use the pure Go implementation which is fully functional.

