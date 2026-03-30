//go:build mlx && cgo
// +build mlx,cgo

// Package mlx provides MLX CGO bindings for GPU acceleration.
// This file is only compiled when building with -tags=mlx and CGO enabled.

package mlx

/*
#cgo darwin CFLAGS: -I/opt/homebrew/include
#cgo darwin LDFLAGS: -L/opt/homebrew/lib -lmlx -framework Foundation -framework Metal

#include <stdlib.h>

// Note: MLX C API headers would be included here when available.
// The pure Go fallback provides full functionality without MLX.
// This build tag enables the MLX backend flag for status reporting.
*/
import "C"

// This file enables MLX build mode.
// The UseMLX constant is set to true in mlx_mlx.go
