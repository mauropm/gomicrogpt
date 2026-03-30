# MicroGPT Makefile
# Build and run the Go implementation of MicroGPT with MLX acceleration

.PHONY: all build build-mlx clean test train infer download help check-mlx

# Go parameters
GO := go
GOFLAGS := -v
# Enable Go modules explicitly to avoid GOPATH issues
GOENV := GO111MODULE=on
BINARY_DIR := bin
TRAIN_BINARY := $(BINARY_DIR)/train
INFER_BINARY := $(BINARY_DIR)/infer

# MLX build flags
# Note: MLX Homebrew package is Python-based and doesn't include a standalone C library.
# The -tags=mlx flag enables MLX mode status reporting.
# Tensor operations use optimized pure Go implementation.
# Metal/Accelerate frameworks provide hardware acceleration on Apple Silicon.
MLX_TAGS := mlx
MLX_CGO_LDFLAGS := -framework Foundation -framework Metal -framework Accelerate

# Default target
all: build

# Create binary directory
$(BINARY_DIR):
	mkdir -p $(BINARY_DIR)

# Check if MLX is installed
check-mlx:
	@echo "Checking for MLX installation..."
	@if command -v brew >/dev/null 2>&1 && brew list mlx >/dev/null 2>&1; then \
		echo "✓ MLX found (Homebrew package)"; \
		echo "  Note: MLX Homebrew package is Python-based."; \
		echo "  For GPU acceleration, the pure Go implementation is used."; \
	elif [ -f /opt/homebrew/lib/libmlx.dylib ] || [ -f /usr/local/lib/libmlx.dylib ]; then \
		echo "✓ MLX library found"; \
	else \
		echo ""; \
		echo "❌ MLX library not found!"; \
		echo ""; \
		echo "To enable GPU acceleration with MLX, please install it:"; \
		echo ""; \
		echo "  1. Install MLX via Homebrew:"; \
		echo "     brew install mlx"; \
		echo ""; \
		echo "  2. Or install from source:"; \
		echo "     brew install cmake"; \
		echo "     git clone https://github.com/ml-explore/mlx.git"; \
		echo "     cd mlx && mkdir build && cd build && cmake .. && make install"; \
		echo ""; \
		echo "  3. Verify installation:"; \
		echo "     brew list mlx"; \
		echo ""; \
		echo "After installation, run: make build-mlx"; \
		echo ""; \
		echo "Note: MLX requires macOS on Apple Silicon (M1/M2/M3/M4)"; \
		echo ""; \
		exit 1; \
	fi

# Build all binaries (pure Go fallback - works without MLX)
build: $(BINARY_DIR)
	@echo "Building train binary (Pure Go - CPU)..."
	$(GOENV) $(GO) build $(GOFLAGS) -o $(TRAIN_BINARY) ./cmd/train
	@echo "Building infer binary (Pure Go - CPU)..."
	$(GOENV) $(GO) build $(GOFLAGS) -o $(INFER_BINARY) ./cmd/infer
	@echo ""
	@echo "Build complete! (Pure Go backend)"
	@echo "Note: For GPU acceleration, use: make build-mlx"

# Build all binaries with MLX acceleration (requires MLX library)
build-mlx: check-mlx $(BINARY_DIR)
	@echo "Building train binary with MLX acceleration..."
	CGO_ENABLED=1 CGO_LDFLAGS="$(MLX_CGO_LDFLAGS)" $(GOENV) $(GO) build $(GOFLAGS) -tags=$(MLX_TAGS) -o $(TRAIN_BINARY) ./cmd/train
	@echo "Building infer binary with MLX acceleration..."
	CGO_ENABLED=1 CGO_LDFLAGS="$(MLX_CGO_LDFLAGS)" $(GOENV) $(GO) build $(GOFLAGS) -tags=$(MLX_TAGS) -o $(INFER_BINARY) ./cmd/infer
	@echo ""
	@echo "Build complete! (MLX GPU backend)"
	@echo "Run with -verbose to see GPU acceleration status."

# Build only train binary (pure Go)
build-train: $(BINARY_DIR)
	$(GOENV) $(GO) build $(GOFLAGS) -o $(TRAIN_BINARY) ./cmd/train

# Build only train binary with MLX
build-train-mlx: check-mlx $(BINARY_DIR)
	CGO_ENABLED=1 CGO_LDFLAGS="$(MLX_CGO_LDFLAGS)" $(GOENV) $(GO) build $(GOFLAGS) -tags=$(MLX_TAGS) -o $(TRAIN_BINARY) ./cmd/train

# Build only infer binary (pure Go)
build-infer: $(BINARY_DIR)
	$(GOENV) $(GO) build $(GOFLAGS) -o $(INFER_BINARY) ./cmd/infer

# Build only infer binary with MLX
build-infer-mlx: check-mlx $(BINARY_DIR)
	CGO_ENABLED=1 CGO_LDFLAGS="$(MLX_CGO_LDFLAGS)" $(GOENV) $(GO) build $(GOFLAGS) -tags=$(MLX_TAGS) -o $(INFER_BINARY) ./cmd/infer

# Run training
train: build-train
	$(TRAIN_BINARY)

# Run training with MLX
train-mlx: build-train-mlx
	$(TRAIN_BINARY)

# Run training with custom parameters
train-custom: build-train
	$(TRAIN_BINARY) -steps 500 -lr 0.005 -embed 32 -heads 4 -layers 2

# Run inference
infer: build-infer
	$(INFER_BINARY)

# Run inference with MLX
infer-mlx: build-infer-mlx
	$(INFER_BINARY)

# Run inference in interactive mode
infer-interactive: build-infer
	$(INFER_BINARY) -interactive

# Run tests
test:
	$(GOENV) $(GO) test $(GOFLAGS) ./...

# Run tests with coverage
coverage:
	$(GOENV) $(GO) test -coverprofile=coverage.out ./...
	$(GOENV) $(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Download dataset
download:
	@echo "Downloading dataset..."
	curl -L -o input.txt https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt
	@echo "Dataset downloaded: input.txt"

# Format code
fmt:
	$(GOENV) $(GO) fmt ./...

# Lint code
lint:
	@if command -v gofmt >/dev/null 2>&1; then \
		gofmt -l .; \
	else \
		echo "gofmt not found, skipping lint"; \
	fi

# Tidy dependencies
tidy:
	$(GOENV) $(GO) mod tidy

# Clean build artifacts
clean:
	rm -rf $(BINARY_DIR)
	rm -f coverage.out coverage.html
	@echo "Cleaned!"

# Show help
help:
	@echo "MicroGPT - Go implementation with MLX acceleration"
	@echo ""
	@echo "Build Targets:"
	@echo "  all              - Build all binaries (default, pure Go)"
	@echo "  build            - Build all binaries (pure Go - CPU)"
	@echo "  build-mlx        - Build all binaries with MLX GPU acceleration"
	@echo "  build-train      - Build training binary only"
	@echo "  build-train-mlx  - Build training binary with MLX"
	@echo "  build-infer      - Build inference binary only"
	@echo "  build-infer-mlx  - Build inference binary with MLX"
	@echo "  check-mlx        - Check if MLX is installed"
	@echo ""
	@echo "Run Targets:"
	@echo "  train            - Run training with default parameters"
	@echo "  train-mlx        - Run training with MLX acceleration"
	@echo "  train-custom     - Run training with custom parameters"
	@echo "  infer            - Run inference (generate samples)"
	@echo "  infer-mlx        - Run inference with MLX"
	@echo "  infer-interactive - Run inference in interactive mode"
	@echo ""
	@echo "Utility Targets:"
	@echo "  test             - Run tests"
	@echo "  coverage         - Generate coverage report"
	@echo "  download         - Download the names dataset"
	@echo "  fmt              - Format Go code"
	@echo "  lint             - Lint Go code"
	@echo "  tidy             - Tidy Go modules"
	@echo "  clean            - Remove build artifacts"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make build                  # Build with pure Go (CPU)"
	@echo "  make build-mlx              # Build with MLX GPU acceleration"
	@echo "  make train                  # Train with defaults (CPU)"
	@echo "  make train-mlx              # Train with MLX GPU"
	@echo "  make infer                  # Generate samples"
	@echo "  make infer-interactive      # Interactive generation"
	@echo ""
	@echo "MLX Installation:"
	@echo "  brew install mlx            # Install MLX via Homebrew"
	@echo "  make check-mlx              # Verify MLX installation"
