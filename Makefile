# MicroGPT Makefile
# Build and run the Go implementation of MicroGPT with MLX acceleration

.PHONY: all build clean test train infer download help

# Go parameters
GO := go
GOFLAGS := -v
# Enable Go modules explicitly to avoid GOPATH issues
GOENV := GO111MODULE=on
BINARY_DIR := bin
TRAIN_BINARY := $(BINARY_DIR)/train
INFER_BINARY := $(BINARY_DIR)/infer

# Default target
all: build

# Create binary directory
$(BINARY_DIR):
	mkdir -p $(BINARY_DIR)

# Build all binaries
build: $(BINARY_DIR)
	@echo "Building train binary..."
	$(GOENV) $(GO) build $(GOFLAGS) -o $(TRAIN_BINARY) ./cmd/train
	@echo "Building infer binary..."
	$(GOENV) $(GO) build $(GOFLAGS) -o $(INFER_BINARY) ./cmd/infer
	@echo "Build complete!"

# Build only train binary
build-train: $(BINARY_DIR)
	$(GOENV) $(GO) build $(GOFLAGS) -o $(TRAIN_BINARY) ./cmd/train

# Build only infer binary
build-infer: $(BINARY_DIR)
	$(GOENV) $(GO) build $(GOFLAGS) -o $(INFER_BINARY) ./cmd/infer

# Run training
train: build-train
	$(TRAIN_BINARY)

# Run training with custom parameters
train-custom: build-train
	$(TRAIN_BINARY) -steps 500 -lr 0.005 -embed 32 -heads 4 -layers 2

# Run inference
infer: build-infer
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
	@echo "Targets:"
	@echo "  all              - Build all binaries (default)"
	@echo "  build            - Build all binaries"
	@echo "  build-train      - Build training binary only"
	@echo "  build-infer      - Build inference binary only"
	@echo "  train            - Run training with default parameters"
	@echo "  train-custom     - Run training with custom parameters"
	@echo "  infer            - Run inference (generate samples)"
	@echo "  infer-interactive - Run inference in interactive mode"
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
	@echo "  make train                  # Train with defaults"
	@echo "  make train-custom           # Train with custom params"
	@echo "  make infer                  # Generate samples"
	@echo "  make infer-interactive      # Interactive generation"
