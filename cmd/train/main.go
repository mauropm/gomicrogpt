// Command train trains a GPT model on a text dataset.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/microgpt/go/dataset"
	"github.com/microgpt/go/inference"
	"github.com/microgpt/go/mlx"
	"github.com/microgpt/go/model"
	"github.com/microgpt/go/tensor"
	"github.com/microgpt/go/tokenizer"
	"github.com/microgpt/go/train"
)

func main() {
	// Command-line flags
	dataPath := flag.String("data", "input.txt", "path to input dataset file")
	steps := flag.Int("steps", 1000, "number of training steps")
	lr := flag.Float64("lr", 0.01, "learning rate")
	beta1 := flag.Float64("beta1", 0.85, "Adam beta1")
	beta2 := flag.Float64("beta2", 0.99, "Adam beta2")
	embedDim := flag.Int("embed", 16, "embedding dimension")
	numHeads := flag.Int("heads", 4, "number of attention heads")
	numLayers := flag.Int("layers", 1, "number of transformer layers")
	blockSize := flag.Int("block", 16, "context block size")
	seed := flag.Int64("seed", 42, "random seed")
	temperature := flag.Float64("temp", 0.5, "sampling temperature")
	samples := flag.Int("samples", 20, "number of samples to generate after training")
	verbose := flag.Bool("verbose", false, "enable verbose output including backend info")
	flag.Parse()

	fmt.Println("=== MicroGPT Training ===")
	fmt.Println()

	// Show backend information (always shown for clarity)
	fmt.Printf("Backend: %s\n", mlx.GetBackendInfo())
	if *verbose {
		fmt.Printf("MLX enabled: %v\n", mlx.IsUsingMLX())
	}
	fmt.Println()

	// Initialize RNG
	tensor.InitRNG(*seed)

	// Load dataset
	fmt.Println("Loading dataset...")
	ds, err := dataset.LoadDefault(*dataPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading dataset: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d documents\n", ds.Len())

	// Create tokenizer
	tok := tokenizer.New(ds.Docs())
	fmt.Printf("Vocabulary size: %d (%d chars + 1 BOS)\n", tok.VocabSize(), tok.NumChars())

	// Create model
	cfg := model.Config{
		VocabSize: tok.VocabSize(),
		EmbedDim:  *embedDim,
		NumHeads:  *numHeads,
		NumLayers: *numLayers,
		BlockSize: *blockSize,
		HeadDim:   *embedDim / *numHeads,
	}
	m := model.New(cfg)
	fmt.Printf("Model created: %d parameters\n", m.NumParams())

	if *verbose {
		fmt.Printf("Model config: embed=%d, heads=%d, layers=%d, block=%d\n",
			*embedDim, *numHeads, *numLayers, *blockSize)
	}
	fmt.Println()

	// Create trainer
	trainCfg := train.Config{
		NumSteps:     *steps,
		LearningRate: *lr,
		Beta1:        *beta1,
		Beta2:        *beta2,
		Eps:          1e-8,
		Temperature:  *temperature,
		Seed:         *seed,
	}
	trainer := train.NewTrainer(trainCfg, m, tok, ds)

	// Train
	fmt.Println("Training...")
	trainer.Train()
	fmt.Println()

	// Inference
	if *verbose {
		fmt.Println("--- Inference (generated samples) ---")
	} else {
		fmt.Println("--- Inference ---")
	}
	genCfg := inference.Config{
		Temperature: *temperature,
		MaxLen:      *blockSize,
		Seed:        *seed + 1, // Different seed for variety
	}
	gen := inference.NewGenerator(genCfg, m, tok)

	for i := 0; i < *samples; i++ {
		sample := gen.Generate()
		fmt.Printf("sample %2d: %s\n", i+1, sample)
	}
}
