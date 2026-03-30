// Command infer runs inference with a trained GPT model.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/microgpt/go/dataset"
	"github.com/microgpt/go/inference"
	"github.com/microgpt/go/mlx"
	"github.com/microgpt/go/model"
	"github.com/microgpt/go/tensor"
	"github.com/microgpt/go/tokenizer"
)

func main() {
	// Command-line flags
	dataPath := flag.String("data", "input.txt", "path to input dataset file (for vocab)")
	embedDim := flag.Int("embed", 16, "embedding dimension")
	numHeads := flag.Int("heads", 4, "number of attention heads")
	numLayers := flag.Int("layers", 1, "number of transformer layers")
	blockSize := flag.Int("block", 16, "context block size")
	seed := flag.Int64("seed", 42, "random seed")
	temperature := flag.Float64("temp", 0.5, "sampling temperature")
	samples := flag.Int("samples", 20, "number of samples to generate")
	interactive := flag.Bool("interactive", false, "run in interactive mode")
	verbose := flag.Bool("verbose", false, "enable verbose output including backend info")
	flag.Parse()

	fmt.Println("=== MicroGPT Inference ===")
	fmt.Println()

	// Show backend information
	fmt.Printf("Backend: %s\n", mlx.GetBackendInfo())
	if *verbose {
		fmt.Printf("MLX enabled: %v\n", mlx.IsUsingMLX())
	}
	fmt.Println()

	// Initialize RNG
	tensor.InitRNG(*seed)

	// Load dataset for vocabulary
	fmt.Println("Loading vocabulary...")
	ds, err := dataset.LoadDefault(*dataPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading dataset: %v\n", err)
		os.Exit(1)
	}

	// Create tokenizer
	tok := tokenizer.New(ds.Docs())
	fmt.Printf("Vocabulary size: %d\n", tok.VocabSize())

	// Create model with random weights (for demo purposes)
	// In a real scenario, you would load trained weights
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

	// Create generator
	genCfg := inference.Config{
		Temperature: *temperature,
		MaxLen:      *blockSize,
		Seed:        *seed,
	}
	gen := inference.NewGenerator(genCfg, m, tok)

	if *interactive {
		runInteractive(gen, tok, *verbose)
	} else {
		// Generate samples
		if *verbose {
			fmt.Printf("--- Generating %d samples ---\n", *samples)
		} else {
			fmt.Println("--- Inference ---")
		}
		for i := 0; i < *samples; i++ {
			sample := gen.Generate()
			fmt.Printf("sample %2d: %s\n", i+1, sample)
		}
	}
}

// runInteractive runs an interactive session for text generation.
func runInteractive(gen *inference.Generator, tok *tokenizer.Tokenizer, verbose bool) {
	fmt.Println("Interactive mode")
	if verbose {
		fmt.Println("Commands:")
		fmt.Println("  <text>     - generate continuation of text")
		fmt.Println("  :temp <n>  - set temperature (0.1-2.0)")
		fmt.Println("  :seed <n>  - set random seed")
		fmt.Println("  :sample    - generate a random sample")
		fmt.Println("  :quit      - exit")
		fmt.Println()
	}

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, ":") {
			parts := strings.Fields(line)
			cmd := parts[0]

			switch cmd {
			case ":temp":
				if len(parts) < 2 {
					fmt.Println("Usage: :temp <value>")
					continue
				}
				temp, err := strconv.ParseFloat(parts[1], 64)
				if err != nil || temp <= 0 {
					fmt.Println("Temperature must be a positive number")
					continue
				}
				gen.SetTemperature(temp)
				fmt.Printf("Temperature set to %.2f\n", temp)

			case ":seed":
				if len(parts) < 2 {
					fmt.Println("Usage: :seed <value>")
					continue
				}
				seed, err := strconv.ParseInt(parts[1], 10, 64)
				if err != nil {
					fmt.Println("Invalid seed value")
					continue
				}
				gen.SetSeed(seed)
				fmt.Printf("Seed set to %d\n", seed)

			case ":sample":
				sample := gen.Generate()
				if verbose {
					fmt.Printf("Generated: %s\n", sample)
				} else {
					fmt.Println(sample)
				}

			case ":quit", ":exit", ":q":
				fmt.Println("Goodbye!")
				return

			default:
				fmt.Printf("Unknown command: %s\n", cmd)
			}
		} else {
			// Generate continuation
			result := gen.GenerateWithPrompt(line)
			if verbose {
				fmt.Printf("Generated: %s\n", result)
			} else {
				fmt.Println(result)
			}
		}
	}
}
