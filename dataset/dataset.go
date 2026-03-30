// Package dataset provides dataset loading and shuffling functionality.
package dataset

import (
	"bufio"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"
)

// DefaultDatasetURL is the names dataset used for training.
const DefaultDatasetURL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"

// Dataset holds a list of documents (strings).
type Dataset struct {
	docs []string
}

// LoadFromFile loads documents from a local file.
func LoadFromFile(path string) (*Dataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	return loadFromReader(file)
}

// LoadFromURL downloads and loads documents from a URL.
func LoadFromURL(url string) (*Dataset, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to download dataset: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	return loadFromReader(resp.Body)
}

// LoadDefault loads the default names dataset, downloading if necessary.
func LoadDefault(localPath string) (*Dataset, error) {
	// Try to load from local file first
	if _, err := os.Stat(localPath); err == nil {
		return LoadFromFile(localPath)
	}

	// Download and save
	fmt.Printf("Downloading dataset from %s...\n", DefaultDatasetURL)
	resp, err := http.Get(DefaultDatasetURL)
	if err != nil {
		return nil, fmt.Errorf("failed to download dataset: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read dataset: %w", err)
	}

	if err := os.WriteFile(localPath, data, 0644); err != nil {
		return nil, fmt.Errorf("failed to save dataset: %w", err)
	}

	return loadFromReader(strings.NewReader(string(data)))
}

// loadFromReader reads documents from an io.Reader.
func loadFromReader(r io.Reader) (*Dataset, error) {
	var docs []string
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			docs = append(docs, line)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading dataset: %w", err)
	}

	return &Dataset{docs: docs}, nil
}

// Docs returns all documents.
func (d *Dataset) Docs() []string {
	return d.docs
}

// Len returns the number of documents.
func (d *Dataset) Len() int {
	return len(d.docs)
}

// Shuffle randomly shuffles the documents in place.
func (d *Dataset) Shuffle() {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(d.docs), func(i, j int) {
		d.docs[i], d.docs[j] = d.docs[j], d.docs[i]
	})
}

// Get returns the document at the given index (with wraparound).
func (d *Dataset) Get(index int) string {
	return d.docs[index%len(d.docs)]
}

// Iterator provides sequential access with wraparound.
type Iterator struct {
	dataset *Dataset
	index   int
}

// NewIterator creates a new iterator over the dataset.
func (d *Dataset) NewIterator() *Iterator {
	return &Iterator{dataset: d, index: 0}
}

// Next returns the next document and advances the iterator.
func (it *Iterator) Next() string {
	doc := it.dataset.docs[it.index%len(it.dataset.docs)]
	it.index++
	return doc
}

// Reset resets the iterator to the beginning.
func (it *Iterator) Reset() {
	it.index = 0
}
