// Package tokenizer provides a character-level tokenizer with BOS handling.
package tokenizer

import (
	"sort"
)

// Tokenizer maps characters to integer IDs and back.
type Tokenizer struct {
	chars      []rune            // sorted unique characters
	charToID   map[rune]int      // character to ID mapping
	idToChar   map[int]rune      // ID to character mapping
	bosID      int               // Beginning of Sequence token ID
	vocabSize  int               // total vocabulary size
}

// New creates a new tokenizer from a list of documents.
func New(docs []string) *Tokenizer {
	// Collect all unique characters
	charSet := make(map[rune]bool)
	for _, doc := range docs {
		for _, ch := range doc {
			charSet[ch] = true
		}
	}

	// Sort characters for deterministic ordering
	chars := make([]rune, 0, len(charSet))
	for ch := range charSet {
		chars = append(chars, ch)
	}
	sort.Slice(chars, func(i, j int) bool {
		return chars[i] < chars[j]
	})

	// Build mappings
	charToID := make(map[rune]int)
	idToChar := make(map[int]rune)
	for i, ch := range chars {
		charToID[ch] = i
		idToChar[i] = ch
	}

	// BOS token ID is after all character IDs
	bosID := len(chars)
	vocabSize := bosID + 1

	return &Tokenizer{
		chars:     chars,
		charToID:  charToID,
		idToChar:  idToChar,
		bosID:     bosID,
		vocabSize: vocabSize,
	}
}

// Encode converts a string to token IDs, with BOS tokens at boundaries.
// Example: "emma" -> [BOS, e, m, m, a, BOS]
func (t *Tokenizer) Encode(s string) []int {
	tokens := make([]int, 0, len(s)+2)
	tokens = append(tokens, t.bosID)
	for _, ch := range s {
		tokens = append(tokens, t.charToID[ch])
	}
	tokens = append(tokens, t.bosID)
	return tokens
}

// EncodeWithoutEndBOS converts a string to token IDs with only leading BOS.
// Used during inference when we're generating continuation.
func (t *Tokenizer) EncodeWithoutEndBOS(s string) []int {
	tokens := make([]int, 0, len(s)+1)
	tokens = append(tokens, t.bosID)
	for _, ch := range s {
		tokens = append(tokens, t.charToID[ch])
	}
	return tokens
}

// Decode converts token IDs back to a string (excluding BOS tokens).
func (t *Tokenizer) Decode(tokens []int) string {
	runes := make([]rune, 0, len(tokens))
	for _, id := range tokens {
		if id == t.bosID {
			continue
		}
		runes = append(runes, t.idToChar[id])
	}
	return string(runes)
}

// DecodeSingle converts a single token ID to a character (empty string for BOS).
func (t *Tokenizer) DecodeSingle(id int) string {
	if id == t.bosID {
		return ""
	}
	return string(t.idToChar[id])
}

// BOS returns the BOS token ID.
func (t *Tokenizer) BOS() int {
	return t.bosID
}

// VocabSize returns the vocabulary size.
func (t *Tokenizer) VocabSize() int {
	return t.vocabSize
}

// Chars returns the sorted list of unique characters.
func (t *Tokenizer) Chars() []rune {
	return t.chars
}

// NumChars returns the number of unique characters (excluding BOS).
func (t *Tokenizer) NumChars() int {
	return len(t.chars)
}
