package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

const (
	encryptionRatio = 20 // percent of blocks to encrypt
)

// Chaotic logistic map substitution
func chaoticSubstitution(data []float64, seed float64) {
	x := seed
	for i := range data {
		x = 4 * x * (1 - x)
		data[i] = math.Float64frombits(math.Float64bits(data[i]) ^ math.Float64bits(x))
	}
}

// Approximate block entropy via histogram
func blockEntropy(block []float64) float64 {
	count := make(map[int]int)
	for _, v := range block {
		b := int(math.Mod(v*100, 256))
		count[b]++
	}
	entropy := 0.0
	n := float64(len(block))
	for _, c := range count {
		p := float64(c) / n
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	return entropy
}

// Load any file in a directory, parse last column as float64
func loadNumericSeries(dir string, skipHeader bool) ([]float64, error) {
	var series []float64
	files, err := filepath.Glob(filepath.Join(dir, "*"))
	if err != nil {
		return nil, err
	}
	fmt.Printf("Found %d files in %s\n", len(files), dir)
	for _, file := range files {
		info, err := os.Stat(file)
		if err != nil || info.IsDir() {
			continue
		}
		fmt.Printf("Reading %s\n", file)
		f, err := os.Open(file)
		if err != nil {
			continue
		}
		scanner := bufio.NewScanner(f)
		if skipHeader && scanner.Scan() {
			// skip header
		}
		for scanner.Scan() {
			line := scanner.Text()
			var fields []string
			if strings.Contains(line, ",") {
				fields = strings.Split(line, ",")
			} else {
				fields = strings.FieldsFunc(line, func(r rune) bool { return r == '\t' || r == ' ' })
			}
			if len(fields) < 1 {
				continue
			}
			valStr := fields[len(fields)-1]
			val, err := strconv.ParseFloat(strings.TrimSpace(valStr), 64)
			if err != nil {
				continue
			}
			series = append(series, val)
		}
		f.Close()
	}
	return series, nil
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Use a very conservative CKKS parameter set that's guaranteed to work
	var params ckks.Parameters
	var err error

	// Use parameter sets with sufficient Q primes to support the ring degree
	paramSets := []ckks.ParametersLiteral{
		// Very conservative - more Q primes for stability
		{
			LogN:         10,                                            // Ring degree 2^10 = 1024
			LogQ:         []int{60, 50, 50, 50, 50, 50, 50, 50, 50, 50}, // Many Q primes
			LogP:         []int{60, 60},                                 // Multiple P primes
			LogSlots:     9,                                             // 2^9 = 512 slots
			DefaultScale: 1 << 50,
		},
		// Slightly larger with many primes
		{
			LogN:         11,                                                // Ring degree 2^11 = 2048
			LogQ:         []int{60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50}, // Many Q primes
			LogP:         []int{60, 60},                                     // Multiple P primes
			LogSlots:     10,                                                // 2^10 = 1024 slots
			DefaultScale: 1 << 50,
		},
		// Standard configuration with sufficient primes
		{
			LogN:         12,                                                    // Ring degree 2^12 = 4096
			LogQ:         []int{60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50}, // Many Q primes
			LogP:         []int{60, 60},                                         // Multiple P primes
			LogSlots:     11,                                                    // 2^11 = 2048 slots
			DefaultScale: 1 << 50,
		},
		// Try the smallest working configuration from Lattigo examples
		{
			LogN:         13,                                                        // Ring degree 2^13 = 8192
			LogQ:         []int{60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50}, // Many Q primes
			LogP:         []int{60, 60},                                             // Multiple P primes
			LogSlots:     12,                                                        // 2^12 = 4096 slots
			DefaultScale: 1 << 50,
		},
		// Fallback to predefined sets (these should work)
		ckks.PN12QP109,
		ckks.PN13QP218,
		ckks.PN14QP438,
	}

	var workingParamIndex = -1
	for i, paramSet := range paramSets {
		fmt.Printf("Trying parameter set %d...\n", i)

		// Handle both custom parameter literals and predefined constants
		var paramErr error
		if i >= 4 { // These are the predefined constants
			switch i {
			case 4:
				params, paramErr = ckks.NewParametersFromLiteral(ckks.PN12QP109)
			case 5:
				params, paramErr = ckks.NewParametersFromLiteral(ckks.PN13QP218)
			case 6:
				params, paramErr = ckks.NewParametersFromLiteral(ckks.PN14QP438)
			}
		} else {
			params, paramErr = ckks.NewParametersFromLiteral(paramSet)
		}

		if paramErr != nil {
			fmt.Printf("Parameter set %d failed creation: %v\n", i, paramErr)
			continue
		}

		// Test if we can create an encoder without errors
		var encoder ckks.Encoder
		var encoderCreated bool
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("Parameter set %d crashed during encoder creation: %v\n", i, r)
					encoderCreated = false
				}
			}()
			encoder = ckks.NewEncoder(params)
			encoderCreated = true
		}()

		if !encoderCreated {
			fmt.Printf("Parameter set %d failed encoder creation\n", i)
			continue
		}

		// Try a simple encoding test with minimal data
		testDataSize := min(4, params.Slots()) // Use very small test data
		testData := make([]complex128, testDataSize)
		for j := 0; j < testDataSize; j++ {
			testData[j] = complex(float64(j+1), 0)
		}

		logSlots := params.LogSlots()
		scale := params.DefaultScale()

		fmt.Printf("Testing encoding with parameter set %d (LogSlots: %d, Slots: %d, QCount: %d, PCount: %d)\n",
			i, logSlots, params.Slots(), params.QCount(), params.PCount())

		// Try to encode a small test vector
		var testSuccess bool
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("Parameter set %d crashed during test encode: %v\n", i, r)
					testSuccess = false
				}
			}()

			// Use the correct number of slots - don't exceed what's available
			actualLogSlots := params.LogSlots()
			testPt := encoder.EncodeNew(testData, actualLogSlots, scale, actualLogSlots)
			if testPt != nil {
				fmt.Printf("Parameter set %d works! Using this configuration.\n", i)
				testSuccess = true
				workingParamIndex = i
			} else {
				fmt.Printf("Parameter set %d returned nil plaintext\n", i)
				testSuccess = false
			}
		}()

		if testSuccess {
			break
		}
	}

	if workingParamIndex == -1 {
		log.Fatal("All parameter sets failed - there may be an issue with the Lattigo installation")
	}

	// Get the number of slots available
	logSlots := params.LogSlots()
	slots := params.Slots()
	fmt.Printf("CKKS parameters allow %d slots (2^%d)\n", slots, logSlots)

	// Validate parameters
	fmt.Printf("Ring degree N: %d\n", params.N())
	fmt.Printf("Log2 of slots: %d\n", logSlots)
	fmt.Printf("Q levels: %d\n", params.QCount())
	fmt.Printf("P levels: %d\n", params.PCount())

	// Use the slot capacity as our section size, but ensure we don't exceed it
	sectionSize := slots

	// Load electricity data - if path doesn't exist, create some dummy data
	var data []float64
	elec, err := loadNumericSeries("./examples/datasets/electricity/households_10240", false)
	if err != nil {
		fmt.Printf("Could not load data from file, using dummy data: %v\n", err)
		// Create some dummy data for testing
		data = make([]float64, 5000)
		for i := range data {
			data[i] = rand.Float64() * 100
		}
	} else {
		data = elec
	}

	fmt.Printf("Total samples loaded: %d\n", len(data))
	if len(data) == 0 {
		log.Fatal("No data to process.")
	}

	// Partition into blocks
	nBlocks := int(math.Ceil(float64(len(data)) / float64(sectionSize)))
	fmt.Printf("Partitioning into %d blocks of up to %d samples each\n", nBlocks, sectionSize)
	blocks := make([][]float64, nBlocks)
	for i := 0; i < nBlocks; i++ {
		start := i * sectionSize
		end := start + sectionSize
		if end > len(data) {
			end = len(data)
		}
		blocks[i] = make([]float64, sectionSize) // Always allocate full slot size
		copy(blocks[i], data[start:end])
		// Fill remaining slots with zeros if needed
		for j := end - start; j < sectionSize; j++ {
			blocks[i][j] = 0.0
		}
		fmt.Printf(" Block %d size: %d (data: %d, padding: %d)\n", i, len(blocks[i]), end-start, sectionSize-(end-start))
	}

	// Compute entropy per block (only on actual data, not padding)
	type entVal struct {
		idx int
		val float64
	}
	total := len(blocks)
	entList := make([]entVal, total)
	for i, blk := range blocks {
		// Calculate entropy only on non-zero values for last block
		actualSize := sectionSize
		if i == len(blocks)-1 && len(data)%sectionSize != 0 {
			actualSize = len(data) % sectionSize
		}
		entList[i] = entVal{i, blockEntropy(blk[:actualSize])}
		fmt.Printf(" Entropy block %d: %.3f\n", i, entList[i].val)
	}

	// Select top X% by entropy
	target := total * encryptionRatio / 100
	if target == 0 {
		target = 1 // Ensure at least one block is encrypted
	}
	fmt.Printf("Selecting top %d blocks for encryption\n", target)
	for i := 0; i < target; i++ {
		maxJ := i
		for j := i + 1; j < total; j++ {
			if entList[j].val > entList[maxJ].val {
				maxJ = j
			}
		}
		entList[i], entList[maxJ] = entList[maxJ], entList[i]
		fmt.Printf(" Selected block %d with entropy %.3f\n", entList[i].idx, entList[i].val)
	}
	selected := make(map[int]bool)
	for i := 0; i < target; i++ {
		selected[entList[i].idx] = true
	}

	// Complete CKKS setup
	kg := ckks.NewKeyGenerator(params)
	sk, pk := kg.GenKeyPair()
	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, pk)
	decryptor := ckks.NewDecryptor(params, sk)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{})

	// Encrypt selected blocks and accumulate sum
	scale := params.DefaultScale()
	var sumCt *rlwe.Ciphertext
	first := true

	for i, blk := range blocks {
		if selected[i] {
			// Apply chaotic substitution
			chaoticSubstitution(blk, rand.Float64())

			// Convert to complex128 for encoding (CKKS expects complex numbers)
			// Ensure we don't exceed the slot capacity
			dataSize := min(len(blk), slots)
			complexData := make([]complex128, dataSize)
			for j := 0; j < dataSize; j++ {
				complexData[j] = complex(blk[j], 0) // Real part only
			}

			fmt.Printf("Attempting to encrypt block %d with %d values (slots available: %d)...\n", i, len(complexData), slots)

			// Add error handling for encoding step
			var pt *rlwe.Plaintext
			func() {
				defer func() {
					if r := recover(); r != nil {
						log.Fatalf("Encoding failed for block %d: %v", i, r)
					}
				}()
				// Use the correct logSlots parameter - don't exceed what's available
				actualLogSlots := params.LogSlots()
				pt = encoder.EncodeNew(complexData, actualLogSlots, scale, actualLogSlots)
			}()

			if pt == nil {
				log.Fatalf("Failed to encode block %d", i)
			}

			// Encrypt
			ct := encryptor.EncryptNew(pt)
			if ct == nil {
				log.Fatalf("Failed to encrypt block %d", i)
			}

			fmt.Printf("Successfully encrypted block %d\n", i)

			// Add to sum
			if first {
				sumCt = ct
				first = false
			} else {
				newSum := evaluator.AddNew(sumCt, ct)
				sumCt = newSum
			}
		} else {
			fmt.Printf("Plain block %d\n", i)
		}
	}

	// Decrypt and display sum
	if sumCt != nil {
		ptRes := decryptor.DecryptNew(sumCt)
		resValues := encoder.Decode(ptRes, logSlots)
		if len(resValues) > 0 {
			totalSum := real(resValues[0])
			fmt.Printf("Decrypted homomorphic sum (approx): %.3f\n", totalSum)

			// Show a few more values to verify
			fmt.Printf("First few decrypted values: ")
			for i := 0; i < min(5, len(resValues)); i++ {
				fmt.Printf("%.3f ", real(resValues[i]))
			}
			fmt.Println()
		} else {
			fmt.Println("No decrypted values available.")
		}
	} else {
		fmt.Println("No encrypted blocks; nothing to decrypt.")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
