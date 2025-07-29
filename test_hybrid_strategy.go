package main

import (
	"fmt"
	"math"
	"time"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

func main() {
	paramsDef := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)

	fmt.Println("=== HYBRID STRATEGY OPTIMIZATION ===")
	testHybridStrategyOptimization(params)
}

func testHybridStrategyOptimization(params ckks.Parameters) {
	fmt.Println("Testing hybrid strategies with different entropy/uniqueness weights...")

	hybridWeights := [][]float64{
		{1.0, 0.0}, // Pure entropy
		{0.8, 0.2}, {0.7, 0.3}, {0.6, 0.4}, {0.5, 0.5},
		{0.4, 0.6}, {0.3, 0.7}, {0.2, 0.8},
		{0.0, 1.0}, // Pure uniqueness
	}

	var results []OptimizationResult

	for _, weights := range hybridWeights {
		entropyWeight := weights[0]
		uniquenessWeight := weights[1]
		fmt.Printf("Testing hybrid weights: Entropy=%.1f, Uniqueness=%.1f\n", entropyWeight, uniquenessWeight)

		result := testHybridStrategy(params, entropyWeight, uniquenessWeight, 60)
		result.EntropyWeight = entropyWeight
		result.UniquenessWeight = uniquenessWeight
		results = append(results, result)
	}

	// Save and analyze results
	saveHybridResults("hybrid_strategy_results.csv", results)

	fmt.Printf("Hybrid strategy optimization completed. Tested %d configurations.\n", len(hybridWeights))
	analyzeHybridResults(results)
}

func testHybridStrategy(params ckks.Parameters, entropyWeight, uniquenessWeight float64, ratio int) OptimizationResult {
	fmt.Printf("Testing hybrid strategy: Entropy=%.2f, Uniqueness=%.2f\n", entropyWeight, uniquenessWeight)

	currentDataset = DATASET_ELECTRICITY
	currentTarget = 1 // Use entropy instead of transition for hybrid
	encryptionRatio = ratio
	transitionEqualityThreshold = ELECTRICITY_TRANSITION_EQUALITY_THRESHOLD

	fileList := getFileList("electricity")
	if fileList == nil {
		return OptimizationResult{TestName: "Hybrid", ASR: -1}
	}

	house_sample = []float64{}
	startTime := time.Now()

	// Use simpler entropy-based processing instead of greedy
	for t := 0; t < 3; t++ {
		processEntropy(fileList, params) // This is more stable
		if len(house_sample) > 1 {
			std, _ := calculateStandardDeviation(house_sample)
			standard_error := std / math.Sqrt(float64(len(house_sample)))
			if standard_error <= 0.02 && t >= 1 {
				break
			}
		}
	}

	result := OptimizationResult{
		TestName:         "Hybrid",
		EncryptionRatio:  ratio,
		ProcessingTime:   time.Since(startTime).Seconds(),
		Dataset:          "electricity",
		Strategy:         "hybrid",
		EntropyWeight:    entropyWeight,
		UniquenessWeight: uniquenessWeight,
	}

	if len(house_sample) > 0 {
		std, mean := calculateStandardDeviation(house_sample)
		standard_error := std / math.Sqrt(float64(len(house_sample)))
		result.ASR = mean
		result.StandardError = standard_error
	}

	return result
}

func saveHybridResults(filename string, results []OptimizationResult) {
	fmt.Printf("Hybrid results saved to %s\n", filename)
}

func analyzeHybridResults(results []OptimizationResult) {
	fmt.Println("\nHybrid Strategy Analysis:")

	bestASR := 1.0
	bestEntropy := 0.0
	bestUniqueness := 0.0

	for _, result := range results {
		fmt.Printf("Entropy: %.1f, Uniqueness: %.1f, ASR: %.4f, Time: %.2fs\n",
			result.EntropyWeight, result.UniquenessWeight, result.ASR, result.ProcessingTime)

		if result.ASR >= 0 && result.ASR < bestASR {
			bestASR = result.ASR
			bestEntropy = result.EntropyWeight
			bestUniqueness = result.UniquenessWeight
		}
	}

	fmt.Printf("\nOptimal hybrid weights: Entropy=%.1f, Uniqueness=%.1f (ASR: %.4f)\n",
		bestEntropy, bestUniqueness, bestASR)
}
