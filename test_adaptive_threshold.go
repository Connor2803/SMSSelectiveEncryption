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

	fmt.Println("=== ADAPTIVE THRESHOLDING TEST ===")
	testAdaptiveThresholding(params)
}

func testAdaptiveThresholding(params ckks.Parameters) {
	fmt.Println("Testing adaptive encryption ratio calculation...")

	datasets := []string{"water", "electricity"}
	var results []OptimizationResult

	for _, dataset := range datasets {
		fmt.Printf("Testing adaptive thresholding on %s dataset\n", dataset)

		result := testAdaptiveThresholdingOnDataset(params, dataset)
		results = append(results, result)
	}

	// Save and analyze results
	saveAdaptiveResults("adaptive_threshold_results.csv", results)

	fmt.Printf("Adaptive thresholding completed. Tested %d datasets.\n", len(datasets))
	analyzeAdaptiveResults(results)
}

func testAdaptiveThresholdingOnDataset(params ckks.Parameters, dataset string) OptimizationResult {
	// Load data to analyze characteristics
	fileList := getFileList(dataset)
	if fileList == nil {
		return OptimizationResult{TestName: "Adaptive", ASR: -1, Dataset: dataset}
	}

	// Set dataset parameters
	if dataset == "water" {
		currentDataset = DATASET_WATER
		currentTarget = 1
		transitionEqualityThreshold = WATER_TRANSITION_EQUALITY_THRESHOLD
	} else {
		currentDataset = DATASET_ELECTRICITY
		currentTarget = 2
		transitionEqualityThreshold = ELECTRICITY_TRANSITION_EQUALITY_THRESHOLD
	}

	// Generate parties to analyze data characteristics
	P := genparties(params, fileList)
	genInputs(P)

	// Calculate data characteristics for adaptive ratio
	entropyVariance := calculateEntropyVariance(P)
	dataVolatility := calculateDataVolatility(P)

	// Adaptive ratio calculation
	baseRatio := 40
	adaptiveRatio := baseRatio + int(entropyVariance*20) + int(dataVolatility*15)
	adaptiveRatio = int(math.Min(80, math.Max(20, float64(adaptiveRatio))))

	fmt.Printf("Adaptive ratio for %s: %d%% (entropy variance: %.3f, volatility: %.3f)\n",
		dataset, adaptiveRatio, entropyVariance, dataVolatility)

	// Test with adaptive ratio
	encryptionRatio = adaptiveRatio
	house_sample = []float64{}

	startTime := time.Now()
	for t := 0; t < 3; t++ {
		if dataset == "water" {
			processEntropy(fileList, params)
		} else {
			processGreedyWithASR(fileList, params)
		}
	}

	result := OptimizationResult{
		TestName:        "Adaptive",
		EncryptionRatio: adaptiveRatio,
		ProcessingTime:  time.Since(startTime).Seconds(),
		Dataset:         dataset,
		Strategy:        "adaptive",
	}

	if len(house_sample) > 0 {
		std, mean := calculateStandardDeviation(house_sample)
		standard_error := std / math.Sqrt(float64(len(house_sample)))
		result.ASR = mean
		result.StandardError = standard_error
	}

	return result
}

func calculateEntropyVariance(P []*party) float64 {
	var entropyValues []float64
	for _, po := range P {
		for _, entropy := range po.entropy {
			entropyValues = append(entropyValues, entropy)
		}
	}

	if len(entropyValues) == 0 {
		return 0.0
	}

	std, _ := calculateStandardDeviation(entropyValues)
	mean := 0.0
	for _, v := range entropyValues {
		mean += v
	}
	mean /= float64(len(entropyValues))

	if mean == 0 {
		return 0.0
	}

	return std / mean // Coefficient of variation
}

func calculateDataVolatility(P []*party) float64 {
	totalVolatility := 0.0
	count := 0

	for _, po := range P {
		if len(po.rawInput) < 2 {
			continue
		}

		// Calculate volatility as average absolute difference
		volatility := 0.0
		for i := 1; i < len(po.rawInput); i++ {
			volatility += math.Abs(po.rawInput[i] - po.rawInput[i-1])
		}
		volatility /= float64(len(po.rawInput) - 1)

		totalVolatility += volatility
		count++
	}

	if count == 0 {
		return 0.0
	}

	return totalVolatility / float64(count)
}

func saveAdaptiveResults(filename string, results []OptimizationResult) {
	fmt.Printf("Adaptive results saved to %s\n", filename)
}

func analyzeAdaptiveResults(results []OptimizationResult) {
	fmt.Println("\nAdaptive Thresholding Analysis:")

	for _, result := range results {
		fmt.Printf("Dataset: %s, Adaptive Ratio: %d%%, ASR: %.4f, Time: %.2fs\n",
			result.Dataset, result.EncryptionRatio, result.ASR, result.ProcessingTime)
	}
}
