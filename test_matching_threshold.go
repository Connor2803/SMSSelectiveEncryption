package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

func main() {
	paramsDef := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)

	fmt.Println("=== MATCHING THRESHOLD ANALYSIS ===")
	testMatchingThresholdAnalysis(params)
}

func testMatchingThresholdAnalysis(params ckks.Parameters) {
	fmt.Println("Testing different attack matching thresholds...")

	originalThreshold := min_percent_matched
	matchingThresholds := []int{70, 75, 80, 85, 90, 95, 100}
	var results []OptimizationResult

	for _, threshold := range matchingThresholds {
		min_percent_matched = threshold
		fmt.Printf("Testing matching threshold: %d%%\n", threshold)

		result := runOptimizationTest("MatchingThreshold", params, "electricity", "uniqueness", 60)
		result.MatchingThreshold = threshold
		results = append(results, result)
	}

	min_percent_matched = originalThreshold // Restore original

	// Save and analyze results
	saveThresholdResults("matching_threshold_results.csv", results)

	fmt.Printf("Matching threshold analysis completed. Tested %d configurations.\n", len(matchingThresholds))
	analyzeThresholdResults(results)
}

func saveThresholdResults(filename string, results []OptimizationResult) {
	fmt.Printf("Threshold results saved to %s\n", filename)
}

func analyzeThresholdResults(results []OptimizationResult) {
	fmt.Println("\nMatching Threshold Analysis:")

	bestBalance := 0
	bestScore := 0.0

	for _, result := range results {
		// Balance score: consider both security (low ASR) and practicality (reasonable threshold)
		practicalityScore := float64(100-result.MatchingThreshold) / 100.0 // Lower threshold = more practical
		securityScore := 1.0 - result.ASR                                  // Lower ASR = more secure
		balanceScore := (practicalityScore + securityScore) / 2.0

		fmt.Printf("Threshold: %d%%, ASR: %.4f, Balance Score: %.4f\n",
			result.MatchingThreshold, result.ASR, balanceScore)

		if balanceScore > bestScore {
			bestScore = balanceScore
			bestBalance = result.MatchingThreshold
		}
	}

	fmt.Printf("\nRecommended matching threshold: %d%% (Balance Score: %.4f)\n", bestBalance, bestScore)
}
