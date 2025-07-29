package main

import (
	"fmt"
	"time"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

func main() {
	paramsDef := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)

	fmt.Println("=== SCALABILITY ANALYSIS ===")
	testScalabilityAnalysis(params)
}

func testScalabilityAnalysis(params ckks.Parameters) {
	fmt.Println("Testing scalability with increasing encryption ratios...")

	encryptionRatios := []int{10, 20, 30, 40, 50, 60, 70, 80, 90}
	var results []OptimizationResult

	for _, ratio := range encryptionRatios {
		fmt.Printf("Testing scalability with %d%% encryption\n", ratio)

		startTime := time.Now()
		result := runOptimizationTest("Scalability", params, "electricity", "uniqueness", ratio)
		result.ProcessingTime = time.Since(startTime).Seconds()
		result.TestName = fmt.Sprintf("Scalability_%d", ratio)
		results = append(results, result)
	}

	// Save and analyze results
	saveScalabilityResults("scalability_analysis_results.csv", results)

	fmt.Printf("Scalability analysis completed. Tested %d ratios.\n", len(encryptionRatios))
	analyzeScalabilityResults(results, encryptionRatios)
}

func saveScalabilityResults(filename string, results []OptimizationResult) {
	fmt.Printf("Scalability results saved to %s\n", filename)
}

func analyzeScalabilityResults(results []OptimizationResult, ratios []int) {
	fmt.Println("\nScalability Analysis:")

	// Print individual results
	for i, result := range results {
		ratio := ratios[i]

		fmt.Printf("Encryption Ratio: %d%%\n", ratio)
		fmt.Printf("  ASR: %.4f\n", result.ASR)
		fmt.Printf("  Processing Time: %.2fs\n", result.ProcessingTime)
		fmt.Printf("  Time per encrypted %: %.3fs\n\n", result.ProcessingTime/float64(ratio))
	}

	// Analyze computational complexity
	if len(results) >= 3 {
		fmt.Println("Computational Complexity Analysis:")

		// Calculate correlation between encryption ratio and processing time
		complexity := analyzeComplexity(results, ratios)
		fmt.Printf("Estimated complexity: %s\n", complexity)

		// Find optimal ratio (balance of security and performance)
		optimalRatio := findOptimalRatio(results, ratios)
		fmt.Printf("Recommended encryption ratio: %d%% (best security/performance balance)\n", optimalRatio)

		// Performance prediction
		predictPerformance(results, ratios)
	}
}

func analyzeComplexity(results []OptimizationResult, ratios []int) string {
	if len(results) < 3 {
		return "Insufficient data"
	}

	// Simple complexity analysis by comparing growth rates
	timeRatios := make([]float64, len(results)-1)
	ratioRatios := make([]float64, len(results)-1)

	for i := 1; i < len(results); i++ {
		timeRatios[i-1] = results[i].ProcessingTime / results[i-1].ProcessingTime
		ratioRatios[i-1] = float64(ratios[i]) / float64(ratios[i-1])
	}

	// Calculate average growth factor
	avgTimeGrowth := 0.0
	avgRatioGrowth := 0.0
	for i := 0; i < len(timeRatios); i++ {
		avgTimeGrowth += timeRatios[i]
		avgRatioGrowth += ratioRatios[i]
	}
	avgTimeGrowth /= float64(len(timeRatios))
	avgRatioGrowth /= float64(len(ratioRatios))

	complexityFactor := avgTimeGrowth / avgRatioGrowth

	if complexityFactor < 1.1 {
		return "Sub-linear (Very efficient)"
	} else if complexityFactor < 1.3 {
		return "Linear (Expected)"
	} else if complexityFactor < 2.0 {
		return "Super-linear (Moderate efficiency loss)"
	} else {
		return "Quadratic or worse (Poor scalability)"
	}
}

func findOptimalRatio(results []OptimizationResult, ratios []int) int {
	bestScore := 0.0
	bestRatio := 50 // Default

	for i, result := range results {
		ratio := ratios[i]

		// Score based on security (lower ASR) and reasonable performance
		securityScore := 1.0 - result.ASR                       // Higher is better
		performanceScore := 1.0 / (1.0 + result.ProcessingTime) // Higher is better (lower time)
		utilizationScore := float64(ratio) / 100.0              // Higher ratio = more utility preserved

		// Weighted combination
		totalScore := securityScore*0.5 + performanceScore*0.3 + utilizationScore*0.2

		if totalScore > bestScore {
			bestScore = totalScore
			bestRatio = ratio
		}
	}

	return bestRatio
}

func predictPerformance(results []OptimizationResult, ratios []int) {
	fmt.Println("\nPerformance Predictions:")

	if len(results) < 2 {
		fmt.Println("Insufficient data for predictions")
		return
	}

	// Simple linear regression for time prediction
	// Using last two points for trend
	last := results[len(results)-1]
	secondLast := results[len(results)-2]

	timeSlope := (last.ProcessingTime - secondLast.ProcessingTime) /
		float64(ratios[len(ratios)-1]-ratios[len(ratios)-2])

	// Predict performance at 100%
	baseTime := last.ProcessingTime
	baseRatio := float64(ratios[len(ratios)-1])
	predicted100Time := baseTime + timeSlope*(100.0-baseRatio)

	fmt.Printf("Predicted time for 100%% encryption: %.2fs\n", predicted100Time)

	// Predict memory usage (if available)
	if last.MemoryUsage > 0 && secondLast.MemoryUsage > 0 {
		memorySlope := float64(last.MemoryUsage-secondLast.MemoryUsage) /
			float64(ratios[len(ratios)-1]-ratios[len(ratios)-2])
		predicted100Memory := float64(last.MemoryUsage) + memorySlope*(100.0-baseRatio)

		fmt.Printf("Predicted memory for 100%% encryption: %.2f MB\n",
			predicted100Memory/(1024*1024))
	}

	// Performance warning
	if predicted100Time > 60 { // More than 1 minute
		fmt.Println("WARNING: 100% encryption may be impractical for real-time applications")
	}
}
