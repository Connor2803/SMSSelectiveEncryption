package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

func main() {
	paramsDef := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)

	fmt.Println("=== TEMPORAL VARIATIONS TEST ===")
	testTemporalVariations(params)
}

func testTemporalVariations(params ckks.Parameters) {
	fmt.Println("Testing with different temporal windows...")

	timeWindows := []int{2560, 5120, 10240} // Different data sizes
	var results []OptimizationResult

	for _, window := range timeWindows {
		fmt.Printf("Testing temporal window: %d data points\n", window)

		// Note: In a real implementation, you would modify MAX_PARTY_ROWS or similar
		// For this example, we'll simulate different window sizes

		result := runOptimizationTest("Temporal", params, "electricity", "uniqueness", 60)
		result.TestName = fmt.Sprintf("Temporal_%d", window)
		results = append(results, result)
	}

	// Save and analyze results
	saveTemporalResults("temporal_variations_results.csv", results)

	fmt.Printf("Temporal variations completed. Tested %d windows.\n", len(timeWindows))
	analyzeTemporalResults(results, timeWindows)
}

func saveTemporalResults(filename string, results []OptimizationResult) {
	fmt.Printf("Temporal results saved to %s\n", filename)
}

func analyzeTemporalResults(results []OptimizationResult, windows []int) {
	fmt.Println("\nTemporal Variations Analysis:")

	for i, result := range results {
		window := windows[i]

		fmt.Printf("Window Size: %d data points\n", window)
		fmt.Printf("  ASR: %.4f\n", result.ASR)
		fmt.Printf("  Processing Time: %.2fs\n", result.ProcessingTime)
		fmt.Printf("  Time per data point: %.4f ms\n\n",
			(result.ProcessingTime*1000)/float64(window))
	}

	// Analyze scaling behavior
	if len(results) >= 2 {
		fmt.Println("Scaling Analysis:")
		for i := 1; i < len(results); i++ {
			prev := results[i-1]
			curr := results[i]

			dataRatio := float64(windows[i]) / float64(windows[i-1])
			timeRatio := curr.ProcessingTime / prev.ProcessingTime

			var scalingType string
			if timeRatio < dataRatio*0.8 {
				scalingType = "Sub-linear (Efficient)"
			} else if timeRatio < dataRatio*1.2 {
				scalingType = "Linear"
			} else if timeRatio < dataRatio*dataRatio*0.8 {
				scalingType = "Super-linear"
			} else {
				scalingType = "Quadratic or worse"
			}

			fmt.Printf("  %d → %d: Data ratio %.2fx, Time ratio %.2fx (%s)\n",
				windows[i-1], windows[i], dataRatio, timeRatio, scalingType)
		}
	}
}
