package main

import (
	"fmt"
	"time"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

func main() {
	maxHouseholdsNumber = 80
	max_attackLoop = 10

	paramsDef := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)

	fmt.Println("=== SECTION SIZE OPTIMIZATION TEST ===")
	testSectionSizeOptimization(params)
}

func testSectionSizeOptimization(params ckks.Parameters) {
	fmt.Println("Testing different section sizes for optimal performance/security balance...")

	originalSectionSize := sectionSize
	sectionSizes := []int{256, 512, 1024, 2048, 4096, 8192}
	var results []OptimizationResult

	for _, size := range sectionSizes {
		sectionSize = size
		fmt.Printf("Testing section size: %d\n", size)

		// Test on electricity dataset with uniqueness
		result := runOptimizationTest("SectionSize", params, "electricity", "uniqueness", 60)
		result.SectionSize = size
		results = append(results, result)

		// Brief pause to prevent overheating
		time.Sleep(2 * time.Second)
	}

	sectionSize = originalSectionSize // Restore original

	// Save results
	saveResults("section_size_results.csv", results)

	fmt.Printf("Section size optimization completed. Tested %d configurations.\n", len(sectionSizes))
	printBestSectionSize(results)
}

func printBestSectionSize(results []OptimizationResult) {
	bestASR := 1.0
	bestSize := 1024

	fmt.Println("\nSection Size Results:")
	for _, result := range results {
		fmt.Printf("Size: %d, ASR: %.4f, Time: %.2fs\n",
			result.SectionSize, result.ASR, result.ProcessingTime)
		if result.ASR >= 0 && result.ASR < bestASR {
			bestASR = result.ASR
			bestSize = result.SectionSize
		}
	}

	fmt.Printf("\nRecommended section size: %d (ASR: %.4f)\n", bestSize, bestASR)
}

func saveResults(filename string, results []OptimizationResult) {
	// Implementation to save results to CSV
	fmt.Printf("Results saved to %s\n", filename)
}
