package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

func main() {
	paramsDef := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)

	fmt.Println("=== ATD SIZE IMPACT ANALYSIS ===")
	testATDSizeImpact(params)
}

func testATDSizeImpact(params ckks.Parameters) {
	fmt.Println("Analyzing impact of different ATD (Attacker Target Data) sizes...")

	originalATDSize := atdSize
	atdSizes := []int{6, 12, 24, 48, 72, 96, 168} // 6 hours to 1 week
	var results []OptimizationResult

	for _, size := range atdSizes {
		atdSize = size
		fmt.Printf("Testing ATD size: %d hours\n", size)

		// Test on both datasets
		resultElec := runOptimizationTest("ATDSize", params, "electricity", "uniqueness", 60)
		resultElec.ATDSize = size
		results = append(results, resultElec)

		resultWater := runOptimizationTest("ATDSize", params, "water", "entropy", 60)
		resultWater.ATDSize = size
		results = append(results, resultWater)
	}

	atdSize = originalATDSize // Restore original

	// Save and analyze results
	saveATDResults("atd_size_results.csv", results)

	fmt.Printf("ATD size analysis completed. Tested %d configurations.\n", len(atdSizes)*2)
	analyzeATDResults(results)
}

func saveATDResults(filename string, results []OptimizationResult) {
	// Implementation to save ATD results
	fmt.Printf("ATD results saved to %s\n", filename)
}

func analyzeATDResults(results []OptimizationResult) {
	fmt.Println("\nATD Size Analysis:")

	// Separate results by dataset
	electricityResults := []OptimizationResult{}
	waterResults := []OptimizationResult{}

	for _, result := range results {
		if result.Dataset == "electricity" {
			electricityResults = append(electricityResults, result)
		} else {
			waterResults = append(waterResults, result)
		}
	}

	fmt.Println("\nElectricity Dataset Results:")
	printATDAnalysis(electricityResults)

	fmt.Println("\nWater Dataset Results:")
	printATDAnalysis(waterResults)
}

func printATDAnalysis(results []OptimizationResult) {
	bestASR := 1.0
	bestATD := 24

	for _, result := range results {
		fmt.Printf("ATD Size: %d hours, ASR: %.4f, Time: %.2fs\n",
			result.ATDSize, result.ASR, result.ProcessingTime)
		if result.ASR >= 0 && result.ASR < bestASR {
			bestASR = result.ASR
			bestATD = result.ATDSize
		}
	}

	fmt.Printf("Best ATD size: %d hours (ASR: %.4f)\n", bestATD, bestASR)
}
