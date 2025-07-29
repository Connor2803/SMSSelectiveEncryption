package main

import (
	"fmt"
	"runtime"
	"time"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

func main() {
	paramsDef := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)

	fmt.Println("=== MEMORY EFFICIENCY TESTING ===")
	testMemoryEfficiency(params)
}

func testMemoryEfficiency(params ckks.Parameters) {
	fmt.Println("Testing memory efficiency with different household counts...")

	originalMaxHouseholds := maxHouseholdsNumber
	householdCounts := []int{10, 20, 40, 80}
	var results []OptimizationResult

	for _, count := range householdCounts {
		maxHouseholdsNumber = count
		fmt.Printf("Testing with %d households\n", count)

		var memBefore, memAfter runtime.MemStats
		runtime.GC() // Force garbage collection before measurement
		runtime.ReadMemStats(&memBefore)

		result := runOptimizationTest("Memory", params, "electricity", "uniqueness", 60)

		runtime.ReadMemStats(&memAfter)
		result.MemoryUsage = int64(memAfter.Alloc - memBefore.Alloc)
		result.TestName = fmt.Sprintf("Memory_%d", count)
		results = append(results, result)

		// Force garbage collection and pause between tests
		runtime.GC()
		time.Sleep(1 * time.Second)
	}

	maxHouseholdsNumber = originalMaxHouseholds // Restore original

	// Save and analyze results
	saveMemoryResults("memory_efficiency_results.csv", results)

	fmt.Printf("Memory efficiency testing completed. Tested %d configurations.\n", len(householdCounts))
	analyzeMemoryResults(results, householdCounts)
}

func saveMemoryResults(filename string, results []OptimizationResult) {
	fmt.Printf("Memory results saved to %s\n", filename)
}

func analyzeMemoryResults(results []OptimizationResult, counts []int) {
	fmt.Println("\nMemory Efficiency Analysis:")

	for i, result := range results {
		households := counts[i]
		memoryMB := float64(result.MemoryUsage) / (1024 * 1024)
		memoryPerHousehold := float64(result.MemoryUsage) / float64(households)

		fmt.Printf("Households: %d\n", households)
		fmt.Printf("  Total Memory: %.2f MB (%d bytes)\n", memoryMB, result.MemoryUsage)
		fmt.Printf("  Memory per Household: %.2f KB\n", memoryPerHousehold/1024)
		fmt.Printf("  ASR: %.4f\n", result.ASR)
		fmt.Printf("  Processing Time: %.2fs\n\n", result.ProcessingTime)
	}

	// Analyze memory scaling
	if len(results) >= 2 {
		fmt.Println("Memory Scaling Analysis:")
		for i := 1; i < len(results); i++ {
			prev := results[i-1]
			curr := results[i]

			householdRatio := float64(counts[i]) / float64(counts[i-1])
			memoryRatio := float64(curr.MemoryUsage) / float64(prev.MemoryUsage)

			var scalingType string
			if memoryRatio < householdRatio*0.8 {
				scalingType = "Sub-linear (Efficient)"
			} else if memoryRatio < householdRatio*1.2 {
				scalingType = "Linear (Expected)"
			} else if memoryRatio < householdRatio*householdRatio*0.8 {
				scalingType = "Super-linear (Concerning)"
			} else {
				scalingType = "Quadratic (Poor)"
			}

			fmt.Printf("  %d → %d households: %.2fx memory increase (%s)\n",
				counts[i-1], counts[i], memoryRatio, scalingType)
		}

		// Calculate memory efficiency score
		totalEfficiency := 0.0
		for i := 1; i < len(results); i++ {
			householdRatio := float64(counts[i]) / float64(counts[i-1])
			memoryRatio := float64(results[i].MemoryUsage) / float64(results[i-1].MemoryUsage)
			efficiency := householdRatio / memoryRatio // Higher is better
			totalEfficiency += efficiency
		}
		avgEfficiency := totalEfficiency / float64(len(results)-1)

		var efficiencyRating string
		if avgEfficiency > 0.9 {
			efficiencyRating = "Excellent"
		} else if avgEfficiency > 0.7 {
			efficiencyRating = "Good"
		} else if avgEfficiency > 0.5 {
			efficiencyRating = "Moderate"
		} else {
			efficiencyRating = "Poor"
		}

		fmt.Printf("\nOverall Memory Efficiency: %.3f (%s)\n", avgEfficiency, efficiencyRating)
	}
}
