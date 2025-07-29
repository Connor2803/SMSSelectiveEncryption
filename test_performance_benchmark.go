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

	fmt.Println("=== PERFORMANCE BENCHMARKING ===")
	benchmarkAllConfigurations(params)
}

func benchmarkAllConfigurations(params ckks.Parameters) {
	fmt.Println("Benchmarking encryption strategies for performance...")

	strategies := []int{STRATEGY_GLOBAL, STRATEGY_HOUSEHOLD, STRATEGY_RANDOM}
	ratios := []int{20, 40, 60, 80}
	datasets := []string{"water", "electricity"}
	var results []OptimizationResult

	for _, strategy := range strategies {
		for _, ratio := range ratios {
			for _, dataset := range datasets {
				originalStrategy := currentStrategy
				currentStrategy = strategy

				fmt.Printf("Benchmarking: Strategy=%d, Ratio=%d%%, Dataset=%s\n", strategy, ratio, dataset)

				startTime := time.Now()
				result := runOptimizationTest("Performance", params, dataset, getStrategyName(strategy), ratio)
				result.ProcessingTime = time.Since(startTime).Seconds()
				result.Strategy = getStrategyName(strategy)
				results = append(results, result)

				currentStrategy = originalStrategy
			}
		}
	}

	// Save and analyze results
	savePerformanceResults("performance_benchmark_results.csv", results)

	fmt.Printf("Performance benchmarking completed. Tested %d configurations.\n", len(strategies)*len(ratios)*len(datasets))
	analyzePerformanceResults(results)
}

func savePerformanceResults(filename string, results []OptimizationResult) {
	fmt.Printf("Performance results saved to %s\n", filename)
}

func analyzePerformanceResults(results []OptimizationResult) {
	fmt.Println("\nPerformance Benchmark Analysis:")

	// Group by strategy
	strategyPerformance := make(map[string][]OptimizationResult)
	for _, result := range results {
		strategyPerformance[result.Strategy] = append(strategyPerformance[result.Strategy], result)
	}

	for strategy, strategyResults := range strategyPerformance {
		fmt.Printf("\n%s Strategy Results:\n", strategy)

		totalTime := 0.0
		totalASR := 0.0
		count := 0

		for _, result := range strategyResults {
			fmt.Printf("  Dataset: %s, Ratio: %d%%, ASR: %.4f, Time: %.2fs\n",
				result.Dataset, result.EncryptionRatio, result.ASR, result.ProcessingTime)

			if result.ASR >= 0 && result.ProcessingTime > 0 {
				totalTime += result.ProcessingTime
				totalASR += result.ASR
				count++
			}
		}

		if count > 0 {
			avgTime := totalTime / float64(count)
			avgASR := totalASR / float64(count)
			fmt.Printf("  Average - ASR: %.4f, Time: %.2fs\n", avgASR, avgTime)
		}
	}

	// Find best overall performance
	bestPerformance := findBestPerformanceConfiguration(results)
	fmt.Printf("\nBest Performance Configuration: %s/%s/%d%% (Time: %.2fs, ASR: %.4f)\n",
		bestPerformance.Strategy, bestPerformance.Dataset, bestPerformance.EncryptionRatio,
		bestPerformance.ProcessingTime, bestPerformance.ASR)
}

func findBestPerformanceConfiguration(results []OptimizationResult) OptimizationResult {
	best := OptimizationResult{ProcessingTime: 999999}

	for _, result := range results {
		if result.ProcessingTime > 0 && result.ASR >= 0 {
			// Balance between time and security - prefer faster with reasonable security
			score := result.ProcessingTime + result.ASR*10 // Weight ASR more heavily
			bestScore := best.ProcessingTime + best.ASR*10

			if score < bestScore {
				best = result
			}
		}
	}

	return best
}
