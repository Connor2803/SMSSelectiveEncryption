package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

func main() {
	paramsDef := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)

	fmt.Println("=== ADVANCED SECURITY ANALYSIS ===")
	testAdvancedSecurityScenarios(params)
}

func testAdvancedSecurityScenarios(params ckks.Parameters) {
	fmt.Println("Testing against different attacker capabilities...")

	attackScenarios := []struct {
		name        string
		uniqueATD   int
		minMatched  int
		attackLoops int
	}{
		{"Basic Attacker", 0, 100, 5},
		{"Statistical Attacker", 0, 85, 10},
		{"Advanced Attacker", 1, 80, 15},
		{"Expert Attacker", 1, 75, 20},
	}

	originalUniqueATD := uniqueATD
	originalMinMatched := min_percent_matched
	originalMaxLoop := max_attackLoop
	var results []OptimizationResult

	for _, scenario := range attackScenarios {
		fmt.Printf("Testing against: %s\n", scenario.name)

		uniqueATD = scenario.uniqueATD
		min_percent_matched = scenario.minMatched
		max_attackLoop = scenario.attackLoops

		result := runOptimizationTest("Security", params, "electricity", "uniqueness", 60)
		result.TestName = fmt.Sprintf("Security_%s", scenario.name)
		results = append(results, result)
	}

	// Restore originals
	uniqueATD = originalUniqueATD
	min_percent_matched = originalMinMatched
	max_attackLoop = originalMaxLoop

	// Save and analyze results
	saveSecurityResults("security_analysis_results.csv", results)

	fmt.Printf("Security analysis completed. Tested %d attack scenarios.\n", len(attackScenarios))
	analyzeSecurityResults(results, attackScenarios)
}

func saveSecurityResults(filename string, results []OptimizationResult) {
	fmt.Printf("Security results saved to %s\n", filename)
}

func analyzeSecurityResults(results []OptimizationResult, scenarios []struct {
	name        string
	uniqueATD   int
	minMatched  int
	attackLoops int
}) {
	fmt.Println("\nSecurity Analysis Results:")

	for i, result := range results {
		scenario := scenarios[i]

		var threatLevel string
		if result.ASR < 0.1 {
			threatLevel = "SECURE"
		} else if result.ASR < 0.3 {
			threatLevel = "MODERATE"
		} else if result.ASR < 0.6 {
			threatLevel = "VULNERABLE"
		} else {
			threatLevel = "HIGH RISK"
		}

		fmt.Printf("%s:\n", scenario.name)
		fmt.Printf("  Unique ATD Required: %v\n", scenario.uniqueATD == 1)
		fmt.Printf("  Matching Threshold: %d%%\n", scenario.minMatched)
		fmt.Printf("  Attack Iterations: %d\n", scenario.attackLoops)
		fmt.Printf("  ASR: %.4f (%s)\n", result.ASR, threatLevel)
		fmt.Printf("  Processing Time: %.2fs\n\n", result.ProcessingTime)
	}

	// Overall security assessment
	worstASR := 0.0
	for _, result := range results {
		if result.ASR > worstASR {
			worstASR = result.ASR
		}
	}

	var overallSecurity string
	if worstASR < 0.2 {
		overallSecurity = "STRONG - Resistant to advanced attacks"
	} else if worstASR < 0.4 {
		overallSecurity = "MODERATE - Some vulnerability to expert attackers"
	} else {
		overallSecurity = "WEAK - Significant vulnerability detected"
	}

	fmt.Printf("Overall Security Assessment: %s (Worst ASR: %.4f)\n", overallSecurity, worstASR)
}
