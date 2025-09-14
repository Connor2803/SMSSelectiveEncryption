package main

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

// Test data - realistic electricity consumption pattern (kWh per hour)
// Includes distinctive patterns that should be targetable by breaking methods
var testElectricityData = [][]float64{
	// House 1: High morning/evening usage pattern
	{2.1, 2.0, 1.9, 1.8, 1.7, 2.5, 3.2, 4.1, 3.8, 3.2, 2.8, 2.5, 2.3, 2.1, 2.4, 2.8, 3.5, 4.2, 4.8, 3.9},
	// House 2: Similar pattern (should be vulnerable to sequence attacks)
	{2.0, 1.9, 1.8, 1.7, 1.6, 2.4, 3.1, 4.0, 3.7, 3.1, 2.7, 2.4, 2.2, 2.0, 2.3, 2.7, 3.4, 4.1, 4.7, 3.8},
	// House 3: Different but has repeating subsequences
	{1.5, 1.5, 1.4, 1.3, 1.2, 1.8, 2.5, 3.2, 2.9, 2.4, 2.0, 1.8, 1.6, 1.5, 1.7, 2.1, 2.8, 3.5, 4.1, 3.3},
	// House 4: Very distinctive pattern (high uniqueness)
	{0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.8, 2.5, 2.0, 1.6, 1.3, 1.1, 0.9, 1.2, 1.6, 2.3, 3.0, 3.6, 2.8},
}

func TestSlidingWindowBreaking(t *testing.T) {
	fmt.Println("=== Testing Sliding Window Breaking ===")

	tolerance := 0.3
	original := copyMatrix(testElectricityData)

	fmt.Printf("Original data (first house): %.3f\n", original[0])

	// Apply sliding window breaking
	broken := applyWindowBasedBreaking(original, tolerance)

	fmt.Printf("Broken data (first house):   %.3f\n", broken[0])

	// Test 1: Total preservation
	for h := 0; h < len(original); h++ {
		origSum := sum(original[h])
		brokenSum := sum(broken[h])
		diff := math.Abs(origSum - brokenSum)

		if diff > 0.001 { // Allow tiny floating point errors
			t.Errorf("House %d: Total not preserved. Original: %.6f, Broken: %.6f, Diff: %.6f",
				h, origSum, brokenSum, diff)
		}
	}
	fmt.Println("✓ Total preservation test passed")

	// Test 2: Values actually changed (tolerance > 0 should cause changes)
	totalChanges := 0
	for h := 0; h < len(original); h++ {
		for i := 0; i < len(original[h]); i++ {
			if math.Abs(original[h][i]-broken[h][i]) > 0.001 {
				totalChanges++
			}
		}
	}

	if tolerance > 0 && totalChanges == 0 {
		t.Error("No values changed despite positive tolerance")
	}
	fmt.Printf("✓ Changed %d values with tolerance %.1f\n", totalChanges, tolerance)

	// Test 3: Changes are bounded by tolerance
	for h := 0; h < len(original); h++ {
		origSum := sum(original[h])
		redistributionBudget := origSum * tolerance

		actualRedistribution := 0.0
		for i := 0; i < len(original[h]); i++ {
			actualRedistribution += math.Abs(original[h][i] - broken[h][i])
		}

		if actualRedistribution > redistributionBudget+0.001 {
			t.Errorf("House %d: Redistribution %.6f exceeds budget %.6f",
				h, actualRedistribution, redistributionBudget)
		}
	}
	fmt.Println("✓ Redistribution budget respected")

	// Test 4: Zero tolerance should produce no changes
	fmt.Println("\n--- Testing zero tolerance ---")
	zeroTolBroken := applyWindowBasedBreaking(original, 0.0)

	for h := 0; h < len(original); h++ {
		for i := 0; i < len(original[h]); i++ {
			if math.Abs(original[h][i]-zeroTolBroken[h][i]) > 0.0001 {
				t.Errorf("Zero tolerance changed value at [%d][%d]: %.6f → %.6f",
					h, i, original[h][i], zeroTolBroken[h][i])
			}
		}
	}
	fmt.Println("✓ Zero tolerance produces no changes")
}

func TestAdaptivePatternBreaking(t *testing.T) {
	fmt.Println("\n=== Testing Adaptive Pattern Breaking ===")

	tolerance := 0.3
	original := copyMatrix(testElectricityData)

	fmt.Printf("Original data (first house): %.3f\n", original[0])

	// Apply adaptive pattern breaking
	broken := applyAdaptivePatternBreaking(original, tolerance)

	fmt.Printf("Broken data (first house):   %.3f\n", broken[0])

	// Test 1: Total preservation
	for h := 0; h < len(original); h++ {
		origSum := sum(original[h])
		brokenSum := sum(broken[h])
		diff := math.Abs(origSum - brokenSum)

		if diff > 0.001 {
			t.Errorf("House %d: Total not preserved. Original: %.6f, Broken: %.6f, Diff: %.6f",
				h, origSum, brokenSum, diff)
		}
	}
	fmt.Println("✓ Total preservation test passed")

	// Test 2: Values actually changed
	totalChanges := 0
	maxChange := 0.0
	for h := 0; h < len(original); h++ {
		for i := 0; i < len(original[h]); i++ {
			change := math.Abs(original[h][i] - broken[h][i])
			if change > 0.001 {
				totalChanges++
				if change > maxChange {
					maxChange = change
				}
			}
		}
	}

	if tolerance > 0 && totalChanges == 0 {
		t.Error("No values changed despite positive tolerance")
	}
	fmt.Printf("✓ Changed %d values, max change: %.6f\n", totalChanges, maxChange)

	// Test 3: Zero tolerance should produce no changes
	fmt.Println("\n--- Testing zero tolerance ---")
	zeroTolBroken := applyAdaptivePatternBreaking(original, 0.0)

	for h := 0; h < len(original); h++ {
		for i := 0; i < len(original[h]); i++ {
			if math.Abs(original[h][i]-zeroTolBroken[h][i]) > 0.0001 {
				t.Errorf("Zero tolerance changed value at [%d][%d]: %.6f → %.6f",
					h, i, original[h][i], zeroTolBroken[h][i])
			}
		}
	}
	fmt.Println("✓ Zero tolerance produces no changes")
}

func TestSequenceMatchingVulnerability(t *testing.T) {
	fmt.Println("\n=== Testing Sequence Matching Vulnerability ===")

	original := copyMatrix(testElectricityData)
	tolerance := 0.4

	// Test both methods
	slidingBroken := applyWindowBasedBreaking(original, tolerance)
	adaptiveBroken := applyAdaptivePatternBreaking(original, tolerance)

	// Count exact sequence matches (length 4) in original data
	seqLength := 4
	originalMatches := countSequenceMatches(original, seqLength)
	slidingMatches := countSequenceMatches(slidingBroken, seqLength)
	adaptiveMatches := countSequenceMatches(adaptiveBroken, seqLength)

	fmt.Printf("4-length sequence matches:\n")
	fmt.Printf("  Original:  %d matches\n", originalMatches)
	fmt.Printf("  Sliding:   %d matches", slidingMatches)
	if originalMatches > 0 {
		fmt.Printf(" (%.1f%% reduction)", 100.0*(float64(originalMatches-slidingMatches)/float64(originalMatches)))
	}
	fmt.Println()
	fmt.Printf("  Adaptive:  %d matches", adaptiveMatches)
	if originalMatches > 0 {
		fmt.Printf(" (%.1f%% reduction)", 100.0*(float64(originalMatches-adaptiveMatches)/float64(originalMatches)))
	}
	fmt.Println()

	// Both methods should reduce sequence matches (or at least not increase them)
	if slidingMatches > originalMatches {
		t.Errorf("Sliding method increased sequence matches: %d → %d", originalMatches, slidingMatches)
	}

	if adaptiveMatches > originalMatches {
		t.Errorf("Adaptive method increased sequence matches: %d → %d", originalMatches, adaptiveMatches)
	}

	fmt.Println("✓ Both methods reduced/maintained sequence vulnerability")
}

func TestPatternDisruption(t *testing.T) {
	fmt.Println("\n=== Testing Pattern Disruption ===")

	original := copyMatrix(testElectricityData)
	tolerance := 0.3

	slidingBroken := applyWindowBasedBreaking(original, tolerance)
	adaptiveBroken := applyAdaptivePatternBreaking(original, tolerance)

	// Count unique 3-patterns
	originalPatterns := countUniquePatterns(original, 3)
	slidingPatterns := countUniquePatterns(slidingBroken, 3)
	adaptivePatterns := countUniquePatterns(adaptiveBroken, 3)

	fmt.Printf("Unique 3-patterns:\n")
	fmt.Printf("  Original:  %d patterns\n", originalPatterns)
	fmt.Printf("  Sliding:   %d patterns\n", slidingPatterns)
	fmt.Printf("  Adaptive:  %d patterns\n", adaptivePatterns)

	// Calculate pattern disruption (how many original patterns were broken)
	slidingDisruption := calculatePatternDisruption(original, slidingBroken, 3)
	adaptiveDisruption := calculatePatternDisruption(original, adaptiveBroken, 3)

	fmt.Printf("Pattern disruption:\n")
	fmt.Printf("  Sliding:   %.1f%%\n", slidingDisruption*100)
	fmt.Printf("  Adaptive:  %.1f%%\n", adaptiveDisruption*100)

	// Test that methods actually disrupted patterns
	if slidingDisruption <= 0 && tolerance > 0 {
		t.Logf("Warning: Sliding method caused no pattern disruption with tolerance %.1f", tolerance)
	}

	if adaptiveDisruption <= 0 && tolerance > 0 {
		t.Logf("Warning: Adaptive method caused no pattern disruption with tolerance %.1f", tolerance)
	}

	fmt.Println("✓ Pattern disruption analysis complete")
}

func TestMethodComparison(t *testing.T) {
	fmt.Println("\n=== Method Comparison Analysis ===")

	original := copyMatrix(testElectricityData)
	tolerances := []float64{0.1, 0.2, 0.3, 0.4, 0.5}

	fmt.Printf("Tolerance | Sliding Matches | Adaptive Matches | Better Method\n")
	fmt.Printf("----------|-----------------|------------------|-------------\n")

	for _, tol := range tolerances {
		slidingBroken := applyWindowBasedBreaking(original, tol)
		adaptiveBroken := applyAdaptivePatternBreaking(original, tol)

		slidingMatches := countSequenceMatches(slidingBroken, 4)
		adaptiveMatches := countSequenceMatches(adaptiveBroken, 4)

		better := "Tie"
		if slidingMatches < adaptiveMatches {
			better = "Sliding"
		} else if adaptiveMatches < slidingMatches {
			better = "Adaptive"
		}

		fmt.Printf("   %.1f    |       %2d        |        %2d        | %s\n",
			tol, slidingMatches, adaptiveMatches, better)
	}
}

func TestFuzzingActuallyWorks(t *testing.T) {
	original := [][]float64{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}}
	fuzzed := applyAdaptivePatternBreaking(original, 0.1)

	fmt.Printf("Original: %v\n", original[0])
	fmt.Printf("Fuzzed:   %v\n", fuzzed[0])

	changes := 0
	for i := 0; i < len(original[0]); i++ {
		if math.Abs(original[0][i]-fuzzed[0][i]) > 0.001 {
			changes++
			fmt.Printf("Change at index %d: %.6f → %.6f\n", i, original[0][i], fuzzed[0][i])
		}
	}

	fmt.Printf("Total changes: %d out of %d\n", changes, len(original[0]))

	if changes == 0 {
		t.Error("No changes made despite tolerance > 0!")
	}
}

func TestSlidingWindowBounds(t *testing.T) {
	fmt.Println("\n=== Testing Sliding Window Bounds ===")

	original := [][]float64{{1.0, 2.0, 3.0, 4.0, 5.0}}

	// Test with non-overlapping windows
	result := applyWindowBasedBreaking(original, 0.3)

	fmt.Printf("Original: %v\n", original[0])
	fmt.Printf("Result:   %v\n", result[0])

	// Index 4 should be unchanged (value should still be 5.0)
	if math.Abs(result[0][4]-5.0) > 0.001 {
		t.Errorf("Index 4 was modified: expected 5.0, got %.6f", result[0][4])
	} else {
		fmt.Printf("✓ Index 4 correctly unchanged: %.6f\n", result[0][4])
	}

	// Check total preservation
	origSum := sum(original[0])
	resultSum := sum(result[0])
	preservation := 1.0 - math.Abs(origSum-resultSum)/origSum

	fmt.Printf("Total preservation: %.6f\n", preservation)

	if preservation < 0.999 {
		t.Errorf("Total not preserved: %.6f", preservation)
	}
}

func TestBreakingFunctionBoundaryConditions(t *testing.T) {
	fmt.Println("\n=== Testing Boundary Conditions ===")

	// Test empty data
	empty := [][]float64{}
	slidingEmpty := applyWindowBasedBreaking(empty, 0.3)
	adaptiveEmpty := applyAdaptivePatternBreaking(empty, 0.3)

	if len(slidingEmpty) != 0 || len(adaptiveEmpty) != 0 {
		t.Error("Empty data should return empty result")
	}

	// Test single value
	single := [][]float64{{5.0}}
	slidingSingle := applyWindowBasedBreaking(single, 0.3)
	adaptiveSingle := applyAdaptivePatternBreaking(single, 0.3)

	if len(slidingSingle) != 1 || len(slidingSingle[0]) != 1 {
		t.Error("Single value data structure changed for sliding")
	}

	if len(adaptiveSingle) != 1 || len(adaptiveSingle[0]) != 1 {
		t.Error("Single value data structure changed for adaptive")
	}

	// Single values might change slightly due to normalization, so allow some tolerance
	if math.Abs(slidingSingle[0][0]-single[0][0]) > 1.0 {
		t.Error("Single value changed too much in sliding method")
	}

	if math.Abs(adaptiveSingle[0][0]-single[0][0]) > 1.0 {
		t.Error("Single value changed too much in adaptive method")
	}

	fmt.Println("✓ Boundary conditions handled correctly")
}

// Test to analyze what's actually happening with attack success rates
func TestAttackSuccessRateImpact(t *testing.T) {
	fmt.Println("\n=== Testing Attack Success Rate Impact ===")

	original := copyMatrix(testElectricityData)
	tolerance := 0.3

	// Create parties from original data
	originalParties := convertDataToParties(original)

	// Set up globals for attack
	globalPartyRows = len(original[0])
	sectionNum = globalPartyRows / sectionSize
	if globalPartyRows%sectionSize != 0 {
		sectionNum++
	}

	// Generate inputs from original data
	generateInputsFromData(originalParties, original)
	intializeEdgeRelated(originalParties)

	// Test with different encryption ratios
	ratios := []int{0, 20, 50, 80}

	fmt.Printf("Encryption | Original ASR | Sliding ASR | Adaptive ASR | Improvement\n")
	fmt.Printf("Ratio      |              |             |              |\n")
	fmt.Printf("-----------|--------------|-------------|--------------|------------\n")

	for _, ratio := range ratios {
		// Test original data
		originalTestParties := convertDataToParties(original)
		generateInputsFromData(originalTestParties, original)
		encryptionRatio = ratio
		intializeEdgeRelated(originalTestParties)
		processGreedyEncryptionSimulation(originalTestParties)
		originalASR := runLimitedAttackTest(originalTestParties, 100)

		// Test sliding method
		slidingBroken := applyWindowBasedBreaking(original, tolerance)
		slidingParties := convertDataToParties(slidingBroken)
		generateInputsFromData(slidingParties, slidingBroken)
		intializeEdgeRelated(slidingParties)
		processGreedyEncryptionSimulation(slidingParties)
		slidingASR := runLimitedAttackTest(slidingParties, 100)

		// Test adaptive method
		adaptiveBroken := applyAdaptivePatternBreaking(original, tolerance)
		adaptiveParties := convertDataToParties(adaptiveBroken)
		generateInputsFromData(adaptiveParties, adaptiveBroken)
		intializeEdgeRelated(adaptiveParties)
		processGreedyEncryptionSimulation(adaptiveParties)
		adaptiveASR := runLimitedAttackTest(adaptiveParties, 100)

		// Calculate improvements
		slidingImprovement := ((originalASR - slidingASR) / originalASR) * 100.0
		adaptiveImprovement := ((originalASR - adaptiveASR) / originalASR) * 100.0

		fmt.Printf("   %2d%%     |    %.3f     |   %.3f     |    %.3f      | S:%.1f%% A:%.1f%%\n",
			ratio, originalASR, slidingASR, adaptiveASR, slidingImprovement, adaptiveImprovement)
	}

	fmt.Println("✓ Attack success rate impact analysis complete")
}

// New test to understand why methods might work "randomly"
func TestDatasetSensitivity(t *testing.T) {
	fmt.Println("\n=== Testing Dataset Sensitivity ===")

	tolerance := 0.3

	// Test with different data characteristics
	datasets := []struct {
		name string
		data [][]float64
	}{
		{"Original", testElectricityData},
		{"Uniform", generateUniformData(4, 20)},
		{"HighVariance", generateHighVarianceData(4, 20)},
		{"LowVariance", generateLowVarianceData(4, 20)},
	}

	fmt.Printf("Dataset      | Sliding Changes | Adaptive Changes | Sliding Matches | Adaptive Matches\n")
	fmt.Printf("-------------|-----------------|------------------|-----------------|------------------\n")

	for _, dataset := range datasets {
		original := copyMatrix(dataset.data)

		// Apply methods
		slidingBroken := applyWindowBasedBreaking(original, tolerance)
		adaptiveBroken := applyAdaptivePatternBreaking(original, tolerance)

		// Count changes
		slidingChanges := countTotalChanges(original, slidingBroken)
		adaptiveChanges := countTotalChanges(original, adaptiveBroken)

		// Count sequence matches
		slidingMatches := countSequenceMatches(slidingBroken, 4)
		adaptiveMatches := countSequenceMatches(adaptiveBroken, 4)

		fmt.Printf("%-12s |      %3d        |       %3d        |       %2d        |        %2d\n",
			dataset.name, slidingChanges, adaptiveChanges, slidingMatches, adaptiveMatches)
	}

	fmt.Println("✓ Dataset sensitivity analysis complete")
}

// Helper functions

func countSequenceMatches(data [][]float64, seqLength int) int {
	matches := 0
	tolerance := 0.01

	// Compare all possible sequence pairs across households
	for h1 := 0; h1 < len(data); h1++ {
		for i := 0; i <= len(data[h1])-seqLength; i++ {
			seq1 := data[h1][i : i+seqLength]

			// Compare with all other households
			for h2 := h1 + 1; h2 < len(data); h2++ {
				for j := 0; j <= len(data[h2])-seqLength; j++ {
					seq2 := data[h2][j : j+seqLength]

					// Check if sequences match within tolerance
					match := true
					for k := 0; k < seqLength; k++ {
						if math.Abs(seq1[k]-seq2[k]) > tolerance {
							match = false
							break
						}
					}

					if match {
						matches++
					}
				}
			}
		}
	}

	return matches
}

func countUniquePatterns(data [][]float64, patternLength int) int {
	patterns := make(map[string]bool)

	for h := 0; h < len(data); h++ {
		for i := 0; i <= len(data[h])-patternLength; i++ {
			var pattern string
			for j := 0; j < patternLength; j++ {
				if j > 0 {
					pattern += ","
				}
				pattern += fmt.Sprintf("%.2f", data[h][i+j])
			}
			patterns[pattern] = true
		}
	}

	return len(patterns)
}

func calculatePatternDisruption(original, broken [][]float64, patternLength int) float64 {
	// Count how many original patterns still exist in broken data
	originalPatterns := extractPatterns(original, patternLength)
	brokenPatterns := extractPatterns(broken, patternLength)

	if len(originalPatterns) == 0 {
		return 0.0
	}

	preserved := 0
	for pattern := range originalPatterns {
		if brokenPatterns[pattern] {
			preserved++
		}
	}

	disruption := 1.0 - float64(preserved)/float64(len(originalPatterns))
	return math.Max(0.0, disruption)
}

func extractPatterns(data [][]float64, patternLength int) map[string]bool {
	patterns := make(map[string]bool)

	for h := 0; h < len(data); h++ {
		for i := 0; i <= len(data[h])-patternLength; i++ {
			var pattern string
			for j := 0; j < patternLength; j++ {
				if j > 0 {
					pattern += ","
				}
				pattern += fmt.Sprintf("%.2f", data[h][i+j])
			}
			patterns[pattern] = true
		}
	}

	return patterns
}

func countTotalChanges(original, modified [][]float64) int {
	changes := 0
	tolerance := 0.001

	for h := 0; h < len(original); h++ {
		for i := 0; i < len(original[h]); i++ {
			if math.Abs(original[h][i]-modified[h][i]) > tolerance {
				changes++
			}
		}
	}

	return changes
}

func runLimitedAttackTest(parties []*party, loops int) float64 {
	if len(parties) == 0 {
		return 0.0
	}

	successes := 0
	for i := 0; i < loops; i++ {
		success := simulateSimpleAttack(parties)
		if success {
			successes++
		}
	}

	return float64(successes) / float64(loops)
}

func simulateSimpleAttack(parties []*party) bool {
	if len(parties) < 2 {
		return false
	}

	// Simple attack: try to match a random sequence from one party in others
	seqLength := 4
	if len(parties[0].rawInput) < seqLength {
		return false
	}

	// Pick random party and sequence
	targetParty := getRandom(len(parties))
	maxStart := len(parties[targetParty].rawInput) - seqLength
	if maxStart <= 0 {
		return false
	}

	startPos := getRandom(maxStart)
	targetSeq := parties[targetParty].rawInput[startPos : startPos+seqLength]

	// Try to find matching sequence in other parties
	matches := 0
	tolerance := 0.1

	for p := 0; p < len(parties); p++ {
		if p == targetParty {
			continue
		}

		for i := 0; i <= len(parties[p].rawInput)-seqLength; i++ {
			seq := parties[p].rawInput[i : i+seqLength]

			match := true
			for j := 0; j < seqLength; j++ {
				if math.Abs(targetSeq[j]-seq[j]) > tolerance {
					match = false
					break
				}
			}

			if match {
				matches++
				break // Found match in this party
			}
		}
	}

	// Attack succeeds if we found the sequence in exactly one other party
	return matches == 1
}

// Data generation functions for testing
func generateUniformData(houses, points int) [][]float64 {
	data := make([][]float64, houses)
	for h := 0; h < houses; h++ {
		data[h] = make([]float64, points)
		for i := 0; i < points; i++ {
			data[h][i] = 2.0 // Uniform consumption
		}
	}
	return data
}

func generateHighVarianceData(houses, points int) [][]float64 {
	data := make([][]float64, houses)
	for h := 0; h < houses; h++ {
		data[h] = make([]float64, points)
		for i := 0; i < points; i++ {
			// High variance: 0.5 to 5.0
			data[h][i] = 0.5 + float64(getRandom(45))/10.0
		}
	}
	return data
}

func generateLowVarianceData(houses, points int) [][]float64 {
	data := make([][]float64, houses)
	base := 2.0
	for h := 0; h < houses; h++ {
		data[h] = make([]float64, points)
		for i := 0; i < points; i++ {
			// Low variance: 1.8 to 2.2
			variance := (float64(getRandom(40)) - 20.0) / 100.0 // -0.2 to +0.2
			data[h][i] = base + variance
		}
	}
	return data
}

// Benchmark tests
func BenchmarkSlidingWindowBreaking(b *testing.B) {
	original := copyMatrix(testElectricityData)
	tolerance := 0.3

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = applyWindowBasedBreaking(original, tolerance)
	}
}

func BenchmarkAdaptivePatternBreaking(b *testing.B) {
	original := copyMatrix(testElectricityData)
	tolerance := 0.3

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = applyAdaptivePatternBreaking(original, tolerance)
	}
}

// Add this to smalldata_test.go
func TestExactMatchBreaking(t *testing.T) {
	fmt.Println("\n=== Testing Exact Match Breaking ===")

	// Create simple test data
	original := [][]float64{{1.0, 2.0, 3.0, 4.0, 5.0}}

	// Test different tolerance levels
	tolerances := []float64{0.0, 0.1, 0.3, 0.5}

	for _, tol := range tolerances {
		fmt.Printf("\n--- Testing tolerance %.1f ---\n", tol)

		// Apply adaptive fuzzing
		fuzzed := applyAdaptivePatternBreaking(original, tol)

		fmt.Printf("Original: %v\n", original[0])
		fmt.Printf("Fuzzed:   %v\n", fuzzed[0])

		// Check if exact match is broken
		exactMatch := reflect.DeepEqual(original[0], fuzzed[0])
		fmt.Printf("Exact match: %v\n", exactMatch)

		// Calculate total preservation
		origSum := sum(original[0])
		fuzzSum := sum(fuzzed[0])
		preservation := 1.0 - math.Abs(origSum-fuzzSum)/origSum
		fmt.Printf("Total preserved: %.6f\n", preservation)

		// Test expectations
		if tol == 0.0 {
			if !exactMatch {
				t.Errorf("Tolerance 0.0 should preserve exact match, but got different values")
			}
		} else {
			if exactMatch {
				t.Errorf("Tolerance %.1f should break exact match, but values are identical", tol)
			}
			if preservation < 0.999 {
				t.Errorf("Total not preserved: expected >99.9%%, got %.3f%%", preservation*100)
			}
		}
	}

	// Test sliding window too
	fmt.Printf("\n--- Testing Sliding Window ---\n")
	slidingFuzzed := applyWindowBasedBreaking(original, 0.3)
	exactMatch := reflect.DeepEqual(original[0], slidingFuzzed[0])
	fmt.Printf("Original: %v\n", original[0])
	fmt.Printf("Sliding:  %v\n", slidingFuzzed[0])
	fmt.Printf("Exact match broken: %v\n", !exactMatch)

	if exactMatch {
		t.Error("Sliding window with tolerance 0.3 should break exact match")
	}
}

func TestAttackDataFlow(t *testing.T) {
	fmt.Println("\n=== Testing Attack Data Flow ===")

	// Simulate the data flow issue
	original := [][]float64{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}}

	// 1. Apply fuzzing
	fuzzed := applyAdaptivePatternBreaking(original, 0.3)
	fmt.Printf("Original: %v\n", original[0])
	fmt.Printf("Fuzzed:   %v\n", fuzzed[0])

	// 2. Simulate selective encryption (50% ratio)
	encrypted := make([]float64, len(fuzzed[0]))
	copy(encrypted, fuzzed[0])

	// Encrypt every other value
	for i := 0; i < len(encrypted); i += 2 {
		encrypted[i] = -0.1 // Encrypted marker
	}
	fmt.Printf("Encrypted: %v\n", encrypted)

	// 3. What attacker should see vs what they currently see
	fmt.Printf("\nAttacker should see: %v (fuzzed values in unencrypted positions)\n", encrypted)
	fmt.Printf("Attacker currently sees: original values in unencrypted positions (BUG!)\n")

	// 4. Test exact matching on different scenarios
	attackerKnows := original[0][1:5] // Attacker knows positions 1-4 from original

	// Scenario A: Attack against original data (should succeed)
	unencryptedOriginal := original[0][1:5]
	matchOriginal := reflect.DeepEqual(attackerKnows, unencryptedOriginal)
	fmt.Printf("Match against original: %v\n", matchOriginal)

	// Scenario B: Attack against fuzzed data (should fail)
	unencryptedFuzzed := fuzzed[0][1:5]
	matchFuzzed := reflect.DeepEqual(attackerKnows, unencryptedFuzzed)
	fmt.Printf("Match against fuzzed: %v\n", matchFuzzed)

	if !matchOriginal {
		t.Error("Attack should succeed against original data")
	}
	if matchFuzzed {
		t.Error("Attack should fail against fuzzed data")
	}
}
