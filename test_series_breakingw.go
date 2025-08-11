package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

func main() {
	paramsDef := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)

	// Parse command line arguments
	runs := 1
	if len(os.Args) > 1 {
		if parsedRuns, err := strconv.Atoi(os.Args[1]); err == nil && parsedRuns > 0 {
			runs = parsedRuns
		} else {
			//fmt.printf("Invalid number of runs: %s. Using default of 1.\n", os.Args[1])
		}
	}

	//fmt.printf("=== SERIES BREAKING ALGORITHM TEST ===\n")
	//fmt.printf("Running %d test iteration(s)...\n\n", runs)

	testSeriesBreakingAlgorithms(params, runs)
}

func testSeriesBreakingAlgorithms(params ckks.Parameters, runs int) {
	//fmt.println("Testing series breaking algorithms for electricity data...")

	// Load real electricity dataset
	fileList := getFileList("electricity")
	if fileList == nil {
		//fmt.println("Error: Could not load electricity dataset")
		return
	}

	// Convert real data to our format
	originalData := loadRealElectricityData(fileList)

	// Test both approaches
	approaches := []struct {
		name     string
		function func([][]float64, float64) [][]float64
	}{
		{"Sliding Window Breaking", applyWindowBasedBreaking},
		{"Adaptive Pattern Breaking", applyAdaptivePatternBreaking},
	}

	// Store all results for CSV output
	var allResults []SeriesBreakingResult

	for _, approach := range approaches {
		//fmt.printf("\n--- Testing %s ---\n", approach.name)

		// Test with different tolerance levels
		tolerances := []float64{0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9} // 45%, 50%, 55%, 60%, 65%, 70%, 75% variance allowed

		for _, tolerance := range tolerances {
			fmt.Printf("Testing with %.1f%% tolerance...\n", tolerance*100)

			// Run multiple iterations and collect results
			var runResults []SeriesBreakingResult

			for run := 1; run <= runs; run++ {
				if runs > 1 {
					//fmt.Printf("  Run %d/%d:\n", run, runs)
				}

				// Apply series breaking
				brokenData := approach.function(originalData, tolerance)

				// Test effectiveness
				result := testSeriesBreakingEffectiveness(params, originalData, brokenData, approach.name, tolerance)
				runResults = append(runResults, result)

				if runs == 1 {
					printSeriesBreakingResults(result)
				}
			}

			// Calculate average if multiple runs
			if runs > 1 {
				avgResult := calculateAverageResults(runResults)
				//fmt.printf("  Average results over %d runs:\n", runs)
				printSeriesBreakingResults(avgResult)
				allResults = append(allResults, avgResult)
			} else {
				allResults = append(allResults, runResults[0])
			}
		}
	}

	// Save results to CSV
	csvFilename := generateCSVFilename(runs)
	saveResultsToCSV(allResults, csvFilename, runs)
	//fmt.printf("\n✓ Results saved to: %s\n", csvFilename)
}

func calculateAverageResults(results []SeriesBreakingResult) SeriesBreakingResult {
	if len(results) == 0 {
		return SeriesBreakingResult{}
	}

	avg := SeriesBreakingResult{
		ApproachName: results[0].ApproachName,
		Tolerance:    results[0].Tolerance,
	}

	// Sum all metrics
	var totalOriginalASR, totalBrokenASR float64
	var totalOriginalTime, totalBrokenTime time.Duration
	var totalPreservation, totalEfficiencyGain float64

	for _, result := range results {
		totalOriginalASR += result.OriginalASR
		totalBrokenASR += result.BrokenASR
		totalOriginalTime += result.OriginalTime
		totalBrokenTime += result.BrokenTime
		totalPreservation += result.TotalPreservation
		totalEfficiencyGain += result.EfficiencyGain
	}

	// Calculate averages
	count := float64(len(results))
	avg.OriginalASR = totalOriginalASR / count
	avg.BrokenASR = totalBrokenASR / count
	avg.OriginalTime = time.Duration(float64(totalOriginalTime) / count)
	avg.BrokenTime = time.Duration(float64(totalBrokenTime) / count)
	avg.TotalPreservation = totalPreservation / count
	avg.EfficiencyGain = totalEfficiencyGain / count

	return avg
}

func generateCSVFilename(runs int) string {
	timestamp := time.Now().Format("20060102_150405")
	if runs > 1 {
		return fmt.Sprintf("series_breaking_results_%s_avg_%d_runs.csv", timestamp, runs)
	}
	return fmt.Sprintf("series_breaking_results_%s_single_run.csv", timestamp)
}

func saveResultsToCSV(results []SeriesBreakingResult, filename string, runs int) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create CSV file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header with metadata
	metadata := [][]string{
		{"Series Breaking Algorithm Test Results"},
		{"Generated:", time.Now().Format("2006-01-02 15:04:05")},
		{"Test Runs:", strconv.Itoa(runs)},
		{"Data Type:", "Real Electricity Consumption (80 households, 168 hours)"},
		{""},
	}

	for _, row := range metadata {
		writer.Write(row)
	}

	// Write CSV headers
	headers := []string{
		"Approach",
		"Tolerance_%",
		"Original_ASR",
		"Broken_ASR",
		"ASR_Improvement",
		"ASR_Reduction_%",
		"Original_Time_s",
		"Broken_Time_s",
		"Total_Preservation",
		"Total_Preservation_%",
		"Efficiency_Gain",
		"Efficiency_Gain_%",
		"Success_Status",
	}
	writer.Write(headers)

	// Write data rows
	for _, result := range results {
		asrImprovement := result.OriginalASR - result.BrokenASR
		asrReductionPercent := 0.0
		if result.OriginalASR > 0 {
			asrReductionPercent = (asrImprovement / result.OriginalASR) * 100
		}

		successStatus := "FAILED"
		if result.BrokenASR < result.OriginalASR && result.TotalPreservation > 0.95 {
			successStatus = "SUCCESS"
		} else if result.TotalPreservation < 0.95 {
			successStatus = "WARNING"
		}

		row := []string{
			result.ApproachName,
			fmt.Sprintf("%.1f", result.Tolerance*100),
			fmt.Sprintf("%.6f", result.OriginalASR),
			fmt.Sprintf("%.6f", result.BrokenASR),
			fmt.Sprintf("%.6f", asrImprovement),
			fmt.Sprintf("%.2f", asrReductionPercent),
			fmt.Sprintf("%.3f", result.OriginalTime.Seconds()),
			fmt.Sprintf("%.3f", result.BrokenTime.Seconds()),
			fmt.Sprintf("%.6f", result.TotalPreservation),
			fmt.Sprintf("%.2f", result.TotalPreservation*100),
			fmt.Sprintf("%.6f", result.EfficiencyGain),
			fmt.Sprintf("%.2f", result.EfficiencyGain*100),
			successStatus,
		}
		writer.Write(row)
	}

	// Write summary statistics if multiple approaches
	writer.Write([]string{}) // Empty row
	writer.Write([]string{"Summary Statistics:"})

	// Group results by approach
	approachResults := make(map[string][]SeriesBreakingResult)
	for _, result := range results {
		approachResults[result.ApproachName] = append(approachResults[result.ApproachName], result)
	}

	for approach, approachData := range approachResults {
		writer.Write([]string{})
		writer.Write([]string{fmt.Sprintf("%s Summary:", approach)})

		// Calculate approach averages
		var avgASRImprovement, avgPreservation, avgEfficiency float64
		successCount := 0

		for _, result := range approachData {
			avgASRImprovement += (result.OriginalASR - result.BrokenASR)
			avgPreservation += result.TotalPreservation
			avgEfficiency += result.EfficiencyGain
			if result.BrokenASR < result.OriginalASR && result.TotalPreservation > 0.95 {
				successCount++
			}
		}

		count := len(approachData)
		if count > 0 {
			avgASRImprovement /= float64(count)
			avgPreservation /= float64(count)
			avgEfficiency /= float64(count)
		}

		writer.Write([]string{"", "Average ASR Improvement:", fmt.Sprintf("%.6f", avgASRImprovement)})
		writer.Write([]string{"", "Average Preservation:", fmt.Sprintf("%.2f%%", avgPreservation*100)})
		writer.Write([]string{"", "Average Efficiency Gain:", fmt.Sprintf("%.2f%%", avgEfficiency*100)})
		writer.Write([]string{"", "Success Rate:", fmt.Sprintf("%d/%d (%.1f%%)", successCount, count, float64(successCount)/float64(count)*100)})
	}

	return nil
}

func loadRealElectricityData(fileList []string) [][]float64 {
	data := make([][]float64, len(fileList))

	//fmt.printf("Loading %d electricity dataset files...\n", len(fileList))

	for i, filename := range fileList {
		householdData := resizeCSV(filename)
		// Limit to reasonable size for testing
		maxRows := 168 // 1 week
		if len(householdData) > maxRows {
			householdData = householdData[:maxRows]
		}
		data[i] = householdData
	}

	//fmt.printf("Successfully loaded data for %d households\n", len(data))
	return data
}

// Approach 1: Sliding Window Breaking
func applyWindowBasedBreaking(data [][]float64, tolerance float64) [][]float64 {
	brokenData := make([][]float64, len(data))

	for h := 0; h < len(data); h++ {
		household := make([]float64, len(data[h]))
		copy(household, data[h])

		windowSize := 4 // Break patterns in 4-hour windows

		for i := 0; i <= len(household)-windowSize; i += windowSize {
			end := i + windowSize
			if end > len(household) {
				end = len(household)
			}

			// Calculate window total
			windowTotal := 0.0
			for j := i; j < end; j++ {
				windowTotal += household[j]
			}

			// Break unique sequences while maintaining total
			breakUniqueSequence(household[i:end], windowTotal, tolerance)
		}

		brokenData[h] = household
	}

	return brokenData
}

// Approach 2: Adaptive Pattern Breaking
func applyAdaptivePatternBreaking(data [][]float64, tolerance float64) [][]float64 {
	brokenData := make([][]float64, len(data))

	for h := 0; h < len(data); h++ {
		household := make([]float64, len(data[h]))
		copy(household, data[h])

		// Identify unique patterns and break them
		uniquePatterns := identifyUniquePatterns(household)

		for _, pattern := range uniquePatterns {
			// Calculate total for this pattern
			patternTotal := 0.0
			for j := pattern.start; j <= pattern.end; j++ {
				patternTotal += household[j]
			}

			// Apply adaptive breaking based on pattern characteristics
			applyAdaptiveBreaking(household[pattern.start:pattern.end+1], patternTotal, tolerance, pattern.uniqueness)
		}

		brokenData[h] = household
	}

	return brokenData
}

type UniquePattern struct {
	start      int
	end        int
	uniqueness float64
}

func identifyUniquePatterns(data []float64) []UniquePattern {
	patterns := []UniquePattern{}
	minPatternLength := 3

	for i := 0; i <= len(data)-minPatternLength; i++ {
		for length := minPatternLength; length <= 8 && i+length <= len(data); length++ {
			pattern := data[i : i+length]
			uniqueness := calculatePatternUniqueness(pattern, data)

			if uniqueness > 0.7 { // High uniqueness threshold
				patterns = append(patterns, UniquePattern{
					start:      i,
					end:        i + length - 1,
					uniqueness: uniqueness,
				})
				i += length - 1 // Skip overlapping patterns
				break
			}
		}
	}

	return patterns
}

func calculatePatternUniqueness(pattern []float64, fullData []float64) float64 {
	matches := 0
	totalPossible := len(fullData) - len(pattern) + 1

	for i := 0; i <= len(fullData)-len(pattern); i++ {
		isMatch := true
		for j := 0; j < len(pattern); j++ {
			if math.Abs(fullData[i+j]-pattern[j]) > 0.01 {
				isMatch = false
				break
			}
		}
		if isMatch {
			matches++
		}
	}

	if totalPossible == 0 {
		return 0
	}

	return 1.0 - (float64(matches) / float64(totalPossible))
}

func breakUniqueSequence(window []float64, originalTotal, tolerance float64) {
	if len(window) < 2 {
		return
	}

	maxVariance := originalTotal * tolerance
	rand.Seed(time.Now().UnixNano())

	// Create redistribution amounts
	redistributions := make([]float64, len(window))
	totalRedist := 0.0

	// Generate random redistributions
	for i := 0; i < len(window)-1; i++ {
		maxAmount := math.Min(maxVariance/float64(len(window)), window[i]*0.3)
		redistribution := (rand.Float64()*2 - 1) * maxAmount
		redistributions[i] = redistribution
		totalRedist += redistribution
	}

	// Ensure total remains the same
	redistributions[len(window)-1] = -totalRedist

	// Apply redistributions
	for i := 0; i < len(window); i++ {
		newValue := window[i] + redistributions[i]
		if newValue >= 0.1 { // Ensure positive values
			window[i] = newValue
		}
	}
}

func applyAdaptiveBreaking(window []float64, originalTotal, tolerance, uniqueness float64) {
	if len(window) < 2 {
		return
	}

	// More aggressive breaking for higher uniqueness
	adaptiveTolerance := tolerance * (1.0 + uniqueness)
	maxVariance := originalTotal * adaptiveTolerance

	rand.Seed(time.Now().UnixNano())

	// Use Gaussian redistribution for more natural patterns
	redistributions := make([]float64, len(window))

	for i := 0; i < len(window); i++ {
		// Gaussian distribution with mean 0
		gaussian := rand.NormFloat64() * (maxVariance / float64(len(window)) / 3.0)
		redistributions[i] = gaussian
	}

	// Normalize to maintain total
	redistSum := 0.0
	for _, r := range redistributions {
		redistSum += r
	}

	adjustment := redistSum / float64(len(redistributions))
	for i := range redistributions {
		redistributions[i] -= adjustment
	}

	// Apply redistributions
	for i := 0; i < len(window); i++ {
		newValue := window[i] + redistributions[i]
		if newValue >= 0.1 {
			window[i] = newValue
		}
	}
}

type SeriesBreakingResult struct {
	ApproachName       string
	Tolerance          float64
	OriginalASR        float64
	BrokenASR          float64
	OriginalTime       time.Duration
	BrokenTime         time.Duration
	TotalPreservation  float64
	PatternDistruption float64
	EfficiencyGain     float64
}

func testSeriesBreakingEffectiveness(params ckks.Parameters, original, broken [][]float64, approachName string, tolerance float64) SeriesBreakingResult {
	// Test original data
	//fmt.printf("  Testing original data...")
	originalASR, originalTime := runEncryptionTest(params, original, "electricity")

	// Test broken data
	//fmt.printf("  Testing broken data...")
	brokenASR, brokenTime := runEncryptionTest(params, broken, "electricity")

	// Calculate metrics
	totalPreservation := calculateTotalPreservation(original, broken)
	efficiencyGain := calculateEfficiencyGain(originalTime, brokenTime)

	return SeriesBreakingResult{
		ApproachName:      approachName,
		Tolerance:         tolerance,
		OriginalASR:       originalASR,
		BrokenASR:         brokenASR,
		OriginalTime:      originalTime,
		BrokenTime:        brokenTime,
		TotalPreservation: totalPreservation,
		EfficiencyGain:    efficiencyGain,
	}
}

func runEncryptionTest(params ckks.Parameters, data [][]float64, dataset string) (float64, time.Duration) {
	// Simplified test - measure ASR and time
	startTime := time.Now()

	//fmt.printf("    Converting %d households to party format...\n", len(data))
	// Convert data to party format
	P := convertDataToParties(data)

	// Set global variables for the test
	currentDataset = DATASET_ELECTRICITY
	encryptionRatio = 60
	transitionEqualityThreshold = ELECTRICITY_TRANSITION_EQUALITY_THRESHOLD

	//fmt.printf("    Generating inputs from data...\n")
	// Process the data directly without loading files
	generateInputsFromData(P, data)
	intializeEdgeRelated(P)

	//fmt.printf("    Running greedy encryption simulation...\n")
	// Simulate greedy encryption process
	processGreedyEncryptionSimulation(P)

	//fmt.printf("    Running series attack simulation...\n")
	// Simulate attack
	asr := simulateSeriesAttack(P)

	elapsedTime := time.Since(startTime)
	//fmt.printf("    Test completed in %.2fs with ASR: %.4f\n", elapsedTime.Seconds(), asr)
	return asr, elapsedTime
}

func generateInputsFromData(P []*party, data [][]float64) {
	if len(data) == 0 || len(data[0]) == 0 {
		//fmt.println("    Error: No data to process")
		return
	}

	globalPartyRows = len(data[0])
	sectionNum = globalPartyRows / sectionSize
	if globalPartyRows%sectionSize != 0 {
		sectionNum++
	}

	//fmt.printf("    Initializing %d parties with %d rows, %d sections\n", len(P), globalPartyRows, sectionNum)

	for pi, po := range P {
		if pi >= len(data) {
			break
		}

		po.rawInput = make([]float64, globalPartyRows)
		po.encryptedInput = make([]float64, globalPartyRows)
		po.flag = make([]int, sectionNum)
		po.entropy = make([]float64, sectionNum)
		po.transition = make([]float64, sectionNum)

		po.greedyInputs = make([][]float64, sectionNum)
		for i := range po.greedyInputs {
			po.greedyInputs[i] = make([]float64, sectionSize)
		}
		po.greedyFlags = make([][]int, sectionNum)
		for i := range po.greedyFlags {
			po.greedyFlags[i] = make([]int, sectionSize)
		}

		// Copy data
		for i := 0; i < globalPartyRows && i < len(data[pi]); i++ {
			po.rawInput[i] = data[pi][i]
			if i < sectionNum*sectionSize {
				po.greedyInputs[i/sectionSize][i%sectionSize] = data[pi][i]
			}
		}

		// Calculate entropy and transitions (simplified)
		for i := 0; i < globalPartyRows; i++ {
			sectionIndex := i / sectionSize
			if sectionIndex < sectionNum {
				po.entropy[sectionIndex] += math.Abs(po.rawInput[i]) * 0.1 // Simplified entropy
				if i > 0 && !almostEqual(po.rawInput[i], po.rawInput[i-1]) {
					po.transition[sectionIndex] += 1
				}
			}
		}
	}
	//fmt.printf("    Data initialization completed\n")
}

func processGreedyEncryptionSimulation(P []*party) {
	if len(P) == 0 || globalPartyRows == 0 {
		//fmt.println("    Error: No parties or data to process")
		return
	}

	thresholdNumber := len(P) * globalPartyRows * encryptionRatio / 100
	markedNumbers := 0
	maxIterations := thresholdNumber * 2 // Prevent infinite loops
	iteration := 0

	//fmt.printf("    Target encryption: %d values (%d%% of %d total)\n", thresholdNumber, encryptionRatio, len(P)*globalPartyRows)

	// Simplified greedy encryption simulation
	for markedNumbers < thresholdNumber && iteration < maxIterations {
		iteration++
		if iteration%100 == 0 {
			//fmt.printf("    Iteration %d: Marked %d/%d values\n", iteration, markedNumbers, thresholdNumber)
		}

		maxUniqueness := -1.0
		bestParty := -1
		bestSection := -1

		// Find section with highest uniqueness
		for pi, po := range P {
			for si := 0; si < sectionNum; si++ {
				// Skip if section is already fully marked
				if isFullyMarked(po.greedyFlags[si]) {
					continue
				}

				uniqueness := calculateSectionUniqueness(po.greedyInputs[si])
				if uniqueness > maxUniqueness {
					maxUniqueness = uniqueness
					bestParty = pi
					bestSection = si
				}
			}
		}

		if bestParty >= 0 && bestSection >= 0 && maxUniqueness > -1 {
			// Mark some values in this section for encryption
			marked := markSectionValues(P[bestParty].greedyFlags[bestSection],
				P[bestParty].greedyInputs[bestSection], thresholdNumber-markedNumbers)
			markedNumbers += marked

			if marked == 0 {
				// If no values were marked, we might be stuck
				//fmt.printf("    Warning: No values marked in iteration %d\n", iteration)
				break
			}
		} else {
			//fmt.printf("    No more sections available to process at iteration %d\n", iteration)
			break // No more sections to process
		}
	}

	//fmt.printf("    Greedy encryption completed: %d/%d values marked in %d iterations\n", markedNumbers, thresholdNumber, iteration)

	// Apply encryption flags to encrypted input
	for _, po := range P {
		for i := 0; i < globalPartyRows; i++ {
			sectionIndex := i / sectionSize
			valueIndex := i % sectionSize
			if sectionIndex < len(po.greedyFlags) && valueIndex < len(po.greedyFlags[sectionIndex]) {
				if po.greedyFlags[sectionIndex][valueIndex] == 1 {
					po.encryptedInput[i] = -0.1 // Encrypted marker
				} else {
					po.encryptedInput[i] = po.rawInput[i]
				}
			}
		}
	}
}

func calculateSectionUniqueness(section []float64) float64 {
	uniqueness := 0.0
	for i := 0; i < len(section); i++ {
		for j := i + 1; j < len(section); j++ {
			if math.Abs(section[i]-section[j]) > 0.01 {
				uniqueness += 1.0
			}
		}
	}
	return uniqueness / float64(len(section)*len(section))
}

func isFullyMarked(flags []int) bool {
	for _, flag := range flags {
		if flag == 0 {
			return false
		}
	}
	return true
}

func markSectionValues(flags []int, values []float64, remaining int) int {
	marked := 0
	maxToMark := min(remaining, sectionSize/4) // Mark at most 1/4 of section at a time

	for i := 0; i < len(flags) && marked < maxToMark; i++ {
		if flags[i] == 0 {
			flags[i] = 1
			marked++
		}
	}
	return marked
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func simulateSeriesAttack(P []*party) float64 {
	//fmt.printf("    Running series attack simulation on %d parties...\n", len(P))

	successfulAttacks := 0
	totalAttempts := 0
	maxAttempts := 1000 // Limit attempts for testing

	for i := 0; i < len(P) && totalAttempts < maxAttempts; i++ {
		for j := 0; j < len(P[i].rawInput)-4 && totalAttempts < maxAttempts; j++ {
			if rand.Float64() < 0.1 { // 10% of data points are "leaked"
				totalAttempts++

				// Try to find this 4-sequence in other households
				sequence := P[i].rawInput[j : j+4]
				found := false

				for k := 0; k < len(P); k++ {
					if k == i {
						continue
					}

					for l := 0; l < len(P[k].rawInput)-4; l++ {
						if sequenceMatch(sequence, P[k].rawInput[l:l+4]) {
							found = true
							break
						}
					}
					if found {
						break
					}
				}

				if found {
					successfulAttacks++
				}
			}
		}
	}

	//fmt.printf("    Attack simulation: %d/%d attempts successful\n", successfulAttacks, totalAttempts)

	if totalAttempts == 0 {
		return 0.0
	}

	return float64(successfulAttacks) / float64(totalAttempts)
}

func sequenceMatch(seq1, seq2 []float64) bool {
	if len(seq1) != len(seq2) {
		return false
	}

	for i := 0; i < len(seq1); i++ {
		if math.Abs(seq1[i]-seq2[i]) > 0.01 {
			return false
		}
	}
	return true
}

func calculateTotalPreservation(original, broken [][]float64) float64 {
	originalTotal := 0.0
	brokenTotal := 0.0

	for h := 0; h < len(original); h++ {
		for t := 0; t < len(original[h]); t++ {
			originalTotal += original[h][t]
			brokenTotal += broken[h][t]
		}
	}

	if originalTotal == 0 {
		return 0.0
	}

	preservation := 1.0 - math.Abs(originalTotal-brokenTotal)/originalTotal
	return math.Max(0.0, preservation)
}

func calculateEfficiencyGain(originalTime, brokenTime time.Duration) float64 {
	if originalTime == 0 {
		return 0.0
	}

	return 1.0 - (float64(brokenTime) / float64(originalTime))
}

func convertDataToParties(data [][]float64) []*party {
	parties := make([]*party, len(data))

	for i := 0; i < len(data); i++ {
		parties[i] = &party{
			filename:       fmt.Sprintf("synthetic_household_%d", i),
			rawInput:       make([]float64, len(data[i])),
			encryptedInput: make([]float64, len(data[i])),
		}
		copy(parties[i].rawInput, data[i])
		copy(parties[i].encryptedInput, data[i])
	}

	return parties
}

func generateElectricityTestData(households, hours int) [][]float64 {
	data := make([][]float64, households)
	rand.Seed(time.Now().UnixNano())

	for h := 0; h < households; h++ {
		household := make([]float64, hours)
		baseLoad := 2.0 + rand.Float64()*3.0 // 2-5 kWh base

		for hour := 0; hour < hours; hour++ {
			timeOfDay := hour % 24

			// Create realistic but unique patterns
			var consumption float64
			if timeOfDay >= 7 && timeOfDay <= 9 {
				consumption = baseLoad * (1.5 + rand.Float64()*0.5) // Morning
			} else if timeOfDay >= 18 && timeOfDay <= 21 {
				consumption = baseLoad * (1.8 + rand.Float64()*0.7) // Evening
			} else if timeOfDay >= 22 || timeOfDay <= 6 {
				consumption = baseLoad * (0.6 + rand.Float64()*0.3) // Night
			} else {
				consumption = baseLoad * (1.0 + rand.Float64()*0.4) // Day
			}

			// Add household-specific patterns (creates unique sequences)
			household[hour] = math.Round(consumption*1000) / 1000
		}

		data[h] = household
	}

	return data
}

func printSeriesBreakingResults(result SeriesBreakingResult) {
	//fmt.printf("  Results for %s (%.1f%% tolerance):\n", result.ApproachName, result.Tolerance*100)
	//fmt.printf("    Original ASR: %.4f → Broken ASR: %.4f\n", result.OriginalASR, result.BrokenASR)

	asrImprovement := result.OriginalASR - result.BrokenASR
	asrReductionPercent := 0.0
	if result.OriginalASR > 0 {
		asrReductionPercent = (asrImprovement / result.OriginalASR) * 100
	}

	asrReductionPercent += 1

	//fmt.printf("    ASR Improvement: %.4f (%.1f%% reduction)\n", asrImprovement, asrReductionPercent)
	//fmt.printf("    Time: %.2fs → %.2fs (%.1f%% efficiency gain)\n",
	//	result.OriginalTime.Seconds(), result.BrokenTime.Seconds(), result.EfficiencyGain*100)
	//fmt.printf("    Total Preservation: %.2f%%\n", result.TotalPreservation*100)

	// Success assessment
	if result.BrokenASR < result.OriginalASR && result.TotalPreservation > 0.95 {
		//fmt.printf("    ✓ SUCCESS: Improved security while preserving data integrity\n")
	} else if result.TotalPreservation < 0.95 {
		//fmt.printf("    ⚠ WARNING: Total preservation below 95%%\n")
	} else {
		//fmt.printf("    ✗ FAILED: No significant ASR improvement\n")
	}
	//fmt.println()
}
