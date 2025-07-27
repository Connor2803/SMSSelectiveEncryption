package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func main() {
	if len(os.Args) != 7 {
		fmt.Println("Usage: go run test_uniqueness_standalone.go <strategy> <dataset> <unique> <target> <ratio> <households>")
		os.Exit(1)
	}

	strategy, _ := strconv.Atoi(os.Args[1])
	dataset, _ := strconv.Atoi(os.Args[2])
	unique, _ := strconv.Atoi(os.Args[3])
	target, _ := strconv.Atoi(os.Args[4])
	ratio, _ := strconv.Atoi(os.Args[5])
	households, _ := strconv.Atoi(os.Args[6])

	fmt.Printf("Running uniqueness test: Strategy=%d, Dataset=%d, Target=%d, Ratio=%d, Households=%d\n",
		strategy, dataset, target, ratio, households)

	// Generate test data
	data := generateTestData(dataset, households)

	// Apply uniqueness-based encryption
	encryptedData, encryptionFlags := applyUniquenessEncryption(data, strategy, ratio, target)

	// Calculate ASR
	asr := calculateASR(data, encryptedData, encryptionFlags)

	// Calculate privacy metrics
	privacy := calculatePrivacyMetrics(data, encryptedData, encryptionFlags)

	// Output results
	outputFile := fmt.Sprintf("../results/uniqueness_test_%d_%d_%d_%d_%d_%d.csv",
		strategy, dataset, unique, target, ratio, households)

	writeResults(outputFile, strategy, dataset, target, ratio, households, asr, privacy)

	fmt.Printf("Test completed. ASR: %.4f, Privacy: %.4f\n", asr, privacy)
	fmt.Printf("Results saved to: %s\n", outputFile)
}

// generateTestData creates synthetic smart meter data
func generateTestData(dataset, households int) [][]float64 {
	rand.Seed(time.Now().UnixNano())
	data := make([][]float64, households)

	hoursInWeek := 168

	for h := 0; h < households; h++ {
		household := make([]float64, hoursInWeek)

		// Generate realistic consumption patterns
		baseLoad := 20.0 + rand.Float64()*30.0 // 20-50 kWh base

		for hour := 0; hour < hoursInWeek; hour++ {
			timeOfDay := hour % 24
			dayOfWeek := hour / 24

			// Daily pattern
			var dailyFactor float64
			if timeOfDay >= 6 && timeOfDay <= 9 {
				dailyFactor = 1.5 // Morning peak
			} else if timeOfDay >= 17 && timeOfDay <= 21 {
				dailyFactor = 1.8 // Evening peak
			} else if timeOfDay >= 22 || timeOfDay <= 5 {
				dailyFactor = 0.6 // Night low
			} else {
				dailyFactor = 1.0 // Day normal
			}

			// Weekend pattern
			var weekendFactor float64
			if dayOfWeek == 5 || dayOfWeek == 6 { // Weekend
				weekendFactor = 0.8 + rand.Float64()*0.4 // More random
			} else {
				weekendFactor = 1.0
			}

			// Add noise
			noise := (rand.Float64() - 0.5) * 0.2

			consumption := baseLoad * dailyFactor * weekendFactor * (1.0 + noise)

			// Ensure positive values
			if consumption < 0 {
				consumption = 0.1
			}

			// Dataset-specific adjustments
			if dataset == 1 { // Water
				consumption *= 0.1 // Water consumption is much lower
			}

			household[hour] = consumption
		}

		data[h] = household
	}

	return data
}

// applyUniquenessEncryption applies encryption based on uniqueness strategy
func applyUniquenessEncryption(data [][]float64, strategy, ratio, target int) ([][]float64, [][]int) {
	encryptedData := make([][]float64, len(data))
	encryptionFlags := make([][]int, len(data))

	for h := 0; h < len(data); h++ {
		household := data[h]
		encrypted := make([]float64, len(household))
		flags := make([]int, len(household))

		// Calculate uniqueness scores
		uniquenessScores := calculateUniquenessScores(household, strategy)

		// Determine encryption threshold
		threshold := calculateEncryptionThreshold(uniquenessScores, ratio)

		// Apply encryption
		for i := 0; i < len(household); i++ {
			if uniquenessScores[i] > threshold {
				// Encrypt (replace with encrypted value)
				encrypted[i] = encryptValue(household[i])
				flags[i] = 1
			} else {
				// Keep original
				encrypted[i] = household[i]
				flags[i] = 0
			}
		}

		encryptedData[h] = encrypted
		encryptionFlags[h] = flags
	}

	return encryptedData, encryptionFlags
}

// calculateUniquenessScores calculates uniqueness for each data point
func calculateUniquenessScores(data []float64, strategy int) []float64 {
	scores := make([]float64, len(data))
	windowSize := 24 // 24-hour window

	for i := 0; i < len(data); i++ {
		switch strategy {
		case 1: // Global uniqueness
			scores[i] = calculateGlobalUniqueness(data, i, windowSize)
		case 2: // Local uniqueness
			scores[i] = calculateLocalUniqueness(data, i, windowSize)
		case 3: // Adaptive uniqueness
			scores[i] = calculateAdaptiveUniqueness(data, i, windowSize)
		default:
			scores[i] = 0.5
		}
	}

	return scores
}

// calculateGlobalUniqueness calculates uniqueness against entire dataset
func calculateGlobalUniqueness(data []float64, index, windowSize int) float64 {
	if index >= len(data) {
		return 0.0
	}

	value := data[index]
	mean, std := calculateMeanStd(data)

	if std == 0 {
		return 0.0
	}

	// Z-score based uniqueness
	zScore := math.Abs((value - mean) / std)
	uniqueness := math.Min(1.0, zScore/3.0) // Normalize to [0,1]

	return uniqueness
}

// calculateLocalUniqueness calculates uniqueness within local window
func calculateLocalUniqueness(data []float64, index, windowSize int) float64 {
	start := maxInt(0, index-windowSize/2)
	end := minInt(len(data), index+windowSize/2+1)

	if end-start < 2 {
		return 0.0
	}

	window := data[start:end]
	mean, std := calculateMeanStd(window)

	if std == 0 {
		return 0.0
	}

	value := data[index]
	zScore := math.Abs((value - mean) / std)
	uniqueness := math.Min(1.0, zScore/2.0)

	return uniqueness
}

// calculateAdaptiveUniqueness uses adaptive threshold based on data characteristics
func calculateAdaptiveUniqueness(data []float64, index, windowSize int) float64 {
	localUniqueness := calculateLocalUniqueness(data, index, windowSize)
	globalUniqueness := calculateGlobalUniqueness(data, index, windowSize)

	// Adaptive weighting based on local variance
	start := maxInt(0, index-windowSize/2)
	end := minInt(len(data), index+windowSize/2+1)
	window := data[start:end]
	_, std := calculateMeanStd(window)

	// Higher variance -> more weight on local uniqueness
	localWeight := math.Min(1.0, std/10.0)
	globalWeight := 1.0 - localWeight

	return localWeight*localUniqueness + globalWeight*globalUniqueness
}

// calculateEncryptionThreshold determines threshold for given encryption ratio
func calculateEncryptionThreshold(scores []float64, ratio int) float64 {
	if len(scores) == 0 {
		return 0.5
	}

	// Sort scores to find percentile threshold
	sortedScores := make([]float64, len(scores))
	copy(sortedScores, scores)

	// Simple bubble sort
	for i := 0; i < len(sortedScores); i++ {
		for j := i + 1; j < len(sortedScores); j++ {
			if sortedScores[i] < sortedScores[j] {
				sortedScores[i], sortedScores[j] = sortedScores[j], sortedScores[i]
			}
		}
	}

	// Find threshold for desired encryption ratio
	thresholdIndex := (ratio * len(sortedScores)) / 100
	if thresholdIndex >= len(sortedScores) {
		thresholdIndex = len(sortedScores) - 1
	}

	return sortedScores[thresholdIndex]
}

// encryptValue simulates encryption (placeholder implementation)
func encryptValue(value float64) float64 {
	// Simple placeholder: add noise to simulate encryption
	noise := (rand.Float64() - 0.5) * 0.1 * value
	return value + noise
}

// calculateASR calculates Attack Success Rate
func calculateASR(original, encrypted [][]float64, flags [][]int) float64 {
	totalAttempts := 0
	successfulAttacks := 0

	for h := 0; h < len(original); h++ {
		for i := 0; i < len(original[h]); i++ {
			if flags[h][i] == 1 { // Encrypted value
				totalAttempts++

				// Simple attack: try to infer from neighboring values
				inferredValue := inferFromNeighbors(encrypted[h], i, flags[h])
				actualValue := original[h][i]

				// Attack succeeds if inference is within 10% of actual value
				if math.Abs(inferredValue-actualValue)/actualValue < 0.1 {
					successfulAttacks++
				}
			}
		}
	}

	if totalAttempts == 0 {
		return 0.0
	}

	return float64(successfulAttacks) / float64(totalAttempts)
}

// inferFromNeighbors attempts to infer encrypted value from neighbors
func inferFromNeighbors(data []float64, index int, flags []int) float64 {
	neighbors := make([]float64, 0)

	// Collect unencrypted neighbors
	for i := maxInt(0, index-5); i < minInt(len(data), index+6); i++ {
		if i != index && flags[i] == 0 {
			neighbors = append(neighbors, data[i])
		}
	}

	if len(neighbors) == 0 {
		return 0.0
	}

	// Return average of neighbors
	sum := 0.0
	for _, v := range neighbors {
		sum += v
	}
	return sum / float64(len(neighbors))
}

// calculatePrivacyMetrics calculates privacy preservation metrics
func calculatePrivacyMetrics(original, encrypted [][]float64, flags [][]int) float64 {
	totalEntropy := 0.0
	numHouseholds := 0

	for h := 0; h < len(original); h++ {
		entropy := calculateEntropy(encrypted[h])
		totalEntropy += entropy
		numHouseholds++
	}

	if numHouseholds == 0 {
		return 0.0
	}

	avgEntropy := totalEntropy / float64(numHouseholds)
	return math.Min(1.0, avgEntropy/4.0) // Normalize to [0,1]
}

// calculateEntropy calculates Shannon entropy
func calculateEntropy(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}

	// Discretize data into bins
	bins := discretizeData(data, 10)
	freq := make(map[int]int)

	for _, bin := range bins {
		freq[bin]++
	}

	entropy := 0.0
	n := float64(len(data))

	for _, count := range freq {
		if count > 0 {
			p := float64(count) / n
			entropy -= p * math.Log2(p)
		}
	}

	return entropy
}

// discretizeData converts continuous data to discrete bins
func discretizeData(data []float64, numBins int) []int {
	if len(data) == 0 {
		return []int{}
	}

	min, max := data[0], data[0]
	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	if max == min {
		return make([]int, len(data))
	}

	binWidth := (max - min) / float64(numBins)
	bins := make([]int, len(data))

	for i, v := range data {
		bin := int((v - min) / binWidth)
		if bin >= numBins {
			bin = numBins - 1
		}
		bins[i] = bin
	}

	return bins
}

// calculateMeanStd calculates mean and standard deviation
func calculateMeanStd(data []float64) (float64, float64) {
	if len(data) == 0 {
		return 0.0, 0.0
	}

	sum := 0.0
	for _, v := range data {
		sum += v
	}
	mean := sum / float64(len(data))

	variance := 0.0
	for _, v := range data {
		variance += (v - mean) * (v - mean)
	}
	variance /= float64(len(data))

	return mean, math.Sqrt(variance)
}

// writeResults writes test results to CSV file
func writeResults(filename string, strategy, dataset, target, ratio, households int, asr, privacy float64) {
	file, err := os.Create(filename)
	if err != nil {
		fmt.Printf("Error creating output file: %v\n", err)
		return
	}
	defer file.Close()

	// Write CSV header
	fmt.Fprintln(file, "Strategy,Dataset,Target,Ratio,Households,ASR,Privacy,Timestamp")

	// Write results
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	fmt.Fprintf(file, "%d,%d,%d,%d,%d,%.6f,%.6f,%s\n",
		strategy, dataset, target, ratio, households, asr, privacy, timestamp)
}

// Utility functions
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
