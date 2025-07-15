/*
running command: // dataset
go run .\RL_model_V1_WATER\generate_metrics_V1.go 2
> Run this code for the electricity dataset.
*/

package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/drlwe"
	"github.com/tuneinsight/lattigo/v4/rlwe"
	utils "lattigo"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"time"
)

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func getRandom(numberRange int) (randNumber int) {
	randNumber = rand.Intn(numberRange) //[0, numberRange-1]
	return
}

type Party struct {
	filename    string
	sk          *rlwe.SecretKey
	rlkEphemSk  *rlwe.SecretKey
	ckgShare    *drlwe.CKGShare
	rkgShareOne *drlwe.RKGShare
	rkgShareTwo *drlwe.RKGShare
	rtgShare    *drlwe.RTGShare
	pcksShare   *drlwe.PCKSShare

	rawInput       []float64   // All data
	rawDates       []string    // All dates corresponding to rawInput
	input          [][]float64 // Encryption data
	plainInput     []float64   // Plaintext data
	flag           []int       // Slice to track which sections have been encrypted
	group          []int
	entropy        []float64   // Entropy for block
	encryptedInput [][]float64 // Transformed data for each block/section after encryption
} // Struct to represent an individual household

type SectionsMetrics struct {
	HouseholdCSVFileName      string
	SectionNumber             int
	DateRange                 string // DD-MM-YYYY to DD-MM-YYYY
	SectionSumUsage           float64
	EncryptionRatio           float64
	RawEntropy                float64 // Per-section, pre-encryption entropy value
	RemainingEntropy          float64 // Per-section, post-encryption entropy value
	ASRMean                   float64
	ASRStandardError          float64
	ASRDuration               time.Duration // Total time for the entire attack to run (in seconds).
	AvgPerAttackRunDuration   float64       // Average time for a single attack to run (in seconds).
	ProgramAllocatedMemoryMiB float64       // Program's allocated memory in MiB for this ratio
} // Struct to represent a section/block in an individual household

type ResultKey struct {
	HouseholdCSVFileName string
	SectionNumber        int
	EncryptionRatio      float64
} // Struct to represent the unique identifier for a section/block.

const DATASET_WATER = 1
const DATASET_ELECTRICITY = 2

const MAX_PARTY_ROWS = 10240 // Total records/meter readings per household (WATER dataset lowest num: 20485)
const sectionSize = 1024     // Block Size: 2048 for summation correctness, 8192 for variance correctness

var currentDataset int // Water(1), Electricity(2)
var maxHouseholdsNumber = 80

var sectionNum int
var globalPartyRows = -1

var max_attackLoop = 1000 // Default value for number of loops for membershipIdentificationAttack.
var atdSize = 12          // Element number of unique attacker data
var uniqueATD int = 0     // Unique attacker data, 1 for true, 0 for false
var usedRandomStartPartyPairs = map[int][]int{}
var min_percent_matched = 90 // Default value for minimum percentage match for identification.

func main() {

	rand.Seed(time.Now().UnixNano())

	var args []int

	for _, arg := range os.Args[1:] {
		num, err := strconv.Atoi(arg)
		if err != nil {
			fmt.Println("Error:", err)
			return
		}
		args = append(args, num)
	}

	if len(args) > 0 {
		currentDataset = args[0]
	}
	var metricsOutputFileName string
	var partyMetricsOutputFileName string

	if currentDataset == DATASET_WATER {
		metricsOutputFileName = "./RL_model_V1_WATER/ML_metrics_WATER.csv"
		partyMetricsOutputFileName = "./RL_model_V1_WATER/ML_party_metrics_WATER.csv"
	} else {
		metricsOutputFileName = "./RL_model_V1_ELECTRICITY/ML_metrics_ELECTRICITY.csv"
		partyMetricsOutputFileName = "./RL_model_V1_ELECTRICITY/ML_party_metrics_ELECTRICITY.csv"
	}
	outputFile1, err := os.Create(metricsOutputFileName)
	check(err)
	outputFile2, err := os.Create(partyMetricsOutputFileName)
	check(err)
	defer outputFile1.Close()
	defer outputFile2.Close()

	metricsWriter := csv.NewWriter(outputFile1)
	partyWriter := csv.NewWriter(outputFile2)
	defer metricsWriter.Flush()
	defer partyWriter.Flush()

	fileList := []string{}
	paramsDef := utils.PN10QP27CI
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)

	// Get the current working directory
	wd, err := os.Getwd()
	check(err)

	var pathFormat string
	var path string
	if strings.Contains(wd, "examples") {
		pathFormat = filepath.Join("..", "..", "..", "examples", "datasets", "%s", "households_%d")
	} else {
		pathFormat = filepath.Join("examples", "datasets", "%s", "households_%d")
	}
	if currentDataset == DATASET_WATER {
		path = fmt.Sprintf(pathFormat, "water", MAX_PARTY_ROWS)
	} else {
		path = fmt.Sprintf(pathFormat, "electricity", MAX_PARTY_ROWS)
	}

	folder := filepath.Join(wd, path)

	err = filepath.Walk(folder, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			fmt.Println(err)
			return err
		}
		if !info.IsDir() {
			fileList = append(fileList, path)
		}
		return nil
	})
	check(err)
	process(fileList[:maxHouseholdsNumber], params, metricsWriter, partyWriter)
}

// process reads the input files, calculates the entropy for different
// encryption ratios, and writes the results to a CSV file.
// process reads the input files, calculates entropy, and writes the results.
func process(fileList []string, params ckks.Parameters, metricsWriter *csv.Writer, partyWriter *csv.Writer) {

	// --- 1. Setup Phase ---
	err := metricsWriter.Write([]string{
		"filename",
		"section",
		"date_range",
		"section_sum_usage",
		"encryption_ratio",
		"section_raw_entropy",
		"section_remaining_entropy",
		"asr_mean",
		"asr_standard_error",
		"asr_attack_duration",
		"avg_time_per_attack_run",
		"allocated_memory_MiB",
	})
	check(err)

	err = partyWriter.Write([]string{
		"filename",
		"encryption_ratio",
		"summation_error",
		"deviation_error",
		"encryption_time_ns",
		"decryption_time_ns",
		"summation_operations_time_ns",
		"deviation_operations_time_ns",
	})
	check(err)

	P := genParties(params, fileList)

	// Populate Party.rawInput and sets the global sectionNum.
	_, expAverage, _, _, _, _ := genInputs(P, &[]SectionsMetrics{})

	allResults := make(map[ResultKey]SectionsMetrics)

	// --- 2. Processing Phase ---
	encryptionRatios := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}

	for _, encRatio := range encryptionRatios {
		for _, party := range P {
			for sectionIdx := 0; sectionIdx < sectionNum; sectionIdx++ {

				// Get the original raw data for the current section.
				start := sectionIdx * sectionSize
				end := (sectionIdx + 1) * sectionSize
				if end > len(party.rawInput) {
					end = len(party.rawInput)
				}
				rawSectionData := party.rawInput[start:end]
				rawSectionDates := party.rawDates[start:end]

				// Calculate raw entropy directly from the raw data.
				rawEntropy := calculateEntropy(rawSectionData)

				// Calculate sum of all the utility meter readings from a section.
				sectionSumUsage := 0.0
				for _, val := range rawSectionData {
					sectionSumUsage += val
				}

				// Determine Date Range for section
				var dateRange string
				if len(rawSectionDates) > 0 {
					firstDate := rawSectionDates[0]
					lastDate := rawSectionDates[len(rawSectionDates)-1]
					dateRange = fmt.Sprintf("%s to %s", firstDate, lastDate)
				} else {
					dateRange = "N/A"
				}

				// Create a temporary copy to apply the encryption ratio to.
				tempSectionInput := make([]float64, len(rawSectionData))
				copy(tempSectionInput, rawSectionData)

				for j := range tempSectionInput {
					tempSectionInput[j] *= (1.0 - encRatio)

					// Apply quantisation (coarser rounding) based on the encryption ratio
					// To simulate the information loss that occurs during a privacy-preserving.
					var precision float64
					if encRatio <= 0.3 {
						precision = 100 // Keep 2 decimal places for low ratios
					} else if encRatio <= 0.7 {
						precision = 10 // Round to 1 decimal places for medium ratios
					} else {
						precision = 1 // Round to 0 decimal place for high ratios
					}
					tempSectionInput[j] = float64(math.Round(tempSectionInput[j]*float64(precision)) / float64(precision))
				}

				// Store the transformed data in Party.encryptedInput for attack
				party.encryptedInput[sectionIdx] = tempSectionInput

				// Calculate remaining entropy from the modified data.
				remainingEntropy := calculateEntropy(tempSectionInput)

				// Create the key for the map
				key := ResultKey{
					HouseholdCSVFileName: filepath.Base(party.filename),
					SectionNumber:        sectionIdx,
					EncryptionRatio:      encRatio,
				}

				// Append the complete result for this data point.
				allResults[key] = SectionsMetrics{
					HouseholdCSVFileName: party.filename,
					SectionNumber:        sectionIdx,
					DateRange:            dateRange,
					SectionSumUsage:      sectionSumUsage,
					EncryptionRatio:      encRatio,
					RawEntropy:           rawEntropy,
					RemainingEntropy:     remainingEntropy,
					// REMINDER: ASR metrics will be updated after the memberIdentificationAttack call for this ratio
				}
			}
		}

		// --- 2.5 Homomorphic Computations Phase --
		// Populate party.input and party.plainInput with the raw data for HE operations.
		// party.input slice  holds sections intended for HE operations.
		// party.plainInput slice holds sections intended for non-HE processing.
		// NOTE: The population of these slices are deterministic, i.e., the first numHESections are sent for HE operations
		// which is based on the proportion of the sections encrypted using the encRatio,
		// i.e., encRatio of 0.4 would encrypt the first 40% of the sections within a party/household.
		expSummationForRatio := make([]float64, len(P))
		expDeviationForRatio := make([]float64, len(P))
		plainSum := make([]float64, len(P))

		for pi, po := range P {
			po.input = make([][]float64, 0)
			po.plainInput = make([]float64, 0)

			numHESections := int(float64(sectionNum) * encRatio)
			totalSections := len(po.rawInput) / sectionSize

			heInputDataFlat := []float64{}

			for s := 0; s < totalSections; s++ {
				sectionStart := s * sectionSize
				sectionEnd := (s + 1) * sectionSize
				if sectionEnd > len(po.rawInput) {
					sectionEnd = len(po.rawInput)
				}
				currentSectionData := po.rawInput[sectionStart:sectionEnd]

				if s < numHESections {
					// This section goes to HE
					po.input = append(po.input, currentSectionData)
					heInputDataFlat = append(heInputDataFlat, currentSectionData...)
				} else {
					// This section goes to plaintext
					po.plainInput = append(po.plainInput, currentSectionData...)

				}
			}
			var heSum float64
			for _, val := range heInputDataFlat {
				heSum += val
			}
			expSummationForRatio[pi] = heSum
			_, _, expDeviationForRatio[pi] = calculateStandardDeviationAndMeanAndVariance(heInputDataFlat)

			var currentPlainSum float64
			for _, val := range po.plainInput {
				currentPlainSum += val
			}
			plainSum[pi] = currentPlainSum
		}

		fmt.Printf("\n--- Starting HE Computations for Encryption Ratio: %.2f ---\n", encRatio)
		summationErrors, deviationErrors, encryptionTimes, decryptionTimes, summationOperationTimes, deviationOperationTimes := doHomomorphicOperations(params, P, expSummationForRatio, expAverage, expDeviationForRatio, plainSum)

		// Write Party-specific metrics to the partyWriter CSV for the current encryption ratio.
		for pi, party := range P { // Loop through each party to write their individual metrics
			err := partyWriter.Write([]string{
				filepath.Base(party.filename),
				fmt.Sprintf("%.2f", encRatio),
				fmt.Sprintf("%.6f", summationErrors[pi]),
				fmt.Sprintf("%.6f", deviationErrors[pi]),
				fmt.Sprintf("%d", encryptionTimes[pi]),
				fmt.Sprintf("%d", decryptionTimes[pi]),
				fmt.Sprintf("%d", summationOperationTimes[pi]),
				fmt.Sprintf("%d", deviationOperationTimes[pi]),
			})
			check(err)
		}

		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		currentAllocatedMemMiB := float64(bToMb(m.Alloc))

		// NOTE: Added to allResults in later statement!

		// --- 3. Attack Phase --
		// Reset house_sample for each encryption ratio iteration to collect new ASRs
		currentRatioAttackResults := []float64{} // Contains the results of each attack loop for a given encryption ratio.

		attackPhaseStartTime := time.Now()

		// After all households/parties and sections have had the current encryption ratio applied,
		// Run the member identification attack.
		fmt.Printf("\n--- Starting Attack Phase for Encryption Ratio: %.2f ---\n", encRatio)
		currentRatioAttackResults = memberIdentificationAttack(P, encRatio)

		// Calculate ASR mean and standard error for the current encryption ratio
		asrMean := 0.0
		asrStdError := 0.0

		if len(currentRatioAttackResults) > 0 {
			// Convert success counts to ASRs (Attack Success Rates)
			var asrRates []float64
			totalAttackableInstancesPerRun := float64(len(P))
			if totalAttackableInstancesPerRun == 0 {
				fmt.Println("ERROR: totalAttackableInstancesPerRun is 0. Cannot calculate ASR.")
			} else {
				for _, successCount := range currentRatioAttackResults {
					asrRates = append(asrRates, successCount/float64(totalAttackableInstancesPerRun))
				}
			}

			stdDev, mean, _ := calculateStandardDeviationAndMeanAndVariance(asrRates)
			asrMean = mean
			asrStdError = stdDev / math.Sqrt(float64(len(asrRates)))
			fmt.Printf("Final ASR Mean: %.6f, ASR Standard Error: %.6f\n", asrMean, asrStdError)
		} else {
			fmt.Println("WARNING: currentRatioAttackResults is empty. ASR cannot be calculated.")
		}

		attackPhaseEndTime := time.Now()
		asrAttackDuration := attackPhaseEndTime.Sub(attackPhaseStartTime)
		avgTimePerAttackRun := asrAttackDuration.Seconds() / float64(max_attackLoop)

		for key, result := range allResults {
			if key.EncryptionRatio == encRatio {
				// Note: Go maps store copies of values, so you must reassign the modified struct
				tempResult := result
				tempResult.ASRMean = asrMean
				tempResult.ASRStandardError = asrStdError
				tempResult.ASRDuration = asrAttackDuration
				tempResult.AvgPerAttackRunDuration = avgTimePerAttackRun
				tempResult.ProgramAllocatedMemoryMiB = currentAllocatedMemMiB

				allResults[key] = tempResult
			}
		}
	}

	// --- 4. Writing Phase ---
	for _, result := range allResults {
		err := metricsWriter.Write([]string{
			filepath.Base(result.HouseholdCSVFileName),
			strconv.Itoa(result.SectionNumber),
			result.DateRange,
			fmt.Sprintf("%.2f", result.SectionSumUsage),
			fmt.Sprintf("%.2f", result.EncryptionRatio),
			fmt.Sprintf("%.6f", result.RawEntropy),
			fmt.Sprintf("%.6f", result.RemainingEntropy),
			fmt.Sprintf("%.6f", result.ASRMean),
			fmt.Sprintf("%.6f", result.ASRStandardError),
			fmt.Sprintf("%.6f", result.ASRDuration.Seconds()),
			fmt.Sprintf("%.6f", result.AvgPerAttackRunDuration),
			fmt.Sprintf("%.2f", result.ProgramAllocatedMemoryMiB),
		})
		check(err)
	}
	metricsWriter.Flush()

}

// File Reading
func readCSV(path string) []string {
	data, err := os.ReadFile(path)
	check(err)
	dArray := strings.Split(string(data), "\n")
	return dArray[:len(dArray)-1]
}

// Parse CSV
func parseCSV(filename string) ([]float64, []string) {
	csvLines := readCSV(filename)
	var readings []float64
	var dates []string

	for _, line := range csvLines {
		slices := strings.Split(line, ",")
		if len(slices) < 2 {
			continue
		}
		utilityDate := strings.TrimSpace(slices[0])
		usageStr := strings.TrimSpace(slices[len(slices)-1])

		usage, err := strconv.ParseFloat(usageStr, 64)
		check(err)
		if err == nil {
			readings = append(readings, usage)
			dates = append(dates, utilityDate)
		}
	}

	return readings, dates
}

// Generate parties/households
func genParties(params ckks.Parameters, fileList []string) []*Party {

	// Create each Party, and allocate the memory for all the shares that the protocols will need
	P := make([]*Party, len(fileList))

	for i, _ := range P {
		po := &Party{}
		po.sk = ckks.NewKeyGenerator(params).GenSecretKey()
		po.filename = fileList[i]
		P[i] = po
	}

	return P
}

// Generate inputs of parties/households
func genInputs(P []*Party, metrics *[]SectionsMetrics) (expSummation, expAverage, expDeviation []float64, min, max, entropySum float64) {
	globalPartyRows = -1
	sectionNum = 0
	min = math.MaxFloat64
	max = float64(-1)
	frequencyMap := map[float64]int{}
	entropyMap := map[float64]float64{}

	entropySum = 0.0

	for pi, po := range P {
		// Setup (rawInput, flags etc.)
		partyRows, partyDates := parseCSV(po.filename)
		if currentDataset == DATASET_WATER { // Reverse chronological order
			for i, j := 0, len(partyRows)-1; i < j; i, j = i+1, j-1 {
				partyRows[i], partyRows[j] = partyRows[j], partyRows[i]
				partyDates[i], partyDates[j] = partyDates[j], partyDates[i]
			}
		}
		lenPartyRows := len(partyRows)
		if lenPartyRows > MAX_PARTY_ROWS {
			lenPartyRows = MAX_PARTY_ROWS
		}
		if globalPartyRows == -1 {
			sectionNum = lenPartyRows / sectionSize // sectionNum = 10, since lenPartyRows = 10240, sectionSize = 1024
			if lenPartyRows%sectionSize != 0 {
				sectionNum++
			}
			globalPartyRows = lenPartyRows
			expSummation = make([]float64, len(P))
			expAverage = make([]float64, len(P))
			expDeviation = make([]float64, len(P))
		} else if globalPartyRows != lenPartyRows {
			//make sure pi.input[] has the same size
			err := errors.New("Not all files have the same rows")
			check(err)
		}

		// Set up Party structure
		po.rawInput = make([]float64, globalPartyRows)
		po.rawDates = make([]string, globalPartyRows)
		po.flag = make([]int, sectionNum)        // Marked sections for encryption.
		po.entropy = make([]float64, sectionNum) // Entropy values for each section of a particular party/household.
		po.encryptedInput = make([][]float64, sectionNum)
		po.group = make([]int, sectionSize)

		// Fill in rawInput & frequencyMap
		for i := 0; i < globalPartyRows; i++ {
			usage := math.Round(partyRows[i]*1000) / 1000
			po.rawInput[i] = usage
			po.rawDates[i] = partyDates[i]
			frequencyMap[usage]++
			expSummation[pi] += po.rawInput[i]
		}

		expAverage[pi] = expSummation[pi] / float64(globalPartyRows)
		for i := range po.rawInput {
			temp := po.rawInput[i] - expAverage[pi]
			expDeviation[pi] += temp * temp
		}
		expDeviation[pi] /= float64(globalPartyRows)

		// Generate entropyMap
		totalRecords := maxHouseholdsNumber * MAX_PARTY_ROWS
		for k, v := range frequencyMap {
			possibility := float64(v) / float64(totalRecords)
			entropyMap[k] = -possibility * math.Log2(possibility)
		}

		for i := range po.rawInput {
			usage := po.rawInput[i]
			singleRecordEntropy := entropyMap[usage] / float64(frequencyMap[usage])
			po.entropy[i/sectionSize] += singleRecordEntropy
			entropySum += singleRecordEntropy // Global all data total entropy
		}
		po.flag = make([]int, sectionNum) // Reset flag for every Party

		// Min, Max based on currentTarget which is always (1) Entropy-based.
		for _, po := range P {
			var targetArr []float64
			targetArr = po.entropy

			for sIndex := range targetArr {
				if targetArr[sIndex] > max {
					max = targetArr[sIndex] // Max entropy seen so far across all parties.
				}
				if targetArr[sIndex] < min {
					min = targetArr[sIndex] // Min entropy seen so far across all parties.
				}
			}
		}

	}
	return
}

// calculateEntropy is a helper function that computes the Shannon entropy for a slice of float64 data and returns this entropy as a 64-bit float.
// Shannon Entropy quantifies the average amount of "information" or "uncertainty" contained in a set of data, this means
// A higher entropy value indicates that the data contains more distinct information.
func calculateEntropy(data []float64) float64 {
	var entropy float64
	frequency := make(map[float64]int)
	for _, val := range data {
		// Rounding here treats inputs that are very close to each other as identical for the purpose of entropy calculations
		// By converting continuous-like data into discrete "bins" before calculating probabilities.
		roundedVal := math.Round(val*1000) / 1000
		frequency[roundedVal]++
	}

	total := float64(len(data))
	if total > 0 {
		for _, count := range frequency {
			if count > 0 {
				p := float64(count) / total
				entropy -= p * math.Log2(p)
			}
		}
	}
	return entropy
}

func doHomomorphicOperations(params ckks.Parameters, P []*Party, expSummation, expAverage, expDeviation, plainSum []float64) (summationErrors []float64, deviationErrors []float64, encryptionTimes []int64, decryptionTimes []int64, summationOperationTimes []int64, deviationOperationTimes []int64) {

	// Initialize new slices for per-party times.
	summationErrors = make([]float64, len(P))
	deviationErrors = make([]float64, len(P))
	encryptionTimes = make([]int64, len(P))
	decryptionTimes = make([]int64, len(P))
	summationOperationTimes = make([]int64, len(P))
	deviationOperationTimes = make([]int64, len(P))

	// Key Generation Variables.
	tkgen := ckks.NewKeyGenerator(params)
	var tsk *rlwe.SecretKey
	var tpk *rlwe.PublicKey

	tsk = tkgen.GenSecretKey()
	tpk = tkgen.GenPublicKey(tsk)

	var rlk *rlwe.RelinearizationKey
	rlk = tkgen.GenRelinearizationKey(tsk, 1)

	rotations := params.RotationsForInnerSum(1, globalPartyRows)
	var rotk *rlwe.RotationKeySet
	rotk = tkgen.GenRotationKeysForRotations(rotations, false, tsk)

	decryptor := ckks.NewDecryptor(params, tsk)
	encoder := ckks.NewEncoder(params)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotk})

	// Generate ciphertexts by calling encPhase helper encryption function.
	encOutputs, encInputsSummation, encInputsNegative, encryptionTimes := encPhase(params, P, tpk, encoder)

	// Decryption Timing
	for pi, partyCiphers := range encOutputs {
		if len(partyCiphers) > 0 {
			start := time.Now()
			for _, ct := range partyCiphers {
				_ = decryptor.DecryptNew(ct)
			}
			decryptionTimes[pi] = time.Since(start).Nanoseconds()
		}
	}

	// Summation calculations ====================================================
	encSummationOuts := make([]*rlwe.Ciphertext, len(P))
	for i, partySections := range encInputsSummation {
		if len(partySections) == 0 {
			continue // Skip if no sections for this party.
		}
		partySummationOpsStart := time.Now()

		tmpCiphertext := partySections[0]
		for j := 1; j < len(partySections); j++ {
			evaluator.Add(tmpCiphertext, partySections[j], tmpCiphertext)
		}

		evaluator.InnerSum(tmpCiphertext, 1, params.Slots(), tmpCiphertext)

		summationOperationTimes[i] = time.Since(partySummationOpsStart).Nanoseconds()
		encSummationOuts[i] = tmpCiphertext
	}

	// Deviation calculation ====================================================
	encDeviationOuts := make([]*rlwe.Ciphertext, len(P))
	for i, partySections := range encInputsNegative {
		if len(partySections) == 0 || encSummationOuts[i] == nil {
			continue
		}
		partyDeviationOpsStart := time.Now()

		avgCiphertext := encSummationOuts[i].CopyNew()
		avgCiphertext.Scale = avgCiphertext.Scale.Mul(rlwe.NewScale(globalPartyRows))

		var aggregatedDeviation *rlwe.Ciphertext

		for j, sectionCipher := range partySections {
			// Step 1: Subtract the average from the value (by adding the negative value)
			currentDeviation := evaluator.AddNew(sectionCipher, avgCiphertext)

			// Step 2: Square the result
			evaluator.MulRelin(currentDeviation, currentDeviation, currentDeviation)

			// Step 3: Aggregate the squared differences
			if j == 0 {
				aggregatedDeviation = currentDeviation
			} else {
				evaluator.Add(aggregatedDeviation, currentDeviation, aggregatedDeviation)
			}
		}

		// Step 4: Perform the final InnerSum (rotation) on the aggregated result
		evaluator.InnerSum(aggregatedDeviation, 1, params.Slots(), aggregatedDeviation)

		aggregatedDeviation.Scale = aggregatedDeviation.Scale.Mul(rlwe.NewScale(globalPartyRows))
		encDeviationOuts[i] = aggregatedDeviation
		deviationOperationTimes[i] = time.Since(partyDeviationOpsStart).Nanoseconds()
	}

	// Collect Errors ===============================================
	ptresSummation := ckks.NewPlaintext(params, params.MaxLevel())
	for i, ct := range encSummationOuts {
		if ct != nil {
			decryptor.Decrypt(ct, ptresSummation)
			resSummation := encoder.Decode(ptresSummation, params.LogSlots())
			totalCKKSSum := real(resSummation[0]) + plainSum[i]
			summationErrors[i] = totalCKKSSum - expSummation[i]
		}
	}

	ptresDeviation := ckks.NewPlaintext(params, params.MaxLevel())
	for i, ct := range encDeviationOuts {
		if ct != nil {
			decryptor.Decrypt(ct, ptresDeviation)
			resDeviation := encoder.Decode(ptresDeviation, params.LogSlots())
			deviationErrors[i] = real(resDeviation[0]) - expDeviation[i]
		}
	}

	//// Decrypt ciphertexts==================
	//for pi, _ := range P {
	//	start := time.Now()
	//	for _, ct := range encOutputs[pi] {
	//		// Decrypt the ciphertext
	//		_ = decryptor.DecryptNew(ct)
	//
	//	}
	//	decryptionTimes[pi] = time.Since(start).Nanoseconds()
	//}

	return summationErrors, deviationErrors,
		encryptionTimes, decryptionTimes,
		summationOperationTimes, deviationOperationTimes
}

// encPhase is a helper function that encrypts the plaintext inputs for each party and returns their encrypted values.
// It also collects the time taken for each party's encryption.
func encPhase(params ckks.Parameters, P []*Party, pk *rlwe.PublicKey, encoder ckks.Encoder) (encOutputs, encInputsSummation, encInputsNegative [][]*rlwe.Ciphertext, encryptionTimes []int64) {

	encOutputs = make([][]*rlwe.Ciphertext, len(P))
	encInputsSummation = make([][]*rlwe.Ciphertext, len(P))
	encInputsNegative = make([][]*rlwe.Ciphertext, len(P))
	encryptionTimes = make([]int64, len(P))

	// Each party encrypts its input vector.
	encryptor := ckks.NewEncryptor(params, pk)
	pt := ckks.NewPlaintext(params, params.MaxLevel())

	for pi, po := range P {
		start := time.Now() // Start timing for the current party (pi)
		for j, _ := range po.input {
			if j == 0 { // This ensures the inner slices are initialized once per party
				encOutputs[pi] = make([]*rlwe.Ciphertext, 0)
				encInputsSummation[pi] = make([]*rlwe.Ciphertext, 0)
				encInputsNegative[pi] = make([]*rlwe.Ciphertext, 0)
			}
			// Encrypt original input.
			encoder.Encode(po.input[j], pt, params.LogSlots())
			tmpCiphertext := ckks.NewCiphertext(params, 1, params.MaxLevel())
			encryptor.Encrypt(pt, tmpCiphertext)
			encOutputs[pi] = append(encOutputs[pi], tmpCiphertext)
			encInputsSummation[pi] = append(encInputsSummation[pi], tmpCiphertext)

			// Turn po.input to negative for HE operations (subtraction and deviation).
			for k, _ := range po.input[j] {
				po.input[j][k] *= -1
			}
			// Encrypt negative input.
			encoder.Encode(po.input[j], pt, params.LogSlots())
			tmpCiphertext = ckks.NewCiphertext(params, 1, params.MaxLevel())
			encryptor.Encrypt(pt, tmpCiphertext)
			encInputsNegative[pi] = append(encInputsNegative[pi], tmpCiphertext)

			// Turn po.input to positive.
			for k, _ := range po.input[j] {
				po.input[j][k] *= -1
			}
		}
		encryptionTimes[pi] = time.Since(start).Nanoseconds()
	}
	return encOutputs, encInputsSummation, encInputsNegative, encryptionTimes
}

func memberIdentificationAttack(P []*Party, currentEncRatio float64) []float64 {
	var attackCount int
	var successCounts = []float64{} // Collects the number of successful attacks for each loop.
	var std float64
	var standard_error float64
	for attackCount = 0; attackCount < max_attackLoop; attackCount++ {
		var successNum = attackParties(P, currentEncRatio)         // Integer result of one attack run.
		successCounts = append(successCounts, float64(successNum)) // Stores the success count for this run.

		// NOTE: These calculations are for internal stopping conditions, not for statistical purposes.
		std, _, _ = calculateStandardDeviationAndMeanAndVariance(successCounts)
		standard_error = std / math.Sqrt(float64(len(successCounts)))
		if standard_error <= 0.01 && attackCount >= 100 { // Stop attack earliy if results stabilise
			attackCount++
			break
		}
	}
	return successCounts
}

func attackParties(P []*Party, currentEncRatio float64) (attackSuccessNum int) {
	attackSuccessNum = 0

	var valid = false
	var randomParty int
	var randomStart int

	if uniqueATD == 0 { // If the user wants the attacker block to be randomly chosen.
		randomParty = getRandom(maxHouseholdsNumber)
		randomStart = getRandomStart(randomParty)
	} else {
		for !valid {
			randomParty = getRandom(maxHouseholdsNumber)
			randomStart = getRandomStart(randomParty)
			var attacker_data_block = P[randomParty].rawInput[randomStart : randomStart+atdSize]
			if uniqueDataBlock(P, attacker_data_block, randomParty, randomStart, "rawInput") {
				valid = true
			}
		}
	}

	var attacker_data_block = P[randomParty].rawInput[randomStart : randomStart+atdSize]

	var matched_households = identifyParty(P, attacker_data_block, randomParty, randomStart, currentEncRatio)
	if len(matched_households) == 1 && matched_households[0] == randomParty {
		attackSuccessNum++
	}
	return
}

// getRandomStart is a helper function that returns an unused random start block/section for the Party
func getRandomStart(party int) int {
	var valid bool = false

	var randomStart int

	for !valid {
		randomStart = getRandom(MAX_PARTY_ROWS - atdSize)
		if !contains(party, randomStart) {
			usedRandomStartPartyPairs[party] = append(usedRandomStartPartyPairs[party], randomStart)
			valid = true
		}
	}
	return randomStart
}

// contains is a helper function that checks if the Party has used the random start block/section before.
func contains(party int, randomStart int) bool {
	var contains bool = false

	val, exists := usedRandomStartPartyPairs[party]

	if exists {
		for _, v := range val {
			if v == randomStart {
				contains = true
			}
		}
	}

	return contains
}

// uniqueDataBlock is a helper function that checks if the data block is unique in the dataset.
func uniqueDataBlock(P []*Party, arr []float64, party int, index int, input_type string) bool {
	var unique bool = true

	for pn, po := range P {
		if pn == party {
			continue
		}
		if input_type == "rawInput" {
			var household_data = po.rawInput
			for i := 0; i < len(household_data)-atdSize+1; i++ {
				var target = household_data[i : i+atdSize]
				if reflect.DeepEqual(target, arr) {
					unique = false
					usedRandomStartPartyPairs[pn] = append(usedRandomStartPartyPairs[pn], i)
					break
				}
			}
		} else {
			for _, household_section := range po.encryptedInput {
				for i := 0; i < len(household_section)-atdSize+1; i++ {
					var target = household_section[i : i+atdSize]
					if reflect.DeepEqual(target, arr) {
						unique = false
						usedRandomStartPartyPairs[pn] = append(usedRandomStartPartyPairs[pn], i)
						break
					}
				}
				if !unique {
					break
				}
			}
		}
		if !unique {
			break
		}
	}
	return unique
}

func identifyParty(P []*Party, attackerRawBlock []float64, originalPartyIdx int, originalRawIndex int, currentEncRatio float64) []int {
	var matched_households = []int{}

	var precision float64
	var epsilon float64
	if currentEncRatio <= 0.3 {
		precision = 100
		epsilon = 0.5 / precision
	} else if currentEncRatio <= 0.7 {
		precision = 10
		epsilon = 0.5 / precision
	} else {
		precision = 1
		epsilon = 0.5 / precision
	}

	simulatedEncryptedBlock := make([]float64, len(attackerRawBlock))
	copy(simulatedEncryptedBlock, attackerRawBlock)

	for j := range simulatedEncryptedBlock {
		simulatedEncryptedBlock[j] *= (1.0 - currentEncRatio)
		simulatedEncryptedBlock[j] = float64(math.Round(simulatedEncryptedBlock[j]*precision) / precision)
	}

	// Minimum length of the array to be considered a match.
	var min_length int = int(math.Ceil(float64(len(simulatedEncryptedBlock)) * float64(min_percent_matched) / 100))

	// Iterate through ALL parties and ALL their encrypted data blocks to find a match.
	for partyIdx, currentParty := range P {
		for sectionIdx := 0; sectionIdx < len(currentParty.encryptedInput); sectionIdx++ {
			encryptedSection := currentParty.encryptedInput[sectionIdx]

			// Iterate through all possible starting positions for a block of `atdSize` within this section.
			for startOffset := 0; startOffset <= len(encryptedSection)-atdSize; startOffset++ {
				candidateEncryptedBlock := encryptedSection[startOffset : startOffset+atdSize]

				var isBlockMatch bool

				if min_length == len(simulatedEncryptedBlock) { // Case: Exact match required.
					isBlockMatch = compareFloatSlices(simulatedEncryptedBlock, candidateEncryptedBlock, epsilon)
				} else { // Case: Percentage match allowed
					matchCount := 0
					mismatchCount := 0
					for k := 0; k < len(simulatedEncryptedBlock); k++ {
						if math.Abs(simulatedEncryptedBlock[k]-candidateEncryptedBlock[k]) < epsilon {
							matchCount++
						} else {
							mismatchCount++
						}
						if mismatchCount > (len(simulatedEncryptedBlock) - min_length) {
							break
						}
					}
					isBlockMatch = float64(matchCount)/float64(len(simulatedEncryptedBlock)) >= float64(min_percent_matched)/100.0
				}

				if isBlockMatch {
					if uniqueATD == 0 {
						if uniqueDataBlocks(P, [][]float64{candidateEncryptedBlock}, partyIdx, startOffset, min_length, epsilon) {
							matched_households = append(matched_households, partyIdx)
							goto NextParty // Found a unique match for this party, move to next party
						}
					} else {
						matched_households = append(matched_households, partyIdx)
						goto NextParty // Found a match, move to next party
					}
				}
			}
		}
	NextParty:
	}

	// Deduplicate matched_households if a Party can be matched multiple times for unique IDs
	uniqueMatches := make(map[int]bool)
	var finalMatchedHouseholds []int
	for _, id := range matched_households {
		if !uniqueMatches[id] {
			uniqueMatches[id] = true
			finalMatchedHouseholds = append(finalMatchedHouseholds, id)
		}
	}

	return finalMatchedHouseholds
}

// compareFloatSlices checks if two float64 slices are approximately equal within an epsilon.
func compareFloatSlices(slice1, slice2 []float64, epsilon float64) bool {
	if len(slice1) != len(slice2) {
		return false
	}
	for i := 0; i < len(slice1); i++ {
		if math.Abs(slice1[i]-slice2[i]) > epsilon {
			return false
		}
	}
	return true
}

// uniqueDataBlocks is a helper function that checks if the data blocks/sections is unique in the dataset.
func uniqueDataBlocks(P []*Party, pos_matches [][]float64, party int, index int, min_length int, epsilon float64) bool {
	var unique bool = true

	for pn, po := range P {
		if pn == party {
			continue
		}
		for _, household_section := range po.encryptedInput {
			for i := 0; i <= len(household_section)-min_length; i++ {
				var target = household_section[i : i+min_length]

				for _, pos_match := range pos_matches {
					if compareFloatSlices(target, pos_match, epsilon) {
						unique = false
						break // Exit inner loop (no need to check other pos_matches)
					}
				}
				if !unique {
					break // Break middle loop (no need to check other blocks in this section)
				}
			}
			if !unique {
				break // Break outer loop (no need to check other sections of this Party)
			}
		}
		if !unique {
			break // Break outermost loop (no need to check other parties)
		}
	}
	return unique
}

func calculateStandardDeviationAndMeanAndVariance(numbers []float64) (standardDeviation, mean, variance float64) {
	var sum float64
	for _, num := range numbers {
		sum += num
	}
	mean = sum / float64(len(numbers))

	var squaredDifferences float64
	for _, num := range numbers {
		difference := num - mean
		squaredDifferences += difference * difference
	}

	variance = squaredDifferences / (float64(len(numbers)) - 1)

	standardDeviation = math.Sqrt(variance)

	return standardDeviation, mean, variance
}
