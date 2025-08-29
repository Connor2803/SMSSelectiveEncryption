/*
running command: // dataset, leaked plaintext size
go run .\WATER_household_level_encryption_ratio_selector\generate_household_level_metrics.go 1 12
> Run this code for the water dataset.
*/

package main

import (
	"encoding/csv"
	"errors"
	"fmt"
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

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/drlwe"
	"github.com/tuneinsight/lattigo/v4/rlwe"
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
	flag           []int       // Slice to track which blocks have been encrypted
	group          []int
	entropy        []float64   // Entropy for block
	encryptedInput [][]float64 // Transformed data for each block after encryption
} // Struct to represent an individual household

type BlocksMetrics struct {
	HouseholdCSVFileName                    string
	BlockNumber                             int
	DateRange                               string // DD-MM-YYYY to DD-MM-YYYY
	BlockSumUsage                           float64
	EncryptionRatio                         float64
	RawEntropy                              float64 // Per-block, pre-encryption entropy value
	RemainingEntropy                        float64 // Per-block, post-encryption entropy value
	ReidentificationRate                    float64
	ReidentificationStandardError           float64
	ReidentificationAttackDuration          time.Duration // Total time for the entire attack to run (in seconds).
	AvgPerReidentificationAttackRunDuration float64       // Average time for a single attack to run (in seconds).
	ProgramAllocatedMemoryMiB               float64       // Program's allocated memory in MiB for this ratio
} // Struct to represent a block in an individual household

type ResultKey struct {
	HouseholdCSVFileName string
	BlockNumber          int
	EncryptionRatio      float64
} // Struct to represent the unique identifier for a block.

const DatasetWater = 1
const DatasetElectricity = 2

const MaxPartyRows = 10240 // Total records/meter readings per household (WATER dataset the fewest row count: 20485, WATER dataset greatest row count: 495048, ELECTRICITY dataset fewest row count: 19188, ELECTRICITY dataset greatest row count: 19864)
const BlockSize = 1024     // Block Size: 2048 for summation correctness, 8192 for variance correctness

var currentDataset int       // Water(1), Electricity(2)
var maxHouseholdsNumber = 70 // Since last 10 households are fixed testing group.

var blockNum int // Number of blocks/block of data for each household, i.e., MAXPARTYROWS/BlockSize
var globalPartyRows = -1

var maxReidentificationAttempts = 1000 // Default value for number of loops for membershipIdentificationAttack.
var leakedPlaintextSize = 12           // Element number of unique attacker data
var usedRandomStartPartyPairs = map[int][]int{}
var minPercentMatched = 90 // Default value for minimum percentage match for identification.

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

	if len(args) > 1 {
		currentDataset = args[0]
		leakedPlaintextSize = args[1]
	}
	if currentDataset == 1 {
		fmt.Printf("Using dataset: Water and leakedPlaintextSize: %d\n", leakedPlaintextSize)
	} else {
		fmt.Printf("Using dataset: Electricity and leakedPlaintextSize: %d\n", leakedPlaintextSize)
	}

	var metricsOutputFileName string
	var partyMetricsOutputFileName string

	if currentDataset == DatasetWater {
		fmt.Printf("Selected DatasetWater")
		metricsOutputFileName = "./WATER_household_level_encryption_ratio_selector/ML_metrics_WATER.csv"
		partyMetricsOutputFileName = "./WATER_household_level_encryption_ratio_selector/ML_party_metrics_WATER.csv"
	} else {
		fmt.Printf("Selected DatasetElectricity")
		metricsOutputFileName = "./ELECTRICITY_household_level_encryption_ratio_selector/ML_metrics_ELECTRICITY.csv"
		partyMetricsOutputFileName = "./ELECTRICITY_household_level_encryption_ratio_selector/ML_party_metrics_ELECTRICITY.csv"
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
	if currentDataset == DatasetWater {
		path = fmt.Sprintf(pathFormat, "water", MaxPartyRows)
	} else {
		path = fmt.Sprintf(pathFormat, "electricity", MaxPartyRows)
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
		"block",
		"date_range",
		"block_sum_usage",
		"encryption_ratio",
		"block_raw_entropy",
		"block_remaining_entropy",
		"reidentification_mean",
		"reidentification_standard_error",
		"reidentification_attack_duration",
		"avg_time_per_reidentification_attack_run",
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

	// Populate Party.rawInput and sets the global blockNum.
	_, expAverage, _, _, _, _ := genInputs(P, &[]BlocksMetrics{})

	allResults := make(map[ResultKey]BlocksMetrics)

	// --- 2. Processing Phase ---
	encryptionRatios := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}

	for _, encRatio := range encryptionRatios {
		for _, party := range P {
			for blockIdx := 0; blockIdx < blockNum; blockIdx++ {

				// Get the original raw data for the current block.
				start := blockIdx * BlockSize
				end := (blockIdx + 1) * BlockSize
				if end > len(party.rawInput) {
					end = len(party.rawInput)
				}
				rawBlockData := party.rawInput[start:end]
				rawBlockDates := party.rawDates[start:end]

				// Calculate raw entropy directly from the raw data.
				rawEntropy := calculateEntropy(rawBlockData)

				// Calculate sum of all the utility meter readings from a block.
				blockSumUsage := 0.0
				for _, val := range rawBlockData {
					blockSumUsage += val
				}

				// Determine Date Range for block
				var dateRange string
				if len(rawBlockDates) > 0 {
					firstDate := rawBlockDates[0]
					lastDate := rawBlockDates[len(rawBlockDates)-1]
					dateRange = fmt.Sprintf("%s to %s", firstDate, lastDate)
				} else {
					dateRange = "N/A"
				}

				// Create a temporary copy to apply the encryption ratio to.
				tempBlockInput := make([]float64, len(rawBlockData))
				copy(tempBlockInput, rawBlockData)

				for j := range tempBlockInput {
					tempBlockInput[j] *= (1.0 - encRatio)

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
					tempBlockInput[j] = float64(math.Round(tempBlockInput[j]*float64(precision)) / float64(precision))
				}

				// Store the transformed data in Party.encryptedInput for attack
				party.encryptedInput[blockIdx] = tempBlockInput

				// Calculate remaining entropy from the modified data.
				remainingEntropy := calculateEntropy(tempBlockInput)

				// Create the key for the map
				key := ResultKey{
					HouseholdCSVFileName: filepath.Base(party.filename),
					BlockNumber:          blockIdx,
					EncryptionRatio:      encRatio,
				}

				// Append the complete result for this data point.
				allResults[key] = BlocksMetrics{
					HouseholdCSVFileName: party.filename,
					BlockNumber:          blockIdx,
					DateRange:            dateRange,
					BlockSumUsage:        blockSumUsage,
					EncryptionRatio:      encRatio,
					RawEntropy:           rawEntropy,
					RemainingEntropy:     remainingEntropy,
					// REMINDER: ASR metrics will be updated after the runReidentificationAttack call for this ratio
				}
			}
		}

		// --- 2.5 Homomorphic Computations Phase --
		// Populate party.input and party.plainInput with the raw data for HE operations.
		// party.input slice  holds blocks intended for HE operations.
		// party.plainInput slice holds blocks intended for non-HE processing.
		// NOTE: The population of these slices are deterministic, i.e., the first numHEBlocks are sent for HE operations
		// which is based on the proportion of the blocks encrypted using the encRatio,
		// i.e., encRatio of 0.4 would encrypt the first 40% of the blocks within a party/household.
		expSummationForRatio := make([]float64, len(P))
		expDeviationForRatio := make([]float64, len(P))
		plainSum := make([]float64, len(P))

		for pi, po := range P {
			po.input = make([][]float64, 0)
			po.plainInput = make([]float64, 0)

			numHEBlocks := int(float64(blockNum) * encRatio)
			totalBlocks := len(po.rawInput) / BlockSize

			heInputDataFlat := []float64{}

			for s := 0; s < totalBlocks; s++ {
				blockStart := s * BlockSize
				blockEnd := (s + 1) * BlockSize
				if blockEnd > len(po.rawInput) {
					blockEnd = len(po.rawInput)
				}
				currentBlockData := po.rawInput[blockStart:blockEnd]

				if s < numHEBlocks {
					// This block goes to HE
					po.input = append(po.input, currentBlockData)
					heInputDataFlat = append(heInputDataFlat, currentBlockData...)
				} else {
					// This block goes to plaintext
					po.plainInput = append(po.plainInput, currentBlockData...)

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
		currentRatioReidentificationAttackSuccessResults := []float64{} // Contains the results of each attack loop for a given encryption ratio.

		reidPhaseStartTime := time.Now()

		// After all households/parties and blocks have had the current encryption ratio applied,
		// Run the member identification attack.
		fmt.Printf("\n--- Starting Attack Phase for Encryption Ratio: %.2f ---\n", encRatio)
		currentRatioReidentificationAttackSuccessResults = runReidentificationAttack(P, encRatio)

		// Calculate ASR mean and standard error for the current encryption ratio
		reidRaterMean := 0.0
		reidRateStdError := 0.0

		if len(currentRatioReidentificationAttackSuccessResults) > 0 {
			// Convert success counts to ASRs (Attack Success Rates)
			var reidRates []float64
			totalAttackableInstancesPerRun := float64(len(P))
			if totalAttackableInstancesPerRun == 0 {
				fmt.Println("ERROR: totalAttackableInstancesPerRun is 0. Cannot calculate ASR.")
			} else {
				for _, successCount := range currentRatioReidentificationAttackSuccessResults {
					reidRates = append(reidRates, successCount/totalAttackableInstancesPerRun)
				}
			}

			stdDev, mean, _ := calculateStandardDeviationAndMeanAndVariance(reidRates)
			reidRaterMean = mean
			reidRateStdError = stdDev / math.Sqrt(float64(len(reidRates)))
			fmt.Printf("Final Reidentification Rate: %.6f, Reidentification Standard Error: %.6f\n", reidRaterMean, reidRateStdError)
		} else {
			fmt.Println("WARNING: currentRatioReidentificationAttackSuccessResults is empty. ASR cannot be calculated.")
		}

		reidPhaseEndTime := time.Now()
		reidAttackDuration := reidPhaseEndTime.Sub(reidPhaseStartTime)
		avgTimePerAttackRun := reidAttackDuration.Seconds() / float64(maxReidentificationAttempts)

		for key, result := range allResults {
			if key.EncryptionRatio == encRatio {
				// Note: Go maps store copies of values, so you must reassign the modified struct
				tempResult := result
				tempResult.ReidentificationRate = reidRaterMean
				tempResult.ReidentificationStandardError = reidRateStdError
				tempResult.ReidentificationAttackDuration = reidAttackDuration
				tempResult.AvgPerReidentificationAttackRunDuration = avgTimePerAttackRun
				tempResult.ProgramAllocatedMemoryMiB = currentAllocatedMemMiB

				allResults[key] = tempResult
			}
		}
	}

	// --- 4. Writing Phase ---
	for _, result := range allResults {
		err := metricsWriter.Write([]string{
			filepath.Base(result.HouseholdCSVFileName),
			strconv.Itoa(result.BlockNumber),
			result.DateRange,
			fmt.Sprintf("%.2f", result.BlockSumUsage),
			fmt.Sprintf("%.2f", result.EncryptionRatio),
			fmt.Sprintf("%.6f", result.RawEntropy),
			fmt.Sprintf("%.6f", result.RemainingEntropy),
			fmt.Sprintf("%.6f", result.ReidentificationRate),
			fmt.Sprintf("%.6f", result.ReidentificationStandardError),
			fmt.Sprintf("%.6f", result.ReidentificationAttackDuration.Seconds()),
			fmt.Sprintf("%.6f", result.AvgPerReidentificationAttackRunDuration),
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
func genInputs(P []*Party, metrics *[]BlocksMetrics) (expSummation, expAverage, expDeviation []float64, min, max, entropySum float64) {
	globalPartyRows = -1
	blockNum = 0
	min = math.MaxFloat64
	max = float64(-1)
	frequencyMap := map[float64]int{}
	entropyMap := map[float64]float64{}

	entropySum = 0.0

	for pi, po := range P {
		// Setup (rawInput, flags etc.)
		partyRows, partyDates := parseCSV(po.filename)
		if currentDataset == DatasetWater { // Reverse chronological order
			for i, j := 0, len(partyRows)-1; i < j; i, j = i+1, j-1 {
				partyRows[i], partyRows[j] = partyRows[j], partyRows[i]
				partyDates[i], partyDates[j] = partyDates[j], partyDates[i]
			}
		}
		lenPartyRows := len(partyRows)
		if lenPartyRows > MaxPartyRows {
			lenPartyRows = MaxPartyRows
		}
		if globalPartyRows == -1 {
			blockNum = lenPartyRows / BlockSize // blockNum = 10, since lenPartyRows = 10240, BlockSize = 1024
			if lenPartyRows%BlockSize != 0 {
				blockNum++
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
		po.flag = make([]int, blockNum)        // Marked blocks for encryption.
		po.entropy = make([]float64, blockNum) // Entropy values for each block of a particular party/household.
		po.encryptedInput = make([][]float64, blockNum)
		po.group = make([]int, BlockSize)

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
		totalRecords := maxHouseholdsNumber * MaxPartyRows
		for k, v := range frequencyMap {
			possibility := float64(v) / float64(totalRecords)
			entropyMap[k] = -possibility * math.Log2(possibility)
		}

		for i := range po.rawInput {
			usage := po.rawInput[i]
			singleRecordEntropy := entropyMap[usage] / float64(frequencyMap[usage])
			po.entropy[i/BlockSize] += singleRecordEntropy
			entropySum += singleRecordEntropy // Global all data total entropy
		}
		po.flag = make([]int, blockNum) // Reset flag for every Party

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
	for i, partyBlocks := range encInputsSummation {
		if len(partyBlocks) == 0 {
			continue // Skip if no blocks for this party.
		}
		partySummationOpsStart := time.Now()

		tmpCiphertext := partyBlocks[0]
		for j := 1; j < len(partyBlocks); j++ {
			evaluator.Add(tmpCiphertext, partyBlocks[j], tmpCiphertext)
		}

		evaluator.InnerSum(tmpCiphertext, 1, params.Slots(), tmpCiphertext)

		summationOperationTimes[i] = time.Since(partySummationOpsStart).Nanoseconds()
		encSummationOuts[i] = tmpCiphertext
	}

	// Deviation calculation ====================================================
	encDeviationOuts := make([]*rlwe.Ciphertext, len(P))
	for i, partyBlocks := range encInputsNegative {
		if len(partyBlocks) == 0 || encSummationOuts[i] == nil {
			continue
		}
		partyDeviationOpsStart := time.Now()

		avgCiphertext := encSummationOuts[i].CopyNew()
		avgCiphertext.Scale = avgCiphertext.Scale.Mul(rlwe.NewScale(globalPartyRows))

		var aggregatedDeviation *rlwe.Ciphertext

		for j, blockCipher := range partyBlocks {
			// Step 1: Subtract the average from the value (by adding the negative value)
			currentDeviation := evaluator.AddNew(blockCipher, avgCiphertext)

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

func runReidentificationAttack(P []*Party, currentEncRatio float64) []float64 {
	var attackCount int
	var successCounts = []float64{} // Collects the number of successful attacks for each loop.

	for attackCount = 0; attackCount < maxReidentificationAttempts; attackCount++ {
		var successNum = identifySourceHousehold(P, currentEncRatio) // Integer result of one attack run.
		successCounts = append(successCounts, float64(successNum))   // Stores the success count for this run.

		// NOTE: These calculations are for internal stopping conditions, not for statistical purposes.
		std, _, _ := calculateStandardDeviationAndMeanAndVariance(successCounts)
		standardError := std / math.Sqrt(float64(len(successCounts)))
		if standardError <= 0.01 && attackCount >= 100 { // Stop attack early if results stabilise.
			attackCount++
			break
		}
	}
	return successCounts
}

func identifySourceHousehold(P []*Party, currentEncRatio float64) (attackSuccessNum int) {
	attackSuccessNum = 0
	var randomParty int
	var randomStart int
	var attackerDataBlock []float64
	var valid = false

	for !valid {
		randomParty = getRandom(maxHouseholdsNumber)
		randomStart = getRandomStart(randomParty)
		if randomStart+leakedPlaintextSize > len(P[randomParty].rawInput) {
			continue
		}
		attackerDataBlock = P[randomParty].rawInput[randomStart : randomStart+leakedPlaintextSize]
		if uniqueDataBlock(P, attackerDataBlock, randomParty, randomStart, "rawInput") {
			valid = true
		}
	}

	isSourceIdentifiable := isPartyIdentifiable(P[randomParty], attackerDataBlock, currentEncRatio)

	if !isSourceIdentifiable {
		return 0
	}

	// If the source is identifiable, check if any other party could also be a match.
	matchedHouseholds := []int{randomParty} // Initialise array with leaked household.
	for partyIdx, party := range P {
		if partyIdx == randomParty {
			continue
		}
		if isPartyIdentifiable(party, attackerDataBlock, currentEncRatio) {
			matchedHouseholds = append(matchedHouseholds, partyIdx)
		}
	}

	// A successful re-identification means only the original source was matched.
	if len(matchedHouseholds) == 1 {
		attackSuccessNum = 1
	}

	return
}

// Helper function to check if a single party can be identified with the leaked block.
func isPartyIdentifiable(targetParty *Party, attackerPlaintextBlock []float64, currentEncRatio float64) bool {
	minMatchCount := math.Ceil(float64(len(attackerPlaintextBlock)) * float64(minPercentMatched) / 100.0)
	numEncryptedBlocks := int(currentEncRatio * float64(blockNum))
	firstPlaintextBlock := numEncryptedBlocks

	for s := firstPlaintextBlock; s < blockNum; s++ {
		blockStart := s * BlockSize
		if blockStart >= len(targetParty.plainInput) {
			continue
		}
		blockEnd := (s + 1) * BlockSize
		if blockEnd > len(targetParty.plainInput) {
			blockEnd = len(targetParty.plainInput)
		}
		targetBlockData := targetParty.plainInput[blockStart:blockEnd]

		if len(targetBlockData) < len(attackerPlaintextBlock) {
			continue // Block is too small
		}

		for i := 0; i <= len(targetBlockData)-len(attackerPlaintextBlock); i++ {
			targetWindow := targetBlockData[i : i+len(attackerPlaintextBlock)]
			matchCount := 0
			for k := 0; k < len(attackerPlaintextBlock); k++ {
				if math.Abs(attackerPlaintextBlock[k]-targetWindow[k]) < 1e-9 {
					matchCount++
				}
			}
			if float64(matchCount) >= minMatchCount {
				return true // A match was found
			}
		}
	}
	return false // No match found in any plaintext block
}

// getRandomStart is a helper function that returns an unused random start block for the Party
func getRandomStart(party int) int {
	var valid bool = false
	var randomStart int

	for !valid {
		randomStart = getRandom(MaxPartyRows - leakedPlaintextSize)
		if !contains(party, randomStart) {
			usedRandomStartPartyPairs[party] = append(usedRandomStartPartyPairs[party], randomStart)
			valid = true
		}
	}
	return randomStart
}

// contains is a helper function that checks if the Party has used the random start block before.
func contains(party int, randomStart int) bool {
	val, exists := usedRandomStartPartyPairs[party]

	if exists {
		for _, v := range val {
			if v == randomStart {
				return true
			}
		}
	}
	return false
}

// uniqueDataBlock is a helper function that checks if the data block is unique in the dataset.
func uniqueDataBlock(P []*Party, arr []float64, party int, index int, inputType string) bool {
	var unique bool = true

	for pn, po := range P {
		if pn == party {
			continue
		}
		if inputType == "rawInput" {
			var householdData = po.rawInput
			for i := 0; i < len(householdData)-leakedPlaintextSize+1; i++ {
				var target = householdData[i : i+leakedPlaintextSize]
				if reflect.DeepEqual(target, arr) {
					unique = false
					usedRandomStartPartyPairs[pn] = append(usedRandomStartPartyPairs[pn], i)
					break
				}
			}
		} else {
			for _, householdBlock := range po.encryptedInput {
				for i := 0; i < len(householdBlock)-leakedPlaintextSize+1; i++ {
					var target = householdBlock[i : i+leakedPlaintextSize]
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
