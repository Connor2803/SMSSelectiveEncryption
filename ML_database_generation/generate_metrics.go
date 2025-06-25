/*
running command: // dataset
go run .\ML_database_generation\generate_metrics.go 1
> Run this code for the Water dataset.
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

var elapsedSKGParty time.Duration
var elapsedPKGParty time.Duration
var elapsedRKGParty time.Duration
var elapsedRTGParty time.Duration

var elapsedEncryptParty time.Duration
var elapsedDecParty time.Duration

var elapsedAddition time.Duration
var elapsedMultiplication time.Duration
var elapsedRotation time.Duration

var elapsedSummation time.Duration
var elapsedDeviation time.Duration
var tmpTimeSummation time.Duration
var tmpTimeDeviation time.Duration

var standardErrorSummation float64
var standardErrorDeviation float64

var elapsedAnalystSummation time.Duration
var elapsedAnalystVariance time.Duration

func runTimed(f func()) time.Duration {
	start := time.Now()
	f()
	return time.Since(start)
}

func runTimedParty(f func(), N int) time.Duration {
	start := time.Now()
	f()
	return time.Duration(time.Since(start).Nanoseconds() / int64(N))
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

const MAX_PARTY_ROWS = 10240 // Total records/meter readings per household
const sectionSize = 1024     // Block Size: 2048 for summation correctness, 8192 for variance correctness

var currentDataset int // Water(1), Electricity(2)
var maxHouseholdsNumber = 80

var sectionNum int
var globalPartyRows = -1
var encryptedSectionNum int

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
	var outputFileName string

	if currentDataset == DATASET_WATER {
		outputFileName = "./ML_database_generation/ML_metrics_WATER.csv"
	} else {
		outputFileName = "./ML_database_generation/ML_metrics_ELECTRICITY.csv"
	}
	outputFile, err := os.Create(outputFileName)
	check(err)
	defer outputFile.Close()

	writer := csv.NewWriter(outputFile)
	defer writer.Flush()

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
	process(fileList[:maxHouseholdsNumber], params, writer)
}

// process reads the input files, calculates the entropy for different
// encryption ratios, and writes the results to a CSV file.
// process reads the input files, calculates entropy, and writes the results.
func process(fileList []string, params ckks.Parameters, writer *csv.Writer) {

	// --- 1. Setup Phase ---
	err := writer.Write([]string{
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

	P := genParties(params, fileList)

	// Populate Party.rawInput and sets the global sectionNum.
	genInputs(P, &[]SectionsMetrics{})

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

					// Apply quantisation (coarser rounding) based on the encryption rati
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

			stdDev, mean := calculateStandardDeviationAndMean(asrRates)
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
		err := writer.Write([]string{
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
	writer.Flush()
}

// This function encrypts each section individually across all parties at a given encryption ratio.
// It applies the ratio to a single section per Party, then computes and stores the remaining entropy
// for that section, leaving all other sections untouched. This enables ML training on section × ratio combinations.
func markEncryptedSectionsByHousehold(encRatio float64, P []*Party, metrics *[]SectionsMetrics) {
	for pi, po := range P {
		// Loop through each section to apply the encryption ratio and calculate entropy
		for sectionIdx := 0; sectionIdx < sectionNum; sectionIdx++ {
			// Create a temporary copy of the raw input for this section
			tempSectionInput := make([]float64, sectionSize)
			copy(tempSectionInput, po.rawInput[sectionIdx*sectionSize:(sectionIdx+1)*sectionSize])

			// Apply the encryption ratio to the temporary section data
			for i := range tempSectionInput {
				tempSectionInput[i] *= (1.0 - encRatio)
			}

			// Calculate the remaining entropy for this modified section
			sectionEntropy := 0.0
			frequency := make(map[float64]int)
			for _, val := range tempSectionInput {
				roundedVal := math.Round(val*1000) / 1000
				frequency[roundedVal]++
			}

			total := float64(len(tempSectionInput))
			for _, count := range frequency {
				if count > 0 {
					p := float64(count) / total
					sectionEntropy -= p * math.Log2(p)
				}
			}

			// Update the metrics for the current section
			metricIndex := pi*sectionNum + sectionIdx
			if metricIndex < len(*metrics) {
				(*metrics)[metricIndex].RemainingEntropy = sectionEntropy
				(*metrics)[metricIndex].EncryptionRatio = encRatio
			}
		}
	}
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

func showHomomorphicMeasure(loop int, params ckks.Parameters) {

	// fmt.Println("1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	fmt.Printf("standardError: %.3f, %.3f\n", standardErrorSummation, standardErrorDeviation)
	fmt.Printf("***** Evaluating Summation/Deviation time for %d households in thirdparty analyst's side: %s,%s\n", maxHouseholdsNumber, time.Duration(elapsedSummation.Nanoseconds()/int64(loop)), time.Duration(elapsedDeviation.Nanoseconds()/int64(loop)))

	// fmt.Println("2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

	// //public key & relinearization key & rotation key
	// fmt.Printf("*****Amortized SKG Time: %s\n", time.Duration(elapsedSKGParty.Nanoseconds()/int64(loop)))
	// fmt.Printf("*****Amortized PKG Time: %s\n", time.Duration(elapsedPKGParty.Nanoseconds()/int64(loop)))
	// fmt.Printf("*****Amortized RKG Time: %s\n", time.Duration(elapsedRKGParty.Nanoseconds()/int64(loop)))
	// fmt.Printf("*****Amortized RTG Time: %s\n", time.Duration(elapsedRTGParty.Nanoseconds()/int64(loop)))

	// //single operation, independent of households' size
	// fmt.Printf("*****Amortized Encrypt Time: %s\n", time.Duration(elapsedEncryptParty.Nanoseconds()/int64(loop)))
	// fmt.Printf("*****Amortized Decrypt Time: %s\n", time.Duration(elapsedDecParty.Nanoseconds()/int64(loop)))
	// fmt.Printf("*****Amortized Ciphertext Addition Time: %s\n", time.Duration(elapsedAddition.Nanoseconds()/int64(loop)))
	// fmt.Printf("*****Amortized Ciphertext Multiplication Time: %s\n", time.Duration(elapsedMultiplication.Nanoseconds()/int64(loop)))
	// fmt.Printf("*****Amortized Ciphertext Rotation Time: %s\n", time.Duration(elapsedRotation.Nanoseconds()/int64(loop*len(params.GaloisElementsForRowInnerSum()))))

	// fmt.Printf("*****Amortized Analyst Time: %s\n", time.Duration(elapsedAnalystSummation.Nanoseconds()/int64(loop)))
	// fmt.Printf("*****Amortized Analyst Time: %s\n", time.Duration(elapsedAnalystVariance.Nanoseconds()/int64(loop)))

	// PrintMemUsage()
}

func doHomomorphicOperations(params ckks.Parameters, P []*Party, expSummation, expAverage, expDeviation, plainSum []float64) {
	// Target private and public keys
	tkgen := ckks.NewKeyGenerator(params)
	var tsk *rlwe.SecretKey
	var tpk *rlwe.PublicKey
	elapsedSKGParty += runTimed(func() {
		tsk = tkgen.GenSecretKey()
	})
	elapsedPKGParty += runTimed(func() {
		tpk = tkgen.GenPublicKey(tsk)
	})

	var rlk *rlwe.RelinearizationKey
	elapsedRKGParty += runTimed(func() {
		rlk = tkgen.GenRelinearizationKey(tsk, 1)
	})

	rotations := params.RotationsForInnerSum(1, globalPartyRows)
	var rotk *rlwe.RotationKeySet
	elapsedRTGParty += runTimed(func() {
		rotk = tkgen.GenRotationKeysForRotations(rotations, false, tsk)
	})

	decryptor := ckks.NewDecryptor(params, tsk)
	encoder := ckks.NewEncoder(params)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotk})

	//generate ciphertexts====================================================
	encInputsSummation, encInputsNegative := encPhase(params, P, tpk, encoder)

	// summation calculation====================================================
	encSummationOuts := make([]*rlwe.Ciphertext, len(P))
	anaTime1 := time.Now()
	var tmpCiphertext *rlwe.Ciphertext
	for i, _ := range encInputsSummation {
		for j, _ := range encInputsSummation[i] {
			if j == 0 {
				tmpCiphertext = encInputsSummation[i][j]
			} else {
				elapsedSummation += runTimed(func() {
					evaluator.Add(tmpCiphertext, encInputsSummation[i][j], tmpCiphertext)
				})
			}

			if j == len(encInputsSummation[i])-1 {
				elapsedSummation += runTimed(func() {
					elapsedRotation += runTimedParty(func() {
						evaluator.InnerSum(tmpCiphertext, 1, params.Slots(), tmpCiphertext)
					}, len(P))
				})
				encSummationOuts[i] = tmpCiphertext
			}
		} //j
	} //i
	elapsedAnalystSummation += time.Since(anaTime1)

	// deviation calculation====================================================
	encDeviationOuts := make([]*rlwe.Ciphertext, len(P))
	anaTime2 := time.Now()
	var avergeCiphertext *rlwe.Ciphertext
	for i, _ := range encInputsNegative {
		for j, _ := range encInputsNegative[i] {
			if j == 0 {
				avergeCiphertext = encSummationOuts[i].CopyNew()
				avergeCiphertext.Scale = avergeCiphertext.Mul(rlwe.NewScale(globalPartyRows))
			}
			elapsedDeviation += runTimed(func() {
				elapsedAddition += runTimedParty(func() {
					evaluator.Add(encInputsNegative[i][j], avergeCiphertext, encInputsNegative[i][j])
				}, len(P))

				elapsedMultiplication += runTimedParty(func() {
					evaluator.MulRelin(encInputsNegative[i][j], encInputsNegative[i][j], encInputsNegative[i][j])
				}, len(P))

				if j == 0 {
					tmpCiphertext = encInputsNegative[i][j]
				} else {
					elapsedRotation += runTimedParty(func() {
						evaluator.Add(tmpCiphertext, encInputsNegative[i][j], tmpCiphertext)
					}, len(P))
				}

				if j == len(encInputsNegative[i])-1 {
					elapsedRotation += runTimedParty(func() {
						evaluator.InnerSum(tmpCiphertext, 1, params.Slots(), tmpCiphertext)
					}, len(P))
					tmpCiphertext.Scale = tmpCiphertext.Mul(rlwe.NewScale(globalPartyRows))
					encDeviationOuts[i] = tmpCiphertext
				}
			})
		} //j
	} //i
	elapsedAnalystVariance += time.Since(anaTime2)

	// Decrypt & Print====================================================
	// fmt.Println("> Decrypt & Result:>>>>>>>>>>>>>")

	// print summation
	ptresSummation := ckks.NewPlaintext(params, params.MaxLevel())
	for i, _ := range encSummationOuts {
		if encSummationOuts[i] != nil {
			decryptor.Decrypt(encSummationOuts[i], ptresSummation) //ciphertext->plaintext
			encoder.Decode(ptresSummation, params.LogSlots())      //resSummation :=
			// fmt.Printf("CKKS Summation of Party[%d]=%.6f\t", i, real(resSummation[0])+plainSum[i])
			// fmt.Printf(" <===> Expected Summation of Party[%d]=%.6f\t", i, expSummation[i])
			// fmt.Println()
		}
	}

	// print deviation
	ptresDeviation := ckks.NewPlaintext(params, params.MaxLevel())
	for i, _ := range encDeviationOuts {
		if encDeviationOuts[i] != nil {
			elapsedDecParty += runTimedParty(func() {
				// decryptor.Decrypt(encAverageOuts[i], ptres)            //ciphertext->plaintext
				decryptor.Decrypt(encDeviationOuts[i], ptresDeviation) //ciphertext->plaintext
			}, len(P))

			// res := encoder.Decode(ptres, params.LogSlots())
			encoder.Decode(ptresDeviation, params.LogSlots()) //resDeviation :=

			// calculatedAverage := real(res[0])
			// calculatedAverage := expAverage[i]

			// fmt.Printf("CKKS Average of Party[%d]=%.6f\t", i, calculatedAverage)
			// fmt.Printf(" <===> Expected Average of Party[%d]=%.6f\t", i, expAverage[i])
			// fmt.Println()

			//extra value for deviation
			// delta := calculatedAverage * calculatedAverage * float64(len(resDeviation)-globalPartyRows) / float64(globalPartyRows)

			// fmt.Printf("CKKS Deviation of Party[%d]=%.6f\t", i, real(resDeviation[0])) //real(resDeviation[0])-delta
			// fmt.Printf(" <===> Expected Deviation of Party[%d]=%.6f\t", i, expDeviation[i])
			// fmt.Println()
		}
	}
	// fmt.Printf("\tDecrypt Time: done %s\n", elapsedDecParty)
	// fmt.Println()

	//print result
	// visibleNum := 4
	// fmt.Println("> Parties:")
	//original data
	// for i, pi := range P {
	// 	fmt.Printf("Party %3d(%d):\t\t", i, len(pi.input))
	// 	for j, element := range pi.input {
	// 		if j < visibleNum || (j > globalPartyRows-visibleNum && j < globalPartyRows) {
	// 			fmt.Printf("[%d]%.6f\t", j, element)
	// 		}
	// 	}
	// 	fmt.Println()
	// }

}

func encPhase(params ckks.Parameters, P []*Party, pk *rlwe.PublicKey, encoder ckks.Encoder) (encInputsSummation, encInputsNegative [][]*rlwe.Ciphertext) {

	encInputsSummation = make([][]*rlwe.Ciphertext, len(P))
	encInputsNegative = make([][]*rlwe.Ciphertext, len(P))

	// Each party encrypts its input vector
	// fmt.Println("> Encrypt Phase<<<<<<<<<<<<<<<<<<")
	encryptor := ckks.NewEncryptor(params, pk)
	pt := ckks.NewPlaintext(params, params.MaxLevel())

	elapsedEncryptParty += runTimedParty(func() {
		for pi, po := range P {
			for j, _ := range po.input {
				if j == 0 {
					encInputsSummation[pi] = make([]*rlwe.Ciphertext, 0)
					encInputsNegative[pi] = make([]*rlwe.Ciphertext, 0)
				}
				//Encrypt
				encoder.Encode(po.input[j], pt, params.LogSlots())
				tmpCiphertext := ckks.NewCiphertext(params, 1, params.MaxLevel())
				encryptor.Encrypt(pt, tmpCiphertext)
				encInputsSummation[pi] = append(encInputsSummation[pi], tmpCiphertext)

				//turn po.input to negative
				for k, _ := range po.input[j] {
					po.input[j][k] *= -1
				}
				//Encrypt
				encoder.Encode(po.input[j], pt, params.LogSlots())
				tmpCiphertext = ckks.NewCiphertext(params, 1, params.MaxLevel())
				encryptor.Encrypt(pt, tmpCiphertext)
				encInputsNegative[pi] = append(encInputsNegative[pi], tmpCiphertext)
				////turn po.input to positive
				for k, _ := range po.input[j] {
					po.input[j][k] *= -1
				}
			}
		}
	}, 2*len(P)) //2 encryption in function

	// fmt.Printf("\tdone  %s\n", elapsedEncryptParty)

	return
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
		std, _ = calculateStandardDeviationAndMean(successCounts)
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
					// If uniqueATD == 0, check for uniqueness of the *candidateEncryptedBlock* among *other parties' encrypted data*
					// To check if this specific block that matched is "unique enough" to count as a successful identification.
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
			continue // Skip the current Party, as we're checking uniqueness against *other* parties
		}
		for _, household_section := range po.encryptedInput { // Iterate through each section of the other Party's encrypted input
			for i := 0; i <= len(household_section)-min_length; i++ {
				var target = household_section[i : i+min_length] // A block from another Party's encrypted data

				for _, pos_match := range pos_matches {
					if compareFloatSlices(target, pos_match, epsilon) {
						unique = false // Found a duplicate in another Party, so it's not unique.
						break          // Exit inner loop (no need to check other pos_matches)
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

func calculateStandardDeviationAndMean(numbers []float64) (float64, float64) {
	var sum float64
	for _, num := range numbers {
		sum += num
	}
	mean := sum / float64(len(numbers))

	var squaredDifferences float64
	for _, num := range numbers {
		difference := num - mean
		squaredDifferences += difference * difference
	}

	variance := squaredDifferences / (float64(len(numbers)) - 1)

	standardDeviation := math.Sqrt(variance)

	return standardDeviation, mean
}
