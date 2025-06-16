/*
running command: // dataset
go run .\ML_database_generation\generate_metrics.go 1
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
	"strconv"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/drlwe"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

type party struct {
	filename    string
	sk          *rlwe.SecretKey
	rlkEphemSk  *rlwe.SecretKey
	ckgShare    *drlwe.CKGShare
	rkgShareOne *drlwe.RKGShare
	rkgShareTwo *drlwe.RKGShare
	rtgShare    *drlwe.RTGShare
	pcksShare   *drlwe.PCKSShare

	rawInput   []float64   // All data
	rawDates   []string    // Meter reading dates
	input      [][]float64 // Encryption data
	plainInput []float64   // Plaintext data
	flag       []int       // Slice to track which sections have been encrypted
	group      []int
	entropy    []float64 // Entropy for block
} // Struct to represent an individual household

type SectionsMetrics struct {
	HouseholdCSVFileName string
	SectionNumber        int
	DateRange            string
	SectionSumUsage      float64
	EncryptionRatio      float64
	RawEntropy           float64 // Per-section, pre-encryption entropy value
	RemainingEntropy     float64 // Per-section, post-encryption entropy value
	Round                int
} // Struct to represent a section/block in an individual household

const STRATEGY_GLOBAL = 1
const STRATEGY_HOUSEHOLD = 2
const STRATEGY_RANDOM = 3

const DATASET_WATER = 1
const DATASET_ELECTRICITY = 2

const MAX_PARTY_ROWS = 10240 // Total records/meter readings per household
const sectionSize = 1024     // Block Size: 2048 for summation correctness, 8192 for variance correctness

var currentStrategy = 2 // Global(1), Household(2), Random(3)
var currentTarget = 1   // Entropy-based (1), Transition-based (2)

var currentDataset int // Water(1), Electricity(2)
var maxHouseholdsNumber = 80

var sectionNum int
var globalPartyRows = -1
var encryptedSectionNum int

func main() {

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

	csvFile, err := os.Create("./ML_database_generation/metrics_new.csv")
	check(err)
	defer csvFile.Close()
	writer := csv.NewWriter(csvFile)
	defer writer.Flush()

	originalOutput := os.Stdout
	defer func() { os.Stdout = originalOutput }()
	os.Stdout = csvFile

	rand.Seed(time.Now().UnixNano())

	//start := time.Now()

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
	} else { //electricity
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
	})
	check(err)

	P := genParties(params, fileList)
	// This call now only populates party.rawInput and sets the global sectionNum.
	genInputs(P, &[]SectionsMetrics{})

	allResults := []SectionsMetrics{}

	// --- 2. Processing Phase ---
	numRatioSteps := 10

	for i := 1; i <= numRatioSteps; i++ {
		encRatio := float64(i) / float64(numRatioSteps)

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

				// Calculate Section Sum Utility Usage
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

					// Apply quantization (coarser rounding) based on the encryption ratio.
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

				// Calculate remaining entropy from the modified data.
				remainingEntropy := calculateEntropy(tempSectionInput)

				// Append the complete result for this data point.
				allResults = append(allResults, SectionsMetrics{
					HouseholdCSVFileName: party.filename,
					SectionNumber:        sectionIdx,
					DateRange:            dateRange,
					SectionSumUsage:      sectionSumUsage,
					EncryptionRatio:      encRatio,
					RawEntropy:           rawEntropy,
					RemainingEntropy:     remainingEntropy,
				})
			}
		}
	}

	// --- 3. Writing Phase ---
	for _, result := range allResults {
		err := writer.Write([]string{
			filepath.Base(result.HouseholdCSVFileName),
			strconv.Itoa(result.SectionNumber),
			result.DateRange, // New column
			fmt.Sprintf("%.2f", result.SectionSumUsage),
			fmt.Sprintf("%.2f", result.EncryptionRatio),
			fmt.Sprintf("%.6f", result.RawEntropy),
			fmt.Sprintf("%.6f", result.RemainingEntropy),
		})
		check(err)
	}
	writer.Flush()
}

/*
This function encrypts each section individually across all parties at a given encryption ratio.
It applies the ratio to a single section per party, then computes and stores the remaining entropy
for that section, leaving all other sections untouched. This enables ML training on section × ratio combinations.
*/
func markEncryptedSectionsByHousehold(encRatio float64, P []*party, metrics *[]SectionsMetrics) {
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

// Trim CSV
func resizeCSV(filename string) ([]float64, []string) {
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

func getRandom(numberRange int) (randNumber int) {
	randNumber = rand.Intn(numberRange) //[0, numberRange-1]
	return
}

// Generate parties/households
func genParties(params ckks.Parameters, fileList []string) []*party {

	// Create each party, and allocate the memory for all the shares that the protocols will need
	P := make([]*party, len(fileList))

	for i, _ := range P {
		po := &party{}
		po.sk = ckks.NewKeyGenerator(params).GenSecretKey()
		po.filename = fileList[i]
		P[i] = po
	}

	return P
}

// Generate inputs of parties/households
func genInputs(P []*party, metrics *[]SectionsMetrics) (expSummation, expAverage, expDeviation []float64, min, max, entropySum float64) {
	globalPartyRows = -1
	sectionNum = 0
	min = math.MaxFloat64
	max = float64(-1)
	frequencyMap := map[float64]int{}
	entropyMap := map[float64]float64{}

	entropySum = 0.0

	for pi, po := range P {
		// Setup (rawInput, flags etc.)
		partyRows, partyDates := resizeCSV(po.filename)
		if currentDataset == DATASET_WATER { // Reverse chronological order
			for i, j := 0, len(partyRows)-1; i < j; i, j = i+1, j-1 {
				partyRows[i], partyRows[j] = partyRows[j], partyRows[i]
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

		// Set up party structure
		po.rawInput = make([]float64, globalPartyRows)
		po.rawDates = make([]string, globalPartyRows)
		po.flag = make([]int, sectionNum)        // Marked sections for encryption
		po.entropy = make([]float64, sectionNum) // Entropy values for each section of that household
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
		po.flag = make([]int, sectionNum) // Reset flag for every party

		// Min, Max based on currentTarget which is always (1) Entropy-based.
		for _, po := range P {
			var targetArr []float64
			targetArr = po.entropy // Holds reference to entropy values for each section of that household.

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

/*
calculateEntropy computes the Shannon entropy for a slice of float64 data and returns this entropy as a float.
Shannon Entropy quantifies the average amount of "information" or "uncertainty" contained in a set of data, this means
A higher entropy value indicates that the data contains more distinct information.
*/
func calculateEntropy(data []float64) float64 {
	var entropy float64
	frequency := make(map[float64]int)
	for _, val := range data {
		// Rounding here treats inputs that are very close to each other as identical for the purpose of entropy calculations
		// By converting continious-like data into discrete "bins" before calculating probabilities.
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
