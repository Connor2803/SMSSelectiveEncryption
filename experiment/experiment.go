/*
running command: // dataset, strategy
go run .\experiment\experiment.go 1
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
	input      [][]float64 // Encryption data
	plainInput []float64   // Plaintext data
	flag       []int       // Slice to track which sections have been encrypted
	group      []int
	entropy    []float64 // Entropy for block
} // Struct to represent an individual household

type RecordMetrics struct {
	HouseholdCSVFileName              string
	MeterReadingDate                  string
	MeterReadingTime                  string
	MeterReadingValue                 float64
	IndividualReadingEntropy          float64 // Per-reading, pre-encryption entropy value
	Strategy                          string  // Global, Household, Random
	EncryptionRatio                   float64
	IndividualReadingRemainingEntropy float64 // Per-reading, post-encryption entropy value
	Round                             int
} // Struct to represent a meter reading in an individual household

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
var maxHouseholdsNumber = 1

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

	csvFile, err := os.Create("./experiment/metrics.csv")
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

// main start
func process(fileList []string, params ckks.Parameters, writer *csv.Writer) {

	allMetrics := []RecordMetrics{} // Holds all metrics from all parties

	// Write header
	err := writer.Write([]string{
		"filename", "date", "time", "utility_used",
		"individual_reading_entropy", "strategy", "encryption_ratio", "individual_reading_remaining_entropy",
	})
	check(err)

	encryptedSectionNum = MAX_PARTY_ROWS / sectionSize // MAX_PARTY_ROWS = 10240, sectionSize = 1024
	if MAX_PARTY_ROWS%sectionSize != 0 {
		encryptedSectionNum++
	}

	// Encrypt sections and update entropy
	//var entropyReduction float64
	//var prevEntropy float64

	var prevEntropy float64
	for en := 0; en < encryptedSectionNum; en++ {
		// Holds metrics from single party
		metrics := []RecordMetrics{}
		// Create each party, and allocate the memory for all the shares that the protocols will need
		P := genParties(params, fileList)
		// Get inputs and fill metrics
		genInputs(P, &metrics)

		encRatio := float64(en+1) / float64(encryptedSectionNum)
		_, entropyReduction := markEncryptedSectionsByHousehold(en, encRatio, prevEntropy, P, &metrics)
		prevEntropy = entropyReduction

		// Append a snapshot of this marking round's metrics to allMetrics
		for _, m := range metrics {
			snapshot := m // Create a copy so the results don't get mutated in the next round.
			snapshot.EncryptionRatio = encRatio
			snapshot.Round = en + 1
			allMetrics = append(allMetrics, snapshot)
		}
	}

	//Write metrics to file
	//for _, rec := range allMetrics {
	//	err := writer.Write([]string{
	//		rec.HouseholdCSVFileName,
	//		rec.MeterReadingDate,
	//		rec.MeterReadingTime,
	//		fmt.Sprintf("%.4f", rec.MeterReadingValue),
	//		fmt.Sprintf("%.6f", rec.IndividualReadingEntropy),
	//		rec.Strategy,
	//		fmt.Sprintf("%.4f", rec.EncryptionRatio),
	//		fmt.Sprintf("%.6f", rec.IndividualReadingRemainingEntropy),
	//	})
	//	check(err)
	//}

	writer.Flush()
}

/*
This function selects one section within each household for encryption based on which section has the most amount of entropy.
This function additionally computes for single records/meter readings within a party/household.
*/
func markEncryptedSectionsByHousehold(en int, encRatio float64, prevEntropyReduction float64, P []*party, metrics *[]RecordMetrics) (plainSum []float64, entropyReduction float64) {
	currentEntropyReduction := 0.0
	plainSum = make([]float64, len(P))
	var targetArr []float64

	// Encrypt one section per household
	for _, po := range P {
		index := 0
		max := -1.0
		targetArr = po.entropy

		for j := 0; j < sectionNum; j++ {
			if po.flag[j] != 1 && targetArr[j] > max {
				max = targetArr[j]
				index = j
			}
		}
		po.flag[index] = 1 // po.flag has sectionNumber elements
		currentEntropyReduction += po.entropy[index]
	}

	// Compute the difference (entropy reduction for this round of marking)
	roundEntropyReduction := currentEntropyReduction - prevEntropyReduction

	// Prepare encrypted and plaintext inputs and process records
	for pi, po := range P {
		po.input = make([][]float64, 0)
		po.plainInput = make([]float64, 0)
		k := 0

		for j := 0; j < globalPartyRows; j++ {
			sectionIndex := j / sectionSize
			recordIndex := pi*globalPartyRows + j

			// Calculate section entropy sum once for encrypted sections
			var sectionEntropySum float64 // Local per-section total entropy
			if po.flag[sectionIndex] == 1 {
				start := sectionIndex * sectionSize // Index (relative to that household) of the first record in the section.
				end := start + sectionSize          // Index of the last record (non-inclusive) in the section
				globalStart := pi*globalPartyRows + start
				globalEnd := pi*globalPartyRows + end
				for s := globalStart; s < globalEnd && s < len(*metrics); s++ {
					sectionEntropySum += (*metrics)[s].IndividualReadingEntropy
				}
			}

			// Update per-record encryption metrics
			if recordIndex < len(*metrics) {
				if po.flag[sectionIndex] == 1 {
					// This record is in an encrypted section
					if currentEntropyReduction <= 0 {
						continue
					} else {
						ratio := (*metrics)[recordIndex].IndividualReadingEntropy / sectionEntropySum                                                                  // How much of the entropy reduction does each record own
						(*metrics)[recordIndex].IndividualReadingRemainingEntropy = (*metrics)[recordIndex].IndividualReadingEntropy - (roundEntropyReduction * ratio) // Reduce by only that amount
						fmt.Printf("round=%d, currententropyreduction=%.4f, record=%d, original=%.6f, encryptionratio=%.4f, entropyreductionratio=%.4f, reduced=%.6f\n", en, currentEntropyReduction, recordIndex, (*metrics)[recordIndex].IndividualReadingEntropy, encRatio, ratio, (*metrics)[recordIndex].IndividualReadingRemainingEntropy)
					}
				} else {
					// Not encrypted yet
					(*metrics)[recordIndex].IndividualReadingRemainingEntropy = (*metrics)[recordIndex].IndividualReadingEntropy
				}
				(*metrics)[recordIndex].Strategy = "Household"
			}

			// Setup encrypted vs plain input
			if j%sectionSize == 0 && po.flag[sectionIndex] == 1 {
				po.input = append(po.input, make([]float64, sectionSize))
				k++
			}
			if po.flag[sectionIndex] == 1 {
				if k > 0 {
					po.input[k-1][j%sectionSize] = po.rawInput[j]
				}
			} else {
				plainSum[pi] += po.rawInput[j]
				po.plainInput = append(po.plainInput, po.rawInput[j])
			}
		}
	}

	return plainSum, currentEntropyReduction
}

// File Reading
func readCSV(path string) []string {
	data, err := os.ReadFile(path)
	check(err)
	dArray := strings.Split(string(data), "\n")
	return dArray[:len(dArray)-1]
}

// Trim CSV
func resizeCSV(filename string, metrics *[]RecordMetrics) []RecordMetrics {
	csvLines := readCSV(filename)

	for lineIndex, line := range csvLines {
		slices := strings.Split(line, ",")
		if len(slices) < 2 {
			continue
		}
		householdFilename := filepath.Base(strings.TrimSuffix(filename, filepath.Ext(filename)))
		utilityDate := strings.TrimSpace(slices[0])
		utilityTime := strings.TrimSpace(slices[1])
		usageStr := strings.TrimSpace(slices[len(slices)-1])

		usage, err := strconv.ParseFloat(usageStr, 64)
		check(err)

		if lineIndex < MAX_PARTY_ROWS {
			entry := RecordMetrics{
				HouseholdCSVFileName: householdFilename,
				MeterReadingDate:     utilityDate,
				MeterReadingTime:     utilityTime,
				MeterReadingValue:    usage,
			}
			*metrics = append(*metrics, entry)
		}
	}

	return *metrics
}

func getRandom(numberRange int) (randNumber int) {
	randNumber = rand.Intn(numberRange) //[0, numberRange-1]
	return
}

// generate parties
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

// generate inputs of parties
func genInputs(P []*party, metrics *[]RecordMetrics) (expSummation, expAverage, expDeviation []float64, min, max, entropySum float64) {
	globalPartyRows = -1
	sectionNum = 0
	min = math.MaxFloat64
	max = float64(-1)
	frequencyMap := map[float64]int{}
	entropyMap := map[float64]float64{}

	entropySum = 0.0

	for pi, po := range P {
		// Setup (rawInput, flags etc.)
		partyRows := resizeCSV(po.filename, metrics)
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
			sectionNum = lenPartyRows / sectionSize // sectionNum = 10, lenPartyRows = 10240, sectionSize = 1024
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
		po.flag = make([]int, sectionNum)        // Marked sections for encryption
		po.entropy = make([]float64, sectionNum) // Entropy values for each section of that household
		po.group = make([]int, sectionSize)

		// Fill in rawInput & frequencyMap
		for i := 0; i < globalPartyRows; i++ {
			usage := math.Round(partyRows[i].MeterReadingValue*1000) / 1000
			po.rawInput[i] = usage
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
			if i < len(*metrics) {
				(*metrics)[i].IndividualReadingEntropy = singleRecordEntropy
			}
		}
		po.flag = make([]int, sectionNum) // Reset flag for every party
	}

	// Min, Max based on currentTarget which is always (1) Entropy-based
	for _, po := range P {
		var targetArr []float64
		targetArr = po.entropy // Holds reference to entropy values for each section of that household

		for sIndex := range targetArr {
			if targetArr[sIndex] > max {
				max = targetArr[sIndex] // Max entropy seen so far across all parties
			}
			if targetArr[sIndex] < min {
				min = targetArr[sIndex] // Min entropy seen so far across all parties
			}
		}
	}
	return
}
