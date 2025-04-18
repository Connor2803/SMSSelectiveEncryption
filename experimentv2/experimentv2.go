/*
running command: // dataset
go run .\experimentv2\experimentv2.go 1
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
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/drlwe"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) <= float64(transitionEqualityThreshold)
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

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

type party struct {
	filename    string
	sk          *rlwe.SecretKey
	rlkEphemSk  *rlwe.SecretKey
	ckgShare    *drlwe.CKGShare
	rkgShareOne *drlwe.RKGShare
	rkgShareTwo *drlwe.RKGShare
	rtgShare    *drlwe.RTGShare
	pcksShare   *drlwe.PCKSShare

	rawInput        []float64 // All data
	partyDates      []string
	partyTimes      []string
	input           [][]float64 // Encryption data
	plainInput      []float64   // Plaintext data
	encryptedInput  []float64   // Encrypted data of rawInput (encrypted value are -0.1)
	flag            []int       // Slice to track which sections have been encrypted
	group           []int
	entropy         []float64 // Entropy per section
	recordEntropies []float64 // Slice to hold individual-level entropies in a block
	transition      []float64 // Transition per section
} // Struct to represent an individual household

type task struct {
	wg          *sync.WaitGroup
	op1         *rlwe.Ciphertext
	op2         *rlwe.Ciphertext
	res         *rlwe.Ciphertext
	elapsedtask time.Duration
}
type RecordMetrics struct {
	HouseholdCSVFileName              string
	MeterReadingDate                  string
	MeterReadingTime                  string
	MeterReadingValue                 float64
	SectionIndex                      int
	IndividualReadingEntropy          float64 // Per-reading, pre-encryption entropy value
	EncryptionRatio                   float64
	IndividualReadingRemainingEntropy float64 // Per-reading, post-encryption entropy value
	Round                             int
} // Struct to represent a meter reading in an individual household

const MAX_PARTY_ROWS = 10240 //241920
const sectionSize = 1024     // element number within a section
const DATASET_WATER = 1
const DATASET_ELECTRICITY = 2
const WATER_TRANSITION_EQUALITY_THRESHOLD = 100
const ELECTRICITY_TRANSITION_EQUALITY_THRESHOLD = 2

var maxHouseholdsNumber int = 1

var NGoRoutine int = 1 // Default number of Go routines
var globalPartyRows = -1

var currentDataset int = 1 //water(1),electricity(2)
var currentTarget int = 1  //entropy(1),transition(2)

var transitionEqualityThreshold int
var sectionNum int

var elapsedTime time.Duration

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

	csvFile, err := os.Create("./experimentv2/metricsv2.csv")
	check(err)
	defer csvFile.Close()
	writer := csv.NewWriter(csvFile)
	defer writer.Flush()

	originalOutput := os.Stdout
	defer func() { os.Stdout = originalOutput }()

	rand.Seed(time.Now().UnixNano())
	//start := time.Now()

	fileList := []string{}
	paramsDef := utils.PN10QP27CI
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)

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
		transitionEqualityThreshold = WATER_TRANSITION_EQUALITY_THRESHOLD
	} else { //electricity
		path = fmt.Sprintf(pathFormat, "electricity", MAX_PARTY_ROWS)
		transitionEqualityThreshold = ELECTRICITY_TRANSITION_EQUALITY_THRESHOLD
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

	P := genParties(params, fileList)
	genInputs(P)

	startTime := time.Now()
	for ratioStep := 1; ratioStep <= 10; ratioStep++ {
		ratio := float64(ratioStep) / 10.0
		fixedSectionIndex := 3 // Or whichever fixed section you want
		markSectionWithRatio(P, fixedSectionIndex)

		for _, po := range P {
			roundMetrics := genMetricsFromParty(po, ratio)
			allMetrics = append(allMetrics, roundMetrics...)
		}
	}
	// Write metrics to file
	err := writer.Write([]string{
		"filename", "date", "time", "section", "utility_used",
		"individual_reading_entropy", "encryption_ratio", "individual_reading_remaining_entropy",
	})
	check(err)
	for _, rec := range allMetrics {
		err = writer.Write([]string{
			rec.HouseholdCSVFileName,
			rec.MeterReadingDate,
			rec.MeterReadingTime,
			fmt.Sprintf("%d", rec.SectionIndex),
			fmt.Sprintf("%.4f", rec.MeterReadingValue),
			fmt.Sprintf("%.6f", rec.IndividualReadingEntropy),
			fmt.Sprintf("%.4f", rec.EncryptionRatio),
			fmt.Sprintf("%.6f", rec.IndividualReadingRemainingEntropy),
		})
		check(err)
	}
	writer.Flush()
	elapsedTime += time.Since(startTime)
}

func markSectionWithRatio(P []*party, sectionIndex int) {
	for _, po := range P {
		po.flag = make([]int, sectionNum)
		po.input = make([][]float64, 0)
		po.plainInput = make([]float64, 0)
		po.encryptedInput = make([]float64, 0)

		po.flag[sectionIndex] = 1 // Only mark the selected section

		start := sectionIndex * sectionSize
		end := start + sectionSize
		k := 0

		for j := 0; j < globalPartyRows; j++ {
			if j >= start && j < end {
				// Encrypted section
				if j%sectionSize == 0 {
					po.input = append(po.input, make([]float64, sectionSize))
					k++
				}
				po.input[k-1][j%sectionSize] = po.rawInput[j]
				po.encryptedInput = append(po.encryptedInput, -0.1)
			} else {
				// Plaintext section
				po.plainInput = append(po.plainInput, po.rawInput[j])
				po.encryptedInput = append(po.encryptedInput, po.rawInput[j])
			}
		}
	}
}

func genMetricsFromParty(po *party, ratio float64) []RecordMetrics {
	metrics := []RecordMetrics{}

	for i := 0; i < globalPartyRows; i++ {
		readingSectionIndex := i / sectionSize
		metric := RecordMetrics{
			HouseholdCSVFileName:     filepath.Base(strings.TrimSuffix(po.filename, filepath.Ext(po.filename))),
			MeterReadingValue:        po.rawInput[i],
			MeterReadingDate:         po.partyDates[i],
			MeterReadingTime:         po.partyTimes[i],
			IndividualReadingEntropy: po.recordEntropies[i],
			EncryptionRatio:          ratio,
			SectionIndex:             readingSectionIndex,
		}
		if po.flag[readingSectionIndex] == 1 {
			// Estimate reduction proportionally to the encryption ratio
			reduction := ratio * po.recordEntropies[i]
			metric.IndividualReadingRemainingEntropy = po.recordEntropies[i] - reduction
			//fmt.Printf("Record %d: ratio=%.2f original=%.6f → reduced=%.6f\n",
			//	i, ratio, po.recordEntropies[i], metric.IndividualReadingRemainingEntropy)
		} else {
			metric.IndividualReadingRemainingEntropy = po.recordEntropies[i]
		}
		metrics = append(metrics, metric)
	}

	return metrics
}

func genParties(params ckks.Parameters, fileList []string) []*party {
	P := make([]*party, len(fileList))

	for i, _ := range P {
		po := &party{}
		po.sk = ckks.NewKeyGenerator(params).GenSecretKey()
		po.filename = fileList[i]
		P[i] = po
	}

	return P
}

func readCSV(path string) []string {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	dArray := strings.Split(string(data), "\n")
	return dArray[:len(dArray)-1]
}

func resizeCSV(filename string) (values []float64, dates []string, times []string) {
	readingCSV := readCSV(filename)

	for _, v := range readingCSV {
		slices := strings.Split(v, ",")
		if len(slices) < 2 {
			continue
		}
		readingDate := strings.TrimSpace(slices[0])
		readingTime := strings.TrimSpace(slices[1])
		valStr := strings.TrimSpace(slices[len(slices)-1])
		fVal, err := strconv.ParseFloat(valStr, 64)
		check(err)
		values = append(values, fVal)
		dates = append(dates, readingDate)
		times = append(times, readingTime)

	}
	return
}

func genInputs(P []*party) (expSummation, expAverage, expDeviation []float64, min, max, entropySum, transitionSum float64) {
	sectionNum = 0
	min = math.MaxFloat64
	max = float64(-1)
	frequencyMap := map[float64]int{}
	entropyMap := map[float64]float64{}
	entropySum = 0.0
	transitionSum = 0
	for pi, po := range P {
		values, dates, times := resizeCSV(po.filename)
		if currentDataset == DATASET_WATER { // Reverse chronological order
			for i, j := 0, len(values)-1; i < j; i, j = i+1, j-1 {
				values[i], values[j] = values[j], values[i]
				dates[i], dates[j] = dates[j], dates[i]
				times[i], times[j] = times[j], times[i]
			}
		}
		lenPartyRows := len(values)
		if lenPartyRows > MAX_PARTY_ROWS {
			lenPartyRows = MAX_PARTY_ROWS
		}

		if pi == 0 {
			sectionNum = lenPartyRows / sectionSize
			if lenPartyRows%sectionSize != 0 {
				sectionNum++
			}
			globalPartyRows = lenPartyRows
			expSummation = make([]float64, len(P))
			expAverage = make([]float64, len(P))
			expDeviation = make([]float64, len(P))
		} else if globalPartyRows != lenPartyRows {
			// Make sure pi.input[] has the same size
			err := errors.New("Not all files have the same rows")
			check(err)
		}

		po.rawInput = make([]float64, globalPartyRows)
		po.recordEntropies = make([]float64, globalPartyRows)
		po.partyDates = make([]string, globalPartyRows)
		po.partyTimes = make([]string, globalPartyRows)

		po.flag = make([]int, sectionNum)
		po.entropy = make([]float64, sectionNum)
		po.transition = make([]float64, sectionNum)
		po.group = make([]int, sectionSize)

		for i := range po.rawInput {
			po.rawInput[i] = math.Round(values[i]*1000) / 1000 // Hold 3 decimal places
			po.partyDates[i] = dates[i]
			po.partyTimes[i] = times[i]

			val, exists := frequencyMap[po.rawInput[i]]
			if exists {
				val++
			} else {
				val = 1
			}
			frequencyMap[po.rawInput[i]] = val

			expSummation[pi] += po.rawInput[i]
		} // each reading entry of individual household

		expAverage[pi] = expSummation[pi] / float64(globalPartyRows)
		for i := range po.rawInput {
			temp := po.rawInput[i] - expAverage[pi]
			expDeviation[pi] += temp * temp
		}
		expDeviation[pi] /= float64(globalPartyRows)
	} // each household

	totalRecords := maxHouseholdsNumber * MAX_PARTY_ROWS
	for k, _ := range frequencyMap {
		possibility := float64(frequencyMap[k]) / float64(totalRecords)
		entropyMap[k] = -possibility * math.Log2(possibility)
	}

	// max,min based on currentTarget
	for _, po := range P {
		for i := range po.rawInput {
			singleRecordEntropy := entropyMap[po.rawInput[i]] / float64(frequencyMap[po.rawInput[i]])
			po.recordEntropies[i] = singleRecordEntropy
			po.entropy[i/sectionSize] += singleRecordEntropy
			entropySum += singleRecordEntropy
			if i > 0 && !almostEqual(po.rawInput[i], po.rawInput[i-1]) {
				po.transition[i/sectionSize] += 1
				transitionSum++
			}
		}
	}

	for _, po := range P {
		var targetArr []float64
		if currentTarget == 1 {
			targetArr = po.entropy
		} else {
			targetArr = po.transition
		}
		for sIndex := range targetArr {
			if targetArr[sIndex] > max {
				max = targetArr[sIndex]
			}
			if targetArr[sIndex] < min {
				min = targetArr[sIndex]
			}
		}
	}

	return
}
