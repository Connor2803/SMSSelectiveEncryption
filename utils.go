package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/drlwe"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

// Common types and structures
type party struct {
	filename    string
	sk          *rlwe.SecretKey
	rlkEphemSk  *rlwe.SecretKey
	ckgShare    *drlwe.CKGShare
	rkgShareOne *drlwe.RKGShare
	rkgShareTwo *drlwe.RKGShare
	rtgShare    *drlwe.RTGShare
	pcksShare   *drlwe.PCKSShare

	rawInput       []float64
	input          [][]float64
	plainInput     []float64
	encryptedInput []float64
	flag           []int
	group          []int
	entropy        []float64
	transition     []float64

	greedyInputs [][]float64
	greedyFlags  [][]int
}

type task struct {
	wg          *sync.WaitGroup
	op1         *rlwe.Ciphertext
	op2         *rlwe.Ciphertext
	res         *rlwe.Ciphertext
	elapsedtask time.Duration
}

type OptimizationResult struct {
	TestName          string
	SectionSize       int
	ATDSize           int
	MatchingThreshold int
	EncryptionRatio   int
	EntropyWeight     float64
	UniquenessWeight  float64
	ASR               float64
	StandardError     float64
	ProcessingTime    float64
	MemoryUsage       int64
	Dataset           string
	Strategy          string
}

// Constants
const MAX_PARTY_ROWS = 10240
const STRATEGY_GLOBAL = 1
const STRATEGY_HOUSEHOLD = 2
const STRATEGY_RANDOM = 3
const DATASET_WATER = 1
const DATASET_ELECTRICITY = 2
const WATER_TRANSITION_EQUALITY_THRESHOLD = 100
const ELECTRICITY_TRANSITION_EQUALITY_THRESHOLD = 2

// Global variables
var sectionSize = 1024
var atdSize = 24
var min_percent_matched = 100
var GLOBAL_ATTACK_LOOP = 2000
var LOCAL_ATTACK_LOOP = 2000
var max_attackLoop = 10
var maxHouseholdsNumber = 80
var NGoRoutine int = 1
var encryptedSectionNum int
var globalPartyRows = -1
var performanceLoops = 1
var currentStrategy int = 1
var currentDataset int = 1
var uniqueATD int = 0
var currentTarget = 1
var encryptionRatio = 20
var transitionEqualityThreshold int
var sectionNum int
var usedRandomStartPartyPairs = map[int][]int{}
var usedHouses = map[int]int{}
var asrList []float64
var edgeNumberArray = []int{}
var house_sample = []float64{}
var elapsedTime time.Duration

// Utility functions
func almostEqual(a, b float64) bool {
	return math.Abs(a-b) <= float64(transitionEqualityThreshold)
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func getRandom(numberRange int) int {
	return rand.Intn(numberRange)
}

func calculateStandardDeviation(numbers []float64) (float64, float64) {
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

// Dataset loading functions
func getFileList(datasetType string) []string {
	wd, err := os.Getwd()
	if err != nil {
		fmt.Println("Error getting current working directory:", err)
		return nil
	}

	var pathFormat string
	var path string
	if strings.Contains(wd, "examples") {
		pathFormat = filepath.Join("..", "..", "..", "examples", "datasets", "%s", "households_%d")
	} else {
		pathFormat = filepath.Join("examples", "datasets", "%s", "households_%d")
	}

	path = fmt.Sprintf(pathFormat, datasetType, MAX_PARTY_ROWS)
	folder := filepath.Join(wd, path)

	fileList := []string{}
	err = filepath.Walk(folder, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			fileList = append(fileList, path)
		}
		return nil
	})
	if err != nil {
		fmt.Printf("Error reading %s dataset: %v\n", datasetType, err)
		return nil
	}

	if len(fileList) < maxHouseholdsNumber {
		return fileList
	}
	return fileList[:maxHouseholdsNumber]
}

func ReadCSV(path string) []string {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	dArray := strings.Split(string(data), "\n")
	return dArray[:len(dArray)-1]
}

func resizeCSV(filename string) []float64 {
	csv := ReadCSV(filename)

	elements := []float64{}
	for _, v := range csv {
		slices := strings.Split(v, ",")
		tmpStr := strings.TrimSpace(slices[len(slices)-1])
		fNum, err := strconv.ParseFloat(tmpStr, 64)
		if err != nil {
			panic(err)
		}
		elements = append(elements, fNum)
	}
	return elements
}

// Party generation and input processing
func genparties(params ckks.Parameters, fileList []string) []*party {
	P := make([]*party, len(fileList))

	for i, _ := range P {
		po := &party{}
		po.sk = ckks.NewKeyGenerator(params).GenSecretKey()
		po.filename = fileList[i]
		P[i] = po
	}
	return P
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
		partyRows := resizeCSV(po.filename)
		lenPartyRows := len(partyRows)
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
			err := errors.New("Not all files have the same rows")
			check(err)
		}

		po.rawInput = make([]float64, globalPartyRows)
		po.encryptedInput = make([]float64, globalPartyRows)
		po.flag = make([]int, sectionNum)
		po.entropy = make([]float64, sectionNum)
		po.transition = make([]float64, sectionNum)
		po.group = make([]int, sectionSize)

		po.greedyInputs = make([][]float64, sectionNum)
		for i := range po.greedyInputs {
			po.greedyInputs[i] = make([]float64, sectionSize)
		}
		po.greedyFlags = make([][]int, sectionNum)
		for i := range po.greedyFlags {
			po.greedyFlags[i] = make([]int, sectionSize)
		}

		delta := 0.0
		for i := range po.rawInput {
			var value float64
			if currentDataset == DATASET_ELECTRICITY {
				realValue := math.Round(partyRows[i]*1000) / 1000
				decoratedValue := math.Round(partyRows[i]*10) / 10
				delta += decoratedValue - realValue
				value = decoratedValue
				if i == len(po.rawInput)-1 {
					value -= delta
				}
			} else {
				value = math.Round(partyRows[i]*1000) / 1000
			}

			po.rawInput[i] = value
			if currentDataset == DATASET_ELECTRICITY {
				po.greedyInputs[i/sectionSize][i%sectionSize] = value
			}

			val, exists := frequencyMap[po.rawInput[i]]
			if exists {
				val++
			} else {
				val = 1
			}
			frequencyMap[po.rawInput[i]] = val

			expSummation[pi] += po.rawInput[i]
		}

		expAverage[pi] = expSummation[pi] / float64(globalPartyRows)
		for i := range po.rawInput {
			temp := po.rawInput[i] - expAverage[pi]
			expDeviation[pi] += temp * temp
		}
		expDeviation[pi] /= float64(globalPartyRows)
	}

	totalRecords := maxHouseholdsNumber * MAX_PARTY_ROWS
	for k, _ := range frequencyMap {
		possibility := float64(frequencyMap[k]) / float64(totalRecords)
		entropyMap[k] = -possibility * math.Log2(possibility)
	}

	for _, po := range P {
		for i := range po.rawInput {
			singleRecordEntropy := entropyMap[po.rawInput[i]] / float64(frequencyMap[po.rawInput[i]])
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

// Common test execution function
func runOptimizationTest(testType string, params ckks.Parameters, dataset, strategy string, ratio int) OptimizationResult {
	// Set dataset and strategy
	if dataset == "water" {
		currentDataset = DATASET_WATER
		currentTarget = 1 // entropy
		transitionEqualityThreshold = WATER_TRANSITION_EQUALITY_THRESHOLD
	} else {
		currentDataset = DATASET_ELECTRICITY
		currentTarget = 2 // transition/uniqueness
		transitionEqualityThreshold = ELECTRICITY_TRANSITION_EQUALITY_THRESHOLD
	}

	encryptionRatio = ratio
	currentStrategy = STRATEGY_GLOBAL

	// Load dataset
	fileList := getFileList(dataset)
	if fileList == nil {
		return OptimizationResult{TestName: testType, ASR: -1, Dataset: dataset, Strategy: strategy}
	}

	// Reset sample data
	house_sample = []float64{}
	elapsedTime = 0

	// Run test with limited loops for optimization
	tLoop := 3 // Reduced for faster optimization testing
	startTime := time.Now()

	for t := 0; t < tLoop; t++ {
		print(".")
		if strategy == "entropy" {
			processEntropy(fileList, params)
		} else {
			processGreedyWithASR(fileList, params)
		}

		// Early exit if we have stable results
		if len(house_sample) > 1 {
			std, _ := calculateStandardDeviation(house_sample)
			standard_error := std / math.Sqrt(float64(len(house_sample)))
			if standard_error <= 0.02 && t >= 1 { // More lenient for optimization
				break
			}
		}
	}

	processingTime := time.Since(startTime).Seconds()

	// Calculate results
	result := OptimizationResult{
		TestName:        testType,
		EncryptionRatio: ratio,
		ProcessingTime:  processingTime,
		Dataset:         dataset,
		Strategy:        strategy,
	}

	if len(house_sample) > 0 {
		std, mean := calculateStandardDeviation(house_sample)
		standard_error := std / math.Sqrt(float64(len(house_sample)))
		result.ASR = mean
		result.StandardError = standard_error
	} else {
		result.ASR = -1
		result.StandardError = -1
	}

	return result
}

func getStrategyName(strategy int) string {
	switch strategy {
	case STRATEGY_GLOBAL:
		return "global"
	case STRATEGY_HOUSEHOLD:
		return "household"
	case STRATEGY_RANDOM:
		return "random"
	default:
		return "unknown"
	}
}

// Attack and encryption functions (implementations from original file)

func processGreedyWithASR(fileList []string, params ckks.Parameters) {
	P := genparties(params, fileList)
	genInputs(P)
	intializeEdgeRelated(P)
	edgeSize := len(P) * (len(P) - 1) * sectionNum * sectionNum / 2
	edges := make([]float64, edgeSize)

	markedFirstHousehold := -1
	markedFirstSection := -1
	markedSecondHousehold := -1
	markedSecondSection := -1
	markedNumbers := 0
	previousMarkedNumbers := 0

	thresholdNumber := len(P) * globalPartyRows * encryptionRatio / 100

	startTime := time.Now()
	for markedNumbers < thresholdNumber {
		maxUniquenessScore := -1.0
		for edge_index := range edges {
			p_index_first, s_index_first, p_index_second, s_index_second := getDetailedBlocksForEdge(edge_index, len(P), sectionNum)
			edges[edge_index] = calculateUniquenessBetweenBlocks(P, p_index_first, s_index_first, p_index_second, s_index_second)

			if edges[edge_index] > maxUniquenessScore {
				maxUniquenessScore = edges[edge_index]
				markedFirstHousehold = p_index_first
				markedFirstSection = s_index_first
				markedSecondHousehold = p_index_second
				markedSecondSection = s_index_second
			}
		}
		previousMarkedNumbers = markedNumbers
		markedNumbers = greedyMarkBlocks(markedNumbers, thresholdNumber, P, markedFirstHousehold, markedFirstSection, markedSecondHousehold, markedSecondSection)
		if markedNumbers == previousMarkedNumbers {
			break
		}
	}

	if markedNumbers < thresholdNumber {
		for markedNumbers < thresholdNumber {
			randomPartyIndex := getRandom(len(P))
			randomBlockIndex := getRandom(sectionNum)
			randomBlockFlags := P[randomPartyIndex].greedyFlags[randomBlockIndex]
			for i := 0; i < sectionSize; i++ {
				if randomBlockFlags[i] == 0 {
					randomBlockFlags[i] = 1
					markedNumbers++
					if markedNumbers == thresholdNumber {
						break
					}
				}
			}
		}
	}
	greedyEncryptBlocks(P)
	elapsedTime += time.Since(startTime)

	memberIdentificationAttack(P)
	usedRandomStartPartyPairs = map[int][]int{}
}

func processEntropy(fileList []string, params ckks.Parameters) {
	P := genparties(params, fileList)
	_, _, _, _, _, entropySum, transitionSum := genInputs(P)
	encryptedSectionNum = sectionNum * encryptionRatio / 100

	var entropyReduction float64
	var transitionReduction float64

	for en := 0; en <= encryptedSectionNum; en++ {
		_, entropyReduction, transitionReduction = markEncryptedSectionsByGlobal(en, P, entropySum, transitionSum)
		entropySum -= entropyReduction
		transitionSum -= transitionReduction

		if en >= 0 {
			memberIdentificationAttack(P)
		}
		usedRandomStartPartyPairs = map[int][]int{}
	}
}

func memberIdentificationAttack(P []*party) {
	var attackSuccessNum int
	var attackCount int
	var sample = []float64{}
	var std float64
	var standard_error float64

	for attackCount = 0; attackCount < max_attackLoop; attackCount++ {
		var successNum = attackParties(P)
		attackSuccessNum += successNum
		sample = append(sample, float64(successNum))
		std, _ = calculateStandardDeviation(sample)
		standard_error = std / math.Sqrt(float64(len(sample)))
		if standard_error <= 0.01 && attackCount >= 5 {
			attackCount++
			break
		}
	}
	house_sample = append(house_sample, float64(attackSuccessNum)/float64(attackCount))
}

func attackParties(P []*party) (attackSuccessNum int) {
	attackSuccessNum = 0
	var valid = false
	var randomParty int
	var randomStart int

	if uniqueATD == 0 {
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
	var matched_households = identifyParty(P, attacker_data_block, randomParty, randomStart)
	if len(matched_households) == 1 && matched_households[0] == randomParty {
		attackSuccessNum++
	}
	return
}

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

func uniqueDataBlock(P []*party, arr []float64, party int, index int, input_type string) bool {
	var unique bool = true

	for pi, po := range P {
		if pi == party {
			continue
		}
		var household_data []float64
		if input_type == "rawInput" {
			household_data = po.rawInput
		} else {
			household_data = po.encryptedInput
		}
		for i := 0; i < len(household_data)-atdSize+1; i++ {
			var target = household_data[i : i+atdSize]
			if reflect.DeepEqual(target, arr) {
				unique = false
				usedRandomStartPartyPairs[pi] = append(usedRandomStartPartyPairs[pi], i)
				break
			}
		}
		if !unique {
			break
		}
	}
	return unique
}

func identifyParty(P []*party, arr []float64, party int, index int) []int {
	var matched_households = []int{}
	var dataset = P[party].encryptedInput[index : index+atdSize]
	var min_length int = int(math.Ceil(float64(len(arr)) * float64(min_percent_matched) / 100))

	if min_length == len(arr) {
		if uniqueATD == 0 {
			if reflect.DeepEqual(dataset, arr) && uniqueDataBlock(P, dataset, party, index, "encryptedInput") {
				matched_households = append(matched_households, party)
			}
		} else {
			if reflect.DeepEqual(dataset, arr) {
				matched_households = append(matched_households, party)
			}
		}
	} else {
		var match int = 0
		var mismatch int = 0
		for i := 0; i < len(arr); i++ {
			if reflect.DeepEqual(arr[i], dataset[i]) {
				match += 1
			} else {
				mismatch += 1
			}
			if mismatch > (len(arr) - min_length) {
				break
			}
		}

		if float64(match)/float64(len(arr)) >= float64(min_percent_matched)/100.0 {
			var pos_matches = [][]float64{}
			if atdSize <= sectionSize {
				var pos_match1 = P[party].encryptedInput[index : index+min_length]
				var post_match2 = P[party].encryptedInput[index+atdSize-min_length : index+atdSize]
				pos_matches = append(pos_matches, pos_match1, post_match2)
			} else {
				for i := 0; i <= len(arr)-min_length; i++ {
					var pos_match = P[party].encryptedInput[index+i : index+min_length+i]
					pos_matches = append(pos_matches, pos_match)
				}
			}
			if uniqueDataBlocks(P, pos_matches, party, index, min_length) {
				matched_households = append(matched_households, party)
			}
		}
	}
	return matched_households
}

func uniqueDataBlocks(P []*party, pos_matches [][]float64, party int, index int, min_length int) bool {
	var unique bool = true

	for pn, po := range P {
		if pn == party {
			continue
		}
		var household_data []float64 = po.encryptedInput
		for i := 0; i < len(household_data)-min_length+1; i++ {
			var target = household_data[i : i+min_length]
			for _, pos_match := range pos_matches {
				if reflect.DeepEqual(target, pos_match) {
					unique = false
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

func calculateUniquenessBetweenBlocks(P []*party, p_index_first, s_index_first, p_index_second, s_index_second int) float64 {
	firstBlock := P[p_index_first].greedyInputs[s_index_first]
	secondBlock := P[p_index_second].greedyInputs[s_index_second]
	firstBlockFlags := P[p_index_first].greedyFlags[s_index_first]
	secondBlockFlags := P[p_index_second].greedyFlags[s_index_second]

	uniquenessScore := 0.0
	for i := 0; i < sectionSize; i++ {
		if firstBlockFlags[i] == 1 || secondBlockFlags[i] == 1 {
			uniquenessScore += 0.9
		} else if firstBlock[i] != secondBlock[i] {
			uniquenessScore += 1
		} else {
			uniquenessScore += 0
		}
	}
	return uniquenessScore
}

func greedyMarkBlocks(markedNumbers, thresholdNumber int, P []*party, markedFirstHousehold, markedFirstSection, markedSecondHousehold, markedSecondSection int) int {
	firstBlock := P[markedFirstHousehold].greedyInputs[markedFirstSection]
	secondBlock := P[markedSecondHousehold].greedyInputs[markedSecondSection]
	firstBlockFlags := P[markedFirstHousehold].greedyFlags[markedFirstSection]
	secondBlockFlags := P[markedSecondHousehold].greedyFlags[markedSecondSection]

	for i := 0; i < sectionSize; i++ {
		if firstBlockFlags[i] == 0 && secondBlockFlags[i] == 0 {
			if firstBlock[i] != secondBlock[i] {
				firstBlockFlags[i] = 1
				markedNumbers++
				if markedNumbers == thresholdNumber {
					break
				}
				secondBlockFlags[i] = 1
				markedNumbers++
				if markedNumbers == thresholdNumber {
					break
				}
			}
		} else if firstBlockFlags[i] == 0 && secondBlockFlags[i] == 1 {
			if firstBlock[i] != secondBlock[i] {
				firstBlockFlags[i] = 1
				markedNumbers++
				if markedNumbers == thresholdNumber {
					break
				}
			}
		} else if firstBlockFlags[i] == 1 && secondBlockFlags[i] == 0 {
			if firstBlock[i] != secondBlock[i] {
				secondBlockFlags[i] = 1
				markedNumbers++
				if markedNumbers == thresholdNumber {
					break
				}
			}
		}
	}
	return markedNumbers
}

func greedyEncryptBlocks(P []*party) {
	for _, po := range P {
		for j := 0; j < globalPartyRows; j++ {
			if po.greedyFlags[j/sectionSize][j%sectionSize] == 1 {
				po.encryptedInput[j] = -0.1
			} else {
				po.encryptedInput[j] = po.rawInput[j]
			}
		}
	}
}

func intializeEdgeRelated(P []*party) {
	edgeNumberArray = make([]int, len(P))
	for i := range edgeNumberArray {
		num := 0
		for j := 1; j < len(P)-i; j++ {
			num += sectionNum * sectionNum
		}
		edgeNumberArray[i] = num
	}
}

func getDetailedBlocksForEdge(edge_index, householdNumber, sectionNumber int) (int, int, int, int) {
	p_index_first, s_index_first, p_index_second, s_index_second := -1, -1, -1, -1
	sectionNumberSqured := sectionNumber * sectionNumber

	sum := 0
	for i := 0; i < len(edgeNumberArray); i++ {
		previousSum := sum
		sum += edgeNumberArray[i]
		if sum > edge_index {
			p_index_first = i
			edge_index -= previousSum
			break
		}
	}

	sum = 0
	for i := p_index_first + 1; i < householdNumber; i++ {
		previousSum := sum
		sum += sectionNumberSqured
		if sum > edge_index {
			p_index_second = i
			edge_index -= previousSum
			break
		}
	}

	sum = 0
	for i := 0; i < sectionNumber; i++ {
		previousSum := sum
		sum += sectionNumber
		if sum > edge_index {
			s_index_first = i
			edge_index -= previousSum
			break
		}
	}

	s_index_second = edge_index
	return p_index_first, s_index_first, p_index_second, s_index_second
}

func markEncryptedSectionsByGlobal(en int, P []*party, entropySum, transitionSum float64) (plainSum []float64, entropyReduction, transitionReduction float64) {
	entropyReduction = 0.0
	transitionReduction = 0
	plainSum = make([]float64, len(P))
	var targetArr []float64

	if en != 0 {
		for k := 0; k < len(P); k++ {
			max := -1.0
			sIndex := -1
			pIndex := -1
			for pi, po := range P {
				if currentTarget == 1 {
					targetArr = po.entropy
				} else {
					targetArr = po.transition
				}
				for si := 0; si < sectionNum; si++ {
					if po.flag[si] != 1 && targetArr[si] > max {
						max = targetArr[si]
						sIndex = si
						pIndex = pi
					}
				}
			}
			P[pIndex].flag[sIndex] = 1
			entropyReduction += P[pIndex].entropy[sIndex]
			transitionReduction += P[pIndex].transition[sIndex]
		}
	}

	for pi, po := range P {
		po.input = make([][]float64, 0)
		po.plainInput = make([]float64, 0)
		po.encryptedInput = make([]float64, 0)
		k := 0
		for j := 0; j < globalPartyRows; j++ {
			if j%sectionSize == 0 && po.flag[j/sectionSize] == 1 {
				po.input = append(po.input, make([]float64, sectionSize))
				k++
			}

			if po.flag[j/sectionSize] == 1 {
				po.input[k-1][j%sectionSize] = po.rawInput[j]
				po.encryptedInput = append(po.encryptedInput, -0.1)
			} else {
				plainSum[pi] += po.rawInput[j]
				po.plainInput = append(po.plainInput, po.rawInput[j])
				po.encryptedInput = append(po.encryptedInput, po.rawInput[j])
			}
		}
	}
	return
}
