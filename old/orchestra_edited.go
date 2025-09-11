package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Flags: choose breaking approach, tolerance(s), encryption ratio(s), attack loops
var (
	flagApproach  = flag.String("approach", "adaptive", "sliding | adaptive")
	flagTol       = flag.Float64("tol", 0.30, "single tolerance in [0,1]")
	flagTols      = flag.String("tols", "", "comma-separated tolerances, e.g. 0.25,0.30,0.35")
	flagRatio     = flag.Int("ratio", 60, "single encryption ratio percent")
	flagRatios    = flag.String("ratios", "", "comma-separated ratios, e.g. 35,40,45")
	flagLoops     = flag.Int("loops", 200, "attack loops for ASR estimate")
	flagATD       = flag.Int("atd", 24, "single attacker leaked block size (samples)")
	flagATDs      = flag.String("atds", "", "comma-separated ATD sizes, e.g. 6,8,12,16")
	flagShowDebug = flag.Bool("debug", true, "print encryption coverage/debug info")
	flagCSVOut    = flag.String("csv", "results.csv", "output CSV file")
)

func main() {
	flag.Parse()

	// Resolve tolerance list
	tolVals := []float64{}
	if strings.TrimSpace(*flagTols) != "" {
		var err error
		tolVals, err = parseFloatList(*flagTols)
		check(err)
	} else {
		tolVals = []float64{*flagTol}
	}

	// Resolve ratio list
	ratioVals := []int{}
	if strings.TrimSpace(*flagRatios) != "" {
		var err error
		ratioVals, err = parseIntList(*flagRatios)
		check(err)
	} else {
		ratioVals = []int{*flagRatio}
	}

	// Resolve ATD list
	atdVals := []int{}
	if strings.TrimSpace(*flagATDs) != "" {
		var err error
		atdVals, err = parseIntList(*flagATDs)
		check(err)
	} else {
		atdVals = []int{*flagATD}
	}

	// Dataset and global thresholds
	currentDataset = DATASET_ELECTRICITY
	transitionEqualityThreshold = ELECTRICITY_TRANSITION_EQUALITY_THRESHOLD

	// 1) Load data using utils.getFileList + resizeCSV
	println("Testing series breaking and uniqueness selection for electricity data...")
	fileList := getFileList("electricity")
	if len(fileList) == 0 {
		panic("no dataset files found")
	}
	original := loadMatrix(fileList) // [][]float64

	// Prepare CSV output
	csvFile, err := os.Create(*flagCSVOut)
	check(err)
	defer csvFile.Close()

	// Write CSV header with ATD column
	_, err = csvFile.WriteString("approach,tolerance,ratio,atd,asr,time\n")
	check(err)

	// Iterate all combinations
	fmt.Printf("Combinations: %d tolerances × %d ratios × %d ATDs, loops=%d, approach=%s\n",
		len(tolVals), len(ratioVals), len(atdVals), *flagLoops, *flagApproach)

	for _, tol := range tolVals {
		for _, ratio := range ratioVals {
			for _, atd := range atdVals {
				asr, took := runOneCombo(original, fileList, *flagApproach, tol, ratio, *flagLoops, atd, *flagShowDebug)

				// Console output
				fmt.Printf("Result: approach=%s, T=%.2f, ratio=%d%%, atd=%d → ASR=%.4f (loops=%d, time=%s)\n",
					*flagApproach, tol, ratio, atd, asr, *flagLoops, took)

				// CSV output with ATD column
				csvLine := fmt.Sprintf("%s,%.2f,%d,%d,%.4f,%s\n",
					*flagApproach, tol, ratio, atd, asr, took)
				_, err = csvFile.WriteString(csvLine)
				check(err)
			}
		}
	}

	fmt.Printf("Results saved to %s\n", *flagCSVOut)
}

// Run a single (tol, ratio) combo end-to-end and return ASR
func runOneCombo(original [][]float64, fileList []string, approach string, tol float64, ratio int, loops int, atdSize int, showDebug bool) (float64, time.Duration) {
	start := time.Now()

	// 2) Series breaking from original each time (no carry-over)
	var broken [][]float64
	switch strings.ToLower(approach) {
	case "sliding":
		broken = applyWindowBasedBreaking(original, tol)
	case "adaptive":
		broken = applyAdaptivePatternBreaking(original, tol)
	case "none":
		broken = original
	default:
		panic(fmt.Sprintf("unknown approach: %s", approach))
	}

	// 3) Use your existing convertDataToParties function
	P := convertDataToParties(broken)

	// Set global variables
	currentDataset = DATASET_ELECTRICITY
	encryptionRatio = ratio
	transitionEqualityThreshold = ELECTRICITY_TRANSITION_EQUALITY_THRESHOLD

	// Use your existing functions to generate inputs and apply encryption
	generateInputsFromData(P, broken)
	intializeEdgeRelated(P)
	processGreedyEncryptionSimulation(P)

	if showDebug {
		total, enc := countEncrypted(P)
		wAll, wMasked, wClean := windowMaskStats(P, atdSize)
		fmt.Printf("Debug: T=%.2f, ratio=%d%%, atd=%d → encrypted %d/%d (%.1f%%), windows masked=%.2f%% clean=%.2f%%\n",
			tol, ratio, atdSize, enc, total, 100.0*float64(enc)/float64(total),
			100.0*float64(wMasked)/float64(wAll),
			100.0*float64(wClean)/float64(wAll))
	}

	// 4) Attack with specified ATD size
	asr := runAttackAverage(P, loops, atdSize)
	took := time.Since(start)
	return asr, took
}

// Prepare party inputs directly from a [][]float64 matrix (BROKEN data)
func generateInputsFromMatrix(P []*party, matrix [][]float64) {
	if len(P) != len(matrix) {
		panic("parties/data mismatch")
	}

	// set sectionNum and globalPartyRows from the first party
	rows := len(matrix[0])
	globalPartyRows = rows
	sectionNum = rows / sectionSize
	if rows%sectionSize != 0 {
		sectionNum++
	}

	for pi, po := range P {
		if len(matrix[pi]) != rows {
			panic("all parties must have same number of rows")
		}

		po.rawInput = make([]float64, globalPartyRows)
		po.encryptedInput = make([]float64, globalPartyRows)
		po.flag = make([]int, sectionNum)

		po.greedyInputs = make([][]float64, sectionNum)
		po.greedyFlags = make([][]int, sectionNum)
		for s := 0; s < sectionNum; s++ {
			po.greedyInputs[s] = make([]float64, sectionSize)
			po.greedyFlags[s] = make([]int, sectionSize)
		}

		for i := 0; i < globalPartyRows; i++ {
			val := matrix[pi][i]
			po.rawInput[i] = val
			po.greedyInputs[i/sectionSize][i%sectionSize] = val
		}
	}
}

// Greedy uniqueness selection across parties, then fill encryptedInput.
func runGreedyUniqSelectionAndEncrypt(P []*party) {
	intializeEdgeRelated(P)

	edgeSize := len(P) * (len(P) - 1) * sectionNum * sectionNum / 2
	edges := make([]float64, edgeSize)

	// how many samples to encrypt (like processGreedyWithASR)
	thresholdNumber := len(P) * globalPartyRows * encryptionRatio / 100
	markedNumbers := 0

	startTime := time.Now()
	for markedNumbers < thresholdNumber {
		maxScore := -1.0
		firstH, firstS, secondH, secondS := -1, -1, -1, -1

		for e := range edges {
			p1, s1, p2, s2 := getDetailedBlocksForEdge(e, len(P), sectionNum)
			score := calculateUniquenessBetweenBlocks(P, p1, s1, p2, s2)
			edges[e] = score
			if score > maxScore {
				maxScore = score
				firstH, firstS, secondH, secondS = p1, s1, p2, s2
			}
		}

		prev := markedNumbers
		markedNumbers = greedyMarkBlocks(markedNumbers, thresholdNumber, P, firstH, firstS, secondH, secondS)
		if markedNumbers == prev {
			break
		}
	}

	// Fallback random marking in case
	if markedNumbers < thresholdNumber {
		type pos struct{ pi, si, idx int }
		var pool []pos
		for pi := range P {
			for si := 0; si < sectionNum; si++ {
				for k := 0; k < sectionSize; k++ {
					if P[pi].greedyFlags[si][k] == 0 {
						pool = append(pool, pos{pi, si, k})
					}
				}
			}
		}
		for _, p := range pool {
			P[p.pi].greedyFlags[p.si][p.idx] = 1
			markedNumbers++
			if markedNumbers >= thresholdNumber {
				break
			}
		}
	}

	greedyEncryptBlocks(P)
	elapsedTime += time.Since(startTime)
}

func runAttackAverage(P []*party, loops int, atdSize int) float64 {
	if loops < 10 {
		loops = 10
	}

	// Bounds check: ensure ATD size doesn't exceed available data
	if len(P) > 0 && len(P[0].rawInput) > 0 {
		maxATD := len(P[0].rawInput) - 1
		if atdSize > maxATD {
			fmt.Printf("Warning: ATD size %d exceeds available data %d, using %d\n",
				atdSize, len(P[0].rawInput), maxATD)
			atdSize = maxATD
		}
	}

	successCount := 0

	// Save and set ATD
	originalATD := globalAttackerTimeDelta
	setAttackerTimeDelta(atdSize)

	for i := 0; i < loops; i++ {
		successCount += attackPartiesSafe(P, atdSize) // Use safe version
	}

	// Restore original ATD
	setAttackerTimeDelta(originalATD)

	return float64(successCount) / float64(loops)
}

func attackPartiesSafe(P []*party, atdSize int) int {
	if len(P) == 0 || len(P[0].rawInput) == 0 {
		return 0
	}

	// Ensure ATD size is valid
	maxDataSize := len(P[0].rawInput)
	if atdSize >= maxDataSize {
		return 0 // Can't attack with ATD >= data size
	}

	randomParty := getRandom(len(P))
	maxStart := maxDataSize - atdSize
	if maxStart <= 0 {
		return 0
	}

	randomStart := getRandom(maxStart)

	// Bounds check before slicing
	if randomStart+atdSize > len(P[randomParty].rawInput) {
		return 0
	}

	attackerDataBlock := P[randomParty].rawInput[randomStart : randomStart+atdSize]
	matchedHouseholds := identifyPartySafe(P, attackerDataBlock, randomParty, randomStart, atdSize)

	if len(matchedHouseholds) == 1 && matchedHouseholds[0] == randomParty {
		return 1
	}
	return 0
}

func identifyPartySafe(P []*party, arr []float64, party int, index int, atdSize int) []int {
	matchedHouseholds := []int{}

	if party >= len(P) || index+atdSize > len(P[party].encryptedInput) {
		return matchedHouseholds
	}

	dataset := P[party].encryptedInput[index : index+atdSize]

	// Simple exact match for testing
	for pi, po := range P {
		if pi == party {
			continue
		}

		for i := 0; i <= len(po.encryptedInput)-atdSize; i++ {
			if i+atdSize > len(po.encryptedInput) {
				break
			}

			target := po.encryptedInput[i : i+atdSize]
			if reflect.DeepEqual(target, dataset) {
				matchedHouseholds = append(matchedHouseholds, pi)
				break
			}
		}
	}

	if len(matchedHouseholds) == 0 {
		matchedHouseholds = append(matchedHouseholds, party)
	}

	return matchedHouseholds
}

func countEncrypted(P []*party) (total, encrypted int) {
	for _, po := range P {
		for _, val := range po.encryptedInput {
			total++
			if val == -0.1 {
				encrypted++
			}
		}
	}
	return
}

func windowMaskStats(P []*party, windowSize int) (totalWindows, maskedWindows, cleanWindows int) {
	for _, po := range P {
		for i := 0; i <= len(po.encryptedInput)-windowSize; i++ {
			totalWindows++
			hasEncrypted := false

			for j := i; j < i+windowSize; j++ {
				if po.encryptedInput[j] == -0.1 {
					hasEncrypted = true
					break
				}
			}

			if hasEncrypted {
				maskedWindows++
			} else {
				cleanWindows++
			}
		}
	}
	return
}

func loadMatrix(fileList []string) [][]float64 {
	data := make([][]float64, len(fileList))
	for i, filename := range fileList {
		data[i] = resizeCSV(filename)
		// Use a reasonable size that works with your attack function
		maxSize := 168 // 1 week of hourly data
		if len(data[i]) > maxSize {
			data[i] = data[i][:maxSize]
		}
	}
	return data
}

func parseIntList(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	result := make([]int, len(parts))
	for i, part := range parts {
		val, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return nil, err
		}
		result[i] = val
	}
	return result, nil
}

func parseFloatList(s string) ([]float64, error) {
	parts := strings.Split(s, ",")
	result := make([]float64, len(parts))
	for i, part := range parts {
		val, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return nil, err
		}
		result[i] = val
	}
	return result, nil
}

// Optional helper if we later sort selections by score
type byScore []struct {
	i int
	v float64
}

func (a byScore) Len() int           { return len(a) }
func (a byScore) Less(i, j int) bool { return a[i].v > a[j].v }
func (a byScore) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// Silence unused imports
var _ = math.Abs
var _ = sort.Float64s
