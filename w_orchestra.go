package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

// Flags: choose breaking approach, tolerance(s), encryption ratio(s), attack loops
var (
	flagApproach  = flag.String("approach", "adaptive", "sliding | adaptive")
	flagTol       = flag.Float64("tol", 0.30, "single tolerance in [0,1]")
	flagTols      = flag.String("tols", "", "comma-separated tolerances, e.g. 0.25,0.30,0.35")
	flagRatio     = flag.Int("ratio", 60, "single encryption ratio percent")
	flagRatios    = flag.String("ratios", "", "comma-separated ratios, e.g. 35,40,45")
	flagLoops     = flag.Int("loops", 200, "attack loops for ASR estimate")
	flagATD       = flag.Int("atd", 24, "attacker leaked block size (samples)")
	flagMatchPct  = flag.Int("match", 100, "minimum percentage match required (0-100)")
	flagFuzzPct   = flag.Int("fuzz-pct", 100, "percentage of datapoints to fuzz (0-100, 100=all data)")
	flagShowDebug = flag.Bool("debug", true, "print encryption coverage/debug info")
	flagASRDebug  = flag.Bool("asr-debug", false, "print detailed ASR attack verification info")
	flagCSVOut    = flag.String("csv", "test_results.csv", "output CSV file")
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

	// Dataset and global thresholds
	currentDataset = DATASET_ELECTRICITY
	transitionEqualityThreshold = ELECTRICITY_TRANSITION_EQUALITY_THRESHOLD
	asrDebugMode = *flagASRDebug

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

	// Write CSV header
	_, err = csvFile.WriteString("approach,tolerance,ratio,asr,time\n")
	check(err)

	// Iterate all combinations
	fmt.Printf("Combinations: %d tolerances × %d ratios, loops=%d, atdSize=%d, approach=%s\n",
		len(tolVals), len(ratioVals), *flagLoops, *flagATD, *flagApproach)

	for _, tol := range tolVals {
		for _, ratio := range ratioVals {
			asr, took := runOneCombo(original, fileList, *flagApproach, tol, ratio, *flagLoops, *flagATD, *flagFuzzPct, *flagShowDebug)

			// Console output
			fmt.Printf("Result: approach=%s, T=%.2f, ratio=%d%% → ASR=%.4f (loops=%d, time=%s)\n",
				*flagApproach, tol, ratio, asr, *flagLoops, took)

			// CSV output
			csvLine := fmt.Sprintf("%s,%.2f,%d,%.4f,%s\n",
				*flagApproach, tol, ratio, asr, took)
			_, err = csvFile.WriteString(csvLine)
			check(err)
		}
	}

	fmt.Printf("Results saved to %s\n", *flagCSVOut)
}

// Run a single (tol, ratio) combo end-to-end and return ASR
func runOneCombo(original [][]float64, fileList []string, approach string, tol float64, ratio int, loops int, atd int, fuzzPct int, showDebug bool) (float64, time.Duration) {
	comboStart := time.Now()
	// 2) Series breaking from original each time (no carry-over)
	var broken [][]float64
	if tol == 0.0 || strings.ToLower(approach) == "none" {
		broken = original // Don't even call the breaking functions
	} else {
		switch strings.ToLower(approach) {
		case "sliding":
			broken = applyWindowBasedBreaking(original, tol, fuzzPct)
		case "adaptive":
			broken = applyAdaptivePatternBreaking(original, tol, fuzzPct)
		default:
			panic("approach must be sliding|adaptive|none")
		}
	}

	fuzzedData := broken // Keep reference to fuzzed data

	if showDebug {
		totalChanges := 0
		for h := 0; h < len(original); h++ {
			for i := 0; i < len(original[h]); i++ {
				if math.Abs(original[h][i]-broken[h][i]) > 0.001 {
					totalChanges++
				}
			}
		}
		fmt.Printf("DEBUG: Fuzzing changed %d values out of %d total (%.2f%%)\n",
			totalChanges, len(original)*len(original[0]),
			float64(totalChanges)/float64(len(original)*len(original[0]))*100)
	}

	// 3) Build parties and generate inputs from the BROKEN data
	paramsDef := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)

	P := genparties(params, fileList)
	generateInputsFromMatrix(P, broken) // fills rawInput, greedyInputs/Flags, etc.

	// 4) Uniqueness-based selective encryption (greedy)
	encryptionRatio = ratio
	runGreedyUniqSelectionAndEncrypt(P)

	// Set up attack scenario correctly:
	// - Attacker gets leaked data from ORIGINAL (unfuzzed) side channel
	// - But compares against FUZZED encrypted output
	for pi, po := range P {
		// What the attacker can see from encrypted stream (fuzzed + encryption markers)
		po.attackVisibleData = make([]float64, len(po.encryptedInput))
		for i := 0; i < len(po.encryptedInput); i++ {
			if po.encryptedInput[i] == -0.1 {
				po.attackVisibleData[i] = -0.1 // Encrypted
			} else {
				po.attackVisibleData[i] = fuzzedData[pi][i] // Fuzzed value
			}
		}

		// CRITICAL FIX: rawInput should contain FUZZED data for attack targets
		// but leaked data should come from ORIGINAL data
		po.rawInput = po.attackVisibleData
	}

	// Configure attacker globals
	uniqueATD = 0
	atdSize = atd
	min_percent_matched = *flagMatchPct // Set matching percentage from command line
	maxHouseholdsNumber = len(P)
	transitionEqualityThreshold = ELECTRICITY_TRANSITION_EQUALITY_THRESHOLD

	if showDebug {
		total, enc := countEncrypted(P)
		wAll, wMasked, wClean := windowMaskStats(P, atdSize)
		fmt.Printf("Debug: T=%.2f, ratio=%d%% → encrypted %d/%d (%.1f%%), windows masked=%.2f%% clean=%.2f%%\n",
			tol, ratio, enc, total, 100.0*float64(enc)/float64(total),
			100.0*float64(wMasked)/float64(wAll),
			100.0*float64(wClean)/float64(wAll))
	}

	// 5) Set original data for realistic attack leaked sequences
	originalLeakedData = original

	// 6) Run attacker loops on encrypted stream
	asr := runAttackAverage(P, loops)
	return asr, time.Since(comboStart)
}

func countEncrypted(P []*party) (total, encrypted int) {
	for _, po := range P {
		total += len(po.encryptedInput)
		for _, v := range po.encryptedInput {
			if v == -0.1 {
				encrypted++
			}
		}
	}
	return
}

// Load all CSVs into [][]float64
func loadMatrix(files []string) [][]float64 {
	n := len(files)
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := resizeCSV(files[i])
		if len(row) > MAX_PARTY_ROWS {
			row = row[:MAX_PARTY_ROWS]
		}
		out[i] = row
	}
	return out
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

func runAttackAverage(P []*party, loops int) float64 {
	if loops < 10 {
		loops = 10
	}

	successCount := 0
	for i := 0; i < loops; i++ {
		successCount += attackParties(P) // returns 0 or 1

		// Progress logging for ASR debug mode - show every 1000 attacks
		if asrDebugMode && (i+1)%1000 == 0 {
			currentASR := float64(successCount) / float64(i+1)
			fmt.Printf("ASR DEBUG: Progress %d/%d attacks, current ASR: %.4f\n", i+1, loops, currentASR)
		}
	}

	if asrDebugMode {
		finalASR := float64(successCount) / float64(loops)
		fmt.Printf("ASR DEBUG: Final result %d/%d successful attacks, ASR: %.4f\n", successCount, loops, finalASR)
	}

	// Return success rate as proportion
	return float64(successCount) / float64(loops)
}

// Window coverage stats (for debug)
func windowMaskStats(P []*party, L int) (total, masked, clean int) {
	for _, po := range P {
		n := len(po.encryptedInput)
		for s := 0; s+L <= n; s++ {
			total++
			hasMask := false
			for k := 0; k < L; k++ {
				if po.encryptedInput[s+k] == -0.1 {
					hasMask = true
					break
				}
			}
			if hasMask {
				masked++
			} else {
				clean++
			}
		}
	}
	return
}

// Helpers to parse lists
func parseFloatList(s string) ([]float64, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	out := make([]float64, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.ParseFloat(p, 64)
		if err != nil {
			return nil, fmt.Errorf("cannot parse float %q: %w", p, err)
		}
		out = append(out, v)
	}
	return out, nil
}

func parseIntList(s string) ([]int, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("cannot parse int %q: %w", p, err)
		}
		out = append(out, v)
	}
	return out, nil
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
