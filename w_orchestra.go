package main

import (
	"flag"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v4/ckks"
)

// Flags: choose breaking approach, tolerance, encryption ratio, attack loops
var (
	flagApproach = flag.String("approach", "adaptive", "sliding | adaptive")
	flagTol      = flag.Float64("tol", 0.30, "series-breaking tolerance in [0,1]")
	flagRatio    = flag.Int("ratio", 60, "encryption ratio percent")
	flagLoops    = flag.Int("loops", 200, "attack loops for ASR estimate")
)

func main() {
	flag.Parse()

	// Dataset and global thresholds
	currentDataset = DATASET_ELECTRICITY
	transitionEqualityThreshold = ELECTRICITY_TRANSITION_EQUALITY_THRESHOLD

	// 1) Load data using utils.getFileList + resizeCSV
	println("Testing series breaking and uniqueness selection for electricity data...")
	fileList := getFileList("electricity")
	if len(fileList) == 0 {
		panic("no dataset files found")
	}
	data := loadMatrix(fileList) // [][]float64

	// 2) Series breaking
	var broken [][]float64
	switch strings.ToLower(*flagApproach) {
	case "sliding":
		broken = applyWindowBasedBreaking(data, *flagTol)
	case "adaptive":
		broken = applyAdaptivePatternBreaking(data, *flagTol)
	default:
		panic("approach must be sliding|adaptive")
	}

	// 3) Build parties and generate inputs from the BROKEN data
	paramsDef := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)

	P := genparties(params, fileList)
	generateInputsFromMatrix(P, broken) // fills rawInput, greedyInputs/Flags, etc.

	encryptionRatio = *flagRatio
	runGreedyUniqSelectionAndEncrypt(P)

	// Force the attacker to read the encrypted stream
	for _, po := range P {
		buf := make([]float64, len(po.encryptedInput))
		copy(buf, po.encryptedInput)
		po.rawInput = buf
	}

	// Configure attacker globals (same as test harness expectations)
	uniqueATD = 0                // allow non-unique leaked blocks
	atdSize = 24                 // leaked block length (set > 1)
	maxHouseholdsNumber = len(P) // use all parties
	transitionEqualityThreshold = ELECTRICITY_TRANSITION_EQUALITY_THRESHOLD

	// Debug: how much did we actually encrypt?
	total, enc := countEncrypted(P)
	fmt.Printf("Debug: encrypted %d/%d samples (%.1f%%), atdSize=%d\n",
		enc, total, 100.0*float64(enc)/float64(total), atdSize)

	// 5) Run attacker loops on encrypted stream
	asr := runAttackAverage(P, *flagLoops)

	fmt.Printf("Break→Uniq pipeline: %s, T=%.2f, ratio=%d%% → ASR=%.4f (loops=%d)\n",
		*flagApproach, *flagTol, *flagRatio, asr, *flagLoops)
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

	// Fallback random marking if needed
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
		// deterministic order is fine; could shuffle
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

// Simple average ASR over N loops using the existing attacker
func runAttackAverage(P []*party, loops int) float64 {
	if loops < 1 {
		loops = 1
	}
	success, cnt := 0, 0
	start := time.Now()
	for i := 0; i < loops; i++ {
		success += attackParties(P) // returns 0/1
		cnt++
	}
	elapsedTime += time.Since(start)
	return float64(success) / float64(cnt)
}

// Optional helper if you later sort selections by score
type byScore []struct {
	i int
	v float64
}

func (a byScore) Len() int           { return len(a) }
func (a byScore) Less(i, j int) bool { return a[i].v > a[j].v }
func (a byScore) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// Silence unused import if you trim anything.
var _ = math.Abs
var _ = sort.Float64s
