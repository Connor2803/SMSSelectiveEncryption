package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
	utils "lattigo"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"time"
)

type RLChoice struct {
	HouseholdID      string    `json:"household_id"`
	SectionNumber    int       `json:"section_number"`
	Ratio            float64   `json:"ratio"`
	RawEntropy       float64   `json:"raw_entropy"`
	RemainingEntropy float64   `json:"remaining_entropy"`
	UtilityReadings  []float64 `json:"original_utility_readings"`
} // RLChoice structure sent from the Python script.

type GoMetrics struct {
	GlobalReidentificationDurationNS int64                        `json:"globalReidentificationDurationNS"`
	GlobalReidentificationRate       float64                      `json:"globalReidentificationRate"`
	GlobalMemoryConsumption          float64                      `json:"globalMemoryConsumptionMiB"`
	AllPartyLevelMetrics             map[string]PartyLevelMetrics `json:"allPartyMetrics"` // Keyed by Household ID
} // Struct that will be returned from this Go program back to the RL model to calculate the final reward.

type PartyLevelMetrics struct {
	PartyID            string  `json:"partyID"`
	SummationError     float64 `json:"summationError"`
	DeviationError     float64 `json:"deviationError"`
	EncryptionTimeNS   int64   `json:"encryptionTimeNS"`
	DecryptionTimeNS   int64   `json:"decryptionTimeNS"`
	SummationOpsTimeNS int64   `json:"summationOpsTimeNS"`
	DeviationOpsTimeNS int64   `json:"deviationOpsTimeNS"`
}
type Party struct {
	householdID             string
	sections                map[int]*Section // map[SectionNumber]*Section
	summationError          float64
	deviationError          float64
	encryptionTime          time.Duration
	decryptionTime          time.Duration
	summationOperationsTime time.Duration
	deviationOperationsTime time.Duration
} // Struct to represent an individual household.

type Section struct {
	readings                 []float64    // The unprocessed individual electricity meter readings in the section.
	encryptedReadings        []float64    // The encrypted individual meter readings in the section.
	encryptedReadingsIndices map[int]bool // The encrypted readings indexes in the original unprocessed readings slice.
	unencryptedReadings      []float64    // The plain-text individual meter readings in the section.
	expSummation             float64      // The sum of the individual electricity meter readings in the section.
	expAverage               float64      // The sum of the individual electricity meter readings in the section divided by the number of meter readings in the section.
	expDeviation             float64      // The standard deviation of the individual electricity meter readings in the section.
	encryptionRatio          float64      // The RL model chosen encryption ratio for the section.
	plainSum                 float64      // The sum of the plaintext individual electricity meter readings in the section.
} // Section holds data for a single block of electricity meter readings of a household.

const MAXPARTYROWS = 10240                         // Total records/meter readings per household (WATER dataset fewest row count: 20485, WATER dataset greatest row count: 495048, ELECTRICITY dataset fewest row count: 19188, ELECTRICITY dataset greatest row count: 19864)
const MAXSECTIONNUMBER = 10                        /// Maximum number of sections to be included in a household (each section will contain MAXPARTYROWS / SECTIONSIZE reading entries).
const SECTIONSIZE = 1024                           // Default value for the number of utility reading rows to be in each section.
var maxReidentificationAttempts = 100              // Default value for number of loops for runReidentificationAttack.
var leakedPlaintextSize = 12                       // Number of electricity meter readings included in the leaked attacker data
var reidentificationMatchThreshold = 70            // Default value for minimum percentage match for identification.
var usedRandomSectionsByParty = map[string][]int{} // 2D slice to hold which household sections have been used as the attacker block.

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

func getExecutableDir() string {
	exePath, err := os.Executable()
	if err != nil {
		log.Fatalf("Failed to get executable path: %v", err)
	}
	return filepath.Dir(exePath)
}

func main() {
	// 1. Get input file path and leakedPlaintextSize from command-line arguments.
	if len(os.Args) < 3 {
		log.Fatal("Usage: ./generate_metrics_v2 <path_to_rl_choices.json> <leakedPlaintextSize>")
	}
	rlChoicesPath := os.Args[1]
	var err error
	leakedPlaintextSize, err = strconv.Atoi(os.Args[2])
	check(err)

	// 2. Read the RL agent's choices from the JSON file
	choices, err := readRLChoices(rlChoicesPath)
	check(err)
	if len(choices) == 100 {
		maxReidentificationAttempts = 30 // As validation and testing dataset much smaller.
	}

	// 3. Load raw data from the master inputs.csv file
	inputCSVPath := filepath.Join(getExecutableDir(), "inputs_V2_ELECTRICITY.csv")
	parties, err := loadDataFromInputsCSV(inputCSVPath, choices)
	check(err)

	// 4. Perform Homomorphic Encryption (HE) cost simulation
	paramsDef := utils.PN10QP27CI
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	check(err)
	doHomomorphicOperations(params, parties)

	// 5. Run memory consumption function.
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	globalAllocatedMemMiB := float64(bToMb(m.Alloc))

	// 5. Run the Member Identification Attack simulation
	reidentificationRate := 0.0
	startReidentificationTime := time.Now()
	reidentificationResult := runReidentificationAttack(parties)

	if len(reidentificationResult) > 0 {
		// Convert success counts to ASRs (Attack Success Rates)
		var reidentificationRates []float64
		totalAttackableInstancesPerRun := float64(len(parties))
		if totalAttackableInstancesPerRun == 0 {
			fmt.Println("ERROR: totalAttackableInstancesPerRun is 0. Cannot calculate ASR.")
		} else {
			for _, successCount := range reidentificationResult {
				reidentificationRates = append(reidentificationRates, successCount/totalAttackableInstancesPerRun)
			}
		}

		_, mean, _ := calculateStandardDeviationAndMeanAndVariance(reidentificationRates)
		reidentificationRate = mean
	}
	endReidentificationTime := time.Now()
	reidentificationTotalDuration := endReidentificationTime.Sub(startReidentificationTime)
	reidentificationDurationPerLoopNS := int64(0)
	if len(reidentificationResult) > 0 {
		reidentificationDurationPerLoopNS = reidentificationTotalDuration.Nanoseconds() / int64(len(reidentificationResult))
	}

	// 6. Prepare the final metrics and print to stdout as JSON
	finalPartyMetrics := make(map[string]PartyLevelMetrics)

	for partyID, partyData := range parties {
		finalPartyMetrics[partyID] = PartyLevelMetrics{
			PartyID:            partyID,
			SummationError:     partyData.summationError,
			DeviationError:     partyData.deviationError,
			EncryptionTimeNS:   partyData.encryptionTime.Nanoseconds(),
			DecryptionTimeNS:   partyData.decryptionTime.Nanoseconds(),
			SummationOpsTimeNS: partyData.summationOperationsTime.Nanoseconds(),
			DeviationOpsTimeNS: partyData.deviationOperationsTime.Nanoseconds(),
		}
	}

	finalResults := GoMetrics{
		GlobalReidentificationRate:       reidentificationRate,
		GlobalReidentificationDurationNS: reidentificationDurationPerLoopNS,
		GlobalMemoryConsumption:          globalAllocatedMemMiB,
		AllPartyLevelMetrics:             finalPartyMetrics,
	}

	output, err := json.Marshal(finalResults)
	check(err)
	fmt.Println(string(output)) // This is captured by the Python script.

}

// readRLChoices parses the JSON file created by the Python script.
func readRLChoices(path string) ([]RLChoice, error) {
	file, err := os.ReadFile(path)
	check(err)
	var choices []RLChoice
	err = json.Unmarshal(file, &choices)
	check(err)
	fmt.Fprintf(os.Stderr, "DEBUG: readRLChoices - Successfully read %d choices from %s\n", len(choices), path)
	if len(choices) > 0 {
		//fmt.Fprintf(os.Stderr, "DEBUG: First choice: HouseholdID=%s, SectionNumber=%d, Ratio=%f\n", choices[0].HouseholdID, choices[0].SectionNumber, choices[0].Ratio)
	}
	return choices, nil
}

// loadDataFromInputsCSV reads the master CSV and organises data into Parties and Sections.
func loadDataFromInputsCSV(path string, choices []RLChoice) (map[string]*Party, error) {
	file, err := os.Open(path)
	check(err)
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	check(err)

	// Create a lookup map for RL choices using same key.
	choiceMap := make(map[string]float64) // key: "householdID-sectionNum"
	chosenHouseholdIDs := make(map[string]bool)

	for _, c := range choices {
		key := fmt.Sprintf("%s-%d", c.HouseholdID, c.SectionNumber)
		choiceMap[key] = c.Ratio
		chosenHouseholdIDs[c.HouseholdID] = true
	}

	parties := make(map[string]*Party)
	// Skip header row (i=0)
	for i := 1; i < len(records); i++ {
		record := records[i]
		householdID := record[0]
		sectionNum, _ := strconv.Atoi(record[1])

		if !chosenHouseholdIDs[householdID] {
			continue
		}

		// Find the party or create a new one
		if _, ok := parties[householdID]; !ok {
			parties[householdID] = &Party{
				householdID: householdID,
				sections:    make(map[int]*Section),
			}
		}
		party := parties[householdID]

		// Get the readings for this section (stored as a JSON string array in CSV).
		var readings []float64
		if err := json.Unmarshal([]byte(record[4]), &readings); err != nil {
			log.Printf("Warning: Skipping malformed readings in row %d: %v", i, err)
			continue
		}

		// Get the encryption ratio for this specific section from the RL agent's choices,
		key := fmt.Sprintf("%s-%d", householdID, sectionNum)
		ratio, ok := choiceMap[key]
		if !ok {
			log.Printf("Warning: No ratio found for household %s section %d. Skipping.", householdID, sectionNum)
			continue
		}

		// Sum the readings array to calculate the utility usage for the section.
		sumUsage := 0.0
		for _, reading := range readings {
			sumUsage += reading
		}

		stdUsage, avgUsage, _ := calculateStandardDeviationAndMeanAndVariance(readings)

		party.sections[sectionNum] = &Section{
			readings:        readings,
			expSummation:    sumUsage,
			expAverage:      avgUsage,
			expDeviation:    stdUsage,
			encryptionRatio: ratio,
		}
	}
	fmt.Fprintf(os.Stderr, "DEBUG: loadDataFromInputsCSV - Finished processing. Number of parties loaded: %d\n", len(parties))
	return parties, nil
}

func doHomomorphicOperations(params ckks.Parameters, parties map[string]*Party) {
	// Key Generation Variables.
	tkgen := ckks.NewKeyGenerator(params)
	var tsk *rlwe.SecretKey
	var tpk *rlwe.PublicKey

	tsk = tkgen.GenSecretKey()
	tpk = tkgen.GenPublicKey(tsk)

	var rlk *rlwe.RelinearizationKey
	rlk = tkgen.GenRelinearizationKey(tsk, 1)

	rotations := params.RotationsForInnerSum(1, SECTIONSIZE)
	var rotk *rlwe.RotationKeySet
	rotk = tkgen.GenRotationKeysForRotations(rotations, false, tsk)

	decryptor := ckks.NewDecryptor(params, tsk)
	encoder := ckks.NewEncoder(params)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotk})

	// Generate ciphertexts by calling performHEEncryption helper encryption function.
	encOutputs, encInputsSummation, encInputsNegative := performHEEncryption(params, parties, tpk, encoder)

	// Decryption Timing
	for partyID, partyCiphers := range encOutputs {
		if len(partyCiphers) > 0 {
			start := time.Now()
			for _, ct := range partyCiphers {
				_ = decryptor.DecryptNew(ct)
			}
			parties[partyID].decryptionTime = time.Since(start)
		}
	}

	// Summation calculations ====================================================
	encSummationOuts := make(map[string]*rlwe.Ciphertext)
	for partyID, partySections := range encInputsSummation {
		if len(partySections) == 0 {
			continue
		}
		partySummationOpsStart := time.Now()

		tmpCiphertext := partySections[0]
		for j := 1; j < len(partySections); j++ {
			evaluator.Add(tmpCiphertext, partySections[j], tmpCiphertext)
		}

		evaluator.InnerSum(tmpCiphertext, 1, params.Slots(), tmpCiphertext)

		parties[partyID].summationOperationsTime = time.Since(partySummationOpsStart)
		encSummationOuts[partyID] = tmpCiphertext
	}

	// Deviation calculation ====================================================
	encDeviationOuts := make(map[string]*rlwe.Ciphertext)
	for partyID, party := range encInputsNegative {
		if len(party) == 0 || encSummationOuts[partyID] == nil {
			continue
		}
		partyDeviationOpsStart := time.Now()

		avgCiphertext := encSummationOuts[partyID].CopyNew()
		avgCiphertext.Scale = avgCiphertext.Scale.Mul(rlwe.NewScale(SECTIONSIZE))

		var aggregatedDeviation *rlwe.Ciphertext

		for sectionIndex, sectionCipher := range party {
			// Step 1: Subtract the average from the value (by adding the negative value).
			currentDeviation := evaluator.AddNew(sectionCipher, avgCiphertext)

			// Step 2: Square the result.
			evaluator.MulRelin(currentDeviation, currentDeviation, currentDeviation)

			// Step 3: Aggregate the squared differences.
			if sectionIndex == 0 {
				aggregatedDeviation = currentDeviation
			} else {
				evaluator.Add(aggregatedDeviation, currentDeviation, aggregatedDeviation)
			}
		}

		// Step 4: Perform the final InnerSum (rotation) on the aggregated result.
		evaluator.InnerSum(aggregatedDeviation, 1, params.Slots(), aggregatedDeviation)

		aggregatedDeviation.Scale = aggregatedDeviation.Scale.Mul(rlwe.NewScale(len(parties)))
		encDeviationOuts[partyID] = aggregatedDeviation
		parties[partyID].deviationOperationsTime = time.Since(partyDeviationOpsStart)
	}

	// Collect Errors ===============================================
	ptresSummation := ckks.NewPlaintext(params, params.MaxLevel())
	for partyID, partyData := range parties {
		var sumErr float64
		for _, section := range partyData.sections {
			if encSummationOuts[partyID] != nil {
				decryptor.Decrypt(encSummationOuts[partyID], ptresSummation)
				resSummation := encoder.Decode(ptresSummation, params.LogSlots())
				totalCKKSSum := real(resSummation[0]) + section.plainSum
				err := totalCKKSSum - section.expSummation
				sumErr += math.Abs(err) // Total summation error per section, per household/party.
			}
		}
		parties[partyID].summationError = sumErr
	}

	ptresDeviation := ckks.NewPlaintext(params, params.MaxLevel())
	for partyID, partyData := range parties {
		var devErr float64
		for _, section := range partyData.sections {
			if encDeviationOuts[partyID] != nil {
				decryptor.Decrypt(encDeviationOuts[partyID], ptresDeviation)
				resDeviation := encoder.Decode(ptresDeviation, params.LogSlots())
				err := real(resDeviation[0]) - section.expDeviation
				devErr += math.Abs(err) // Total deviation error per section, per household/party.
			}
		}
		parties[partyID].deviationError = devErr
	}

	return
}

// encPhase is a helper function that encrypts the plaintext inputs for each party and returns their encrypted values.
// It also collects the time taken for each party's encryption.
func performHEEncryption(params ckks.Parameters, parties map[string]*Party, pk *rlwe.PublicKey, encoder ckks.Encoder) (encOutputs, encInputsSummation, encInputsNegative map[string][]*rlwe.Ciphertext) {

	encOutputs = make(map[string][]*rlwe.Ciphertext)
	encInputsSummation = make(map[string][]*rlwe.Ciphertext)
	encInputsNegative = make(map[string][]*rlwe.Ciphertext)

	// Each party encrypts its input vector.
	encryptor := ckks.NewEncryptor(params, pk)
	pt := ckks.NewPlaintext(params, params.MaxLevel())

	for partyID, party := range parties {
		start := time.Now() // Start timing for the current party (pi).

		for sectionIndex, section := range party.sections {
			if sectionIndex == 0 { // This ensures the inner slices are initialized once per party.
				encOutputs[partyID] = make([]*rlwe.Ciphertext, 0)
				encInputsSummation[partyID] = make([]*rlwe.Ciphertext, 0)
				encInputsNegative[partyID] = make([]*rlwe.Ciphertext, 0)
			}
			readings := section.readings
			encryptCount := int(float64(len(readings)) * section.encryptionRatio)
			if encryptCount == 0 {
				continue
			}
			tmpEncryptedReadings := make([]float64, encryptCount)
			encryptedIndices := make(map[int]bool, encryptCount)
			perm := rand.Perm(len(readings))[:encryptCount]
			for i, idx := range perm {
				tmpEncryptedReadings[i] = readings[idx] // Encrypt a random percentage sample of the readings within a section according to the RL model assigned encryption ratio.
				encryptedIndices[idx] = true
			}

			// Encrypt original input.
			encoder.Encode(tmpEncryptedReadings, pt, params.LogSlots())
			tmpCiphertext := ckks.NewCiphertext(params, 1, params.MaxLevel())
			encryptor.Encrypt(pt, tmpCiphertext)
			encOutputs[partyID] = append(encOutputs[partyID], tmpCiphertext)
			encInputsSummation[partyID] = append(encInputsSummation[partyID], tmpCiphertext)

			// Turn po.input to negative for HE operations (subtraction and deviation).
			negReadings := make([]float64, len(tmpEncryptedReadings))
			for i, val := range tmpEncryptedReadings {
				negReadings[i] = -val
			}

			// Encrypt negative input.
			encoder.Encode(negReadings, pt, params.LogSlots())
			tmpCiphertext = ckks.NewCiphertext(params, 1, params.MaxLevel())
			encryptor.Encrypt(pt, tmpCiphertext)
			encInputsNegative[partyID] = append(encInputsNegative[partyID], tmpCiphertext)

			plainSum := 0.0
			tmpPlaintTextReadings := make([]float64, 0, len(readings))
			for idx, val := range readings {
				if !encryptedIndices[idx] {
					plainSum += val
					tmpPlaintTextReadings = append(tmpPlaintTextReadings, val)
				}
			}
			section.plainSum = plainSum
			section.encryptedReadings = tmpEncryptedReadings
			section.encryptedReadingsIndices = encryptedIndices
			section.unencryptedReadings = tmpPlaintTextReadings

		}
		parties[partyID].encryptionTime = time.Since(start)
	}
	return encOutputs, encInputsSummation, encInputsNegative
}

// These group of functions investigate if an attacker was able to gather unencrypted/plain-text electricity meter readings
// would they be able to find out which household did it originate from within entire dataset.
func runReidentificationAttack(parties map[string]*Party) []float64 {
	var attackCount int
	var successCounts = []float64{} // Collects the number of successful attacks for each loop.
	var std float64
	var standardError float64

	partyIDs := make([]string, 0, len(parties))
	for id := range parties {
		partyIDs = append(partyIDs, id)
	}

	for attackCount = 0; attackCount < maxReidentificationAttempts; attackCount++ {
		//fmt.Fprintf(os.Stderr, "DEBUG: runReidentificationAttack - Attempt loop %d/%d\n", attackCount+1, maxReidentificationAttempts)
		var successNum = identifySourceHousehold(parties, partyIDs) // Integer result of one attack run.
		successCounts = append(successCounts, float64(successNum))  // Stores the success count for this run.

		// NOTE: These calculations are for internal stopping conditions, not for statistical purposes.
		std, _, _ = calculateStandardDeviationAndMeanAndVariance(successCounts)
		standardError = std / math.Sqrt(float64(len(successCounts)))
		if standardError <= 0.01 && attackCount >= 100 { // Stop attack earlier if results stabilise.
			//fmt.Fprintf(os.Stderr, "DEBUG: runReidentificationAttack - Stopping early due to stable results (standard error <= 0.01) after %d attacks.\n", attackCount+1)
			attackCount++
			break
		}
	}
	return successCounts
}

func identifySourceHousehold(parties map[string]*Party, partyIDs []string) (reidenitificationSuccessNum int) {
	reidenitificationSuccessNum = 0

	var valid bool
	var randomPartyID string
	var randomSectionIndex, offset int
	var attackerDataBlock []float64
	iteration := 0

	for !valid {
		iteration++
		if iteration > 500 {
			//fmt.Fprintln(os.Stderr, "WARNING: identifySourceHousehold - Could not find a unique data block after 500 iterations. Skipping this attack run.")
			return 0
		}
		randomPartyID = partyIDs[rand.Intn(len(partyIDs))]
		randomSectionIndex = getRandomSection(randomPartyID)

		section, ok := parties[randomPartyID].sections[randomSectionIndex]
		if !ok || len(section.readings) < leakedPlaintextSize {
			//fmt.Fprintf(os.Stderr, "DEBUG: identifySourceHousehold - Skipping party %s section %d (not found or too small).\n", randomPartyID, randomSectionIndex)
			continue
		}

		offset = getRandom(len(section.readings) - leakedPlaintextSize + 1)
		attackerDataBlock = section.readings[offset : offset+leakedPlaintextSize]

		//fmt.Fprintf(os.Stderr, "DEBUG: identifySourceHousehold - Checking uniqueness for party %s, section %d, offset %d...\n", randomPartyID, randomSectionIndex, offset)
		if uniqueDataBlock(parties, attackerDataBlock, randomPartyID, randomSectionIndex, "sections") {
			//fmt.Fprintln(os.Stderr, "DEBUG: identifySourceHousehold - Found unique data block.")
			valid = true
		} else {
			//fmt.Fprintln(os.Stderr, "DEBUG: identifySourceHousehold - Data block not unique, trying again.")

		}
	}

	//fmt.Fprintf(os.Stderr, "DEBUG: identifySourceHousehold - Calling identifyParty for originalPartyID: %s\n", randomPartyID)
	matchedHouseholds := identifyParty(parties, attackerDataBlock, randomPartyID)
	//fmt.Fprintf(os.Stderr, "DEBUG: identifySourceHousehold - identifyParty returned %d matches.\n", len(matchedHouseholds))

	if len(matchedHouseholds) == 1 && matchedHouseholds[0] == randomPartyID {
		reidenitificationSuccessNum++
		//fmt.Fprintln(os.Stderr, "DEBUG: identifySourceHousehold - Attack successful: Original party uniquely identified.")
	} else {
		//fmt.Fprintln(os.Stderr, "DEBUG: identifySourceHousehold - Attack failed: Original party not uniquely identified.")
		if len(matchedHouseholds) > 1 {
			//fmt.Fprintf(os.Stderr, "DEBUG: Matched households: %v\n", matchedHouseholds)
		}
	}
	return reidenitificationSuccessNum
}

// getRandomSection is a helper function that returns an unused random start block/section for the Party
func getRandomSection(partyID string) int {
	var valid bool = false
	var randomSection int

	for !valid {
		randomSection = getRandom(MAXSECTIONNUMBER)
		if !contains(partyID, randomSection) {
			usedRandomSectionsByParty[partyID] = append(usedRandomSectionsByParty[partyID], randomSection)
			valid = true
		}
	}
	return randomSection
}

// contains is a helper function that checks if the Party has used the random start block/section before.
func contains(partyID string, randomStart int) bool {
	var contains bool = false

	val, exists := usedRandomSectionsByParty[partyID]

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
func uniqueDataBlock(parties map[string]*Party, arr []float64, partyName string, sectionIndex int, inputType string) bool {
	var unique bool = true
	comparisonCount := 0

	for partyID, partyData := range parties {
		if partyID == partyName {
			continue
		}
		if inputType == "sections" {
			// Only compare to the same section index.
			section, ok := partyData.sections[sectionIndex]
			if !ok || len(section.readings) < leakedPlaintextSize {
				continue
			}
			householdData := section.readings
			for i := 0; i < len(householdData)-leakedPlaintextSize+1; i++ {
				comparisonCount++
				var target = householdData[i : i+leakedPlaintextSize]
				if reflect.DeepEqual(target, arr) {
					unique = false
					usedRandomSectionsByParty[partyID] = append(usedRandomSectionsByParty[partyID], i)
					break
				}
			}
		} else {
			// Compare to all sections in the party.
			for _, section := range partyData.sections {
				readings := section.readings
				if len(readings) < leakedPlaintextSize {
					continue
				}
				for i := 0; i < len(readings)-leakedPlaintextSize+1; i++ {
					var target = readings[i : i+leakedPlaintextSize]
					if reflect.DeepEqual(target, arr) {
						unique = false
						usedRandomSectionsByParty[partyID] = append(usedRandomSectionsByParty[partyID], i)
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
	//fmt.Fprintf(os.Stderr, "DEBUG: uniqueDataBlock - Completed %d comparisons. Unique: %t\n", comparisonCount, unique)
	return unique
}

func identifyParty(parties map[string]*Party, attackerPlainTextBlock []float64, originalPartyID string) []string {
	var matchedHouseholds []string
	matchAttempts := 0

	// Match threshold: # of elements that must match.
	minMatchCount := math.Ceil(float64(leakedPlaintextSize) * float64(reidentificationMatchThreshold) / 100)
	//fmt.Fprintf(os.Stderr, "DEBUG: identifyParty - minMatchCount: %.0f\n", minMatchCount)

	for partyID, party := range parties {
		//if partyID == originalPartyID {
		//	continue
		//}
		for _, section := range party.sections {
			unencryptedReadings := section.unencryptedReadings
			if len(unencryptedReadings) < len(attackerPlainTextBlock) {
				continue
			}

			// Attacker slides their block over the target's unencrypted readings to find a match.
			for i := 0; i <= len(unencryptedReadings)-len(attackerPlainTextBlock); i++ {
				matchAttempts++
				targetWindow := unencryptedReadings[i : i+len(attackerPlainTextBlock)]
				matchCount := 0
				for k := 0; k < len(attackerPlainTextBlock); k++ {
					if math.Abs(attackerPlainTextBlock[k]-targetWindow[k]) < 1e-9 {
						matchCount++
					}
				}

				if float64(matchCount) >= minMatchCount {
					matchedHouseholds = append(matchedHouseholds, partyID)
					//fmt.Fprintf(os.Stderr, "DEBUG: identifyParty - Found match for party %s in section (matchCount: %d)\n", partyID, matchCount)
					goto NextParty // Found a match, stop checking other sections for this party.
				}
			}
		}
	NextParty:
	}

	//fmt.Fprintf(os.Stderr, "DEBUG: identifyParty - Total match attempts: %d\n", matchAttempts)

	// Deduplicate party matches.
	unique := make(map[string]bool)
	var finalMatches []string
	for _, id := range matchedHouseholds {
		if !unique[id] {
			unique[id] = true
			finalMatches = append(finalMatches, id)
		}
	}

	return finalMatches
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
