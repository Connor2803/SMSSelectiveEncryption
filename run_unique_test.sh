#!/bin/bash
# filepath: run_uniqueness_comprehensive.sh

echo "=== Uniqueness Strategy Comprehensive Testing ==="

# Create directory structure
mkdir -p tests/uniqueness_comprehensive
mkdir -p tests/results

# Copy the standalone test if it doesn't exist
if [ ! -f "tests/uniqueness_comprehensive/test_uniqueness_standalone.go" ]; then
    echo "Please ensure test_uniqueness_standalone.go is in tests/uniqueness_comprehensive/"
    exit 1
fi

# Test parameters
strategies=(1 2 3)  # Global, Local, Adaptive
datasets=(1 2)      # Water, Electricity  
encryption_ratios=(20 40 60 80)
households=20

echo "Starting comprehensive uniqueness testing..."
echo "Strategies: ${strategies[*]}"
echo "Datasets: ${datasets[*]}"
echo "Encryption ratios: ${encryption_ratios[*]}"
echo "Households: $households"
echo ""

# Run tests
test_count=0
total_tests=$((${#strategies[@]} * ${#datasets[@]} * ${#encryption_ratios[@]}))

for strategy in "${strategies[@]}"; do
    for dataset in "${datasets[@]}"; do
        for ratio in "${encryption_ratios[@]}"; do
            test_count=$((test_count + 1))
            echo "[$test_count/$total_tests] Testing: Strategy=$strategy, Dataset=$dataset, Ratio=$ratio"
            
            cd tests/uniqueness_comprehensive
            go run test_uniqueness_standalone.go "$strategy" "$dataset" "0" "1" "$ratio" "$households"
            cd ../..
            
            # Brief pause between tests
            sleep 1
        done
    done
done

echo ""
echo "=== Testing Complete ==="
echo "Results saved in tests/results/"

# Generate summary
echo "=== Generating Summary ==="
echo "Strategy,Dataset,Target,Ratio,Households,ASR,Privacy,Timestamp" > tests/results/combined_results.csv

# Combine all individual result files
for file in tests/results/uniqueness_test_*.csv; do
    if [ -f "$file" ]; then
        tail -n +2 "$file" >> tests/results/combined_results.csv
    fi
done

echo "Combined results saved to tests/results/combined_results.csv"

# Count total tests
result_files=$(ls tests/results/uniqueness_test_*.csv 2>/dev/null | wc -l)
echo "Generated $result_files individual result files"