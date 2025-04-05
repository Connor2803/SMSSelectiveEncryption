#!/bin/bash

#running command: // strategy, dataset, target, uniqueATD, encryptionRatio, atdSize, maxHouseholdsNumber
#go run .\asr\test_asr.go 1 2 1 0 60 24 80

# Define possible values for parameters
strategies=(1 2 3) # 1 = global, 2 = household, 3 = random
dataset=1 # 1 = water, 2 = electricity
target=1 # 1 = entropy, 2 = transition
uniqueATDbool=0
encryptionRatio=0 # range between 0 - 100, in 20 integer increments
atdSizes=(3 6 9 12 15 18 21 24 27 30 33 36 39 42 45 48)
maxHouseholdsNumber=80

# Create output directory
output_dir="./asr/asr_tests_water_entropy_0_text"
mkdir -p "$output_dir"
sleep 5  # Sleep for 5 seconds before starting

# Run Go program for each combination
for strategy in "${strategies[@]}"; do
    for atdSize in "${atdSizes[@]}"; do
        echo "Running: go run ./asr/test_asr.go $strategy $dataset $target $uniqueATDbool $encryptionRatio $atdSize $maxHouseholdsNumber"
        go run "./asr/test_asr.go" "$strategy" "$dataset" "$target" "$uniqueATDbool" "$encryptionRatio" "$atdSize" "$maxHouseholdsNumber"

        # Define expected output file name (in parent directory)
        output_file="./ASR_time_${strategy}_${dataset}_${target}_${uniqueATDbool}_${encryptionRatio}_${atdSize}_${maxHouseholdsNumber}.txt"

        # Wait for up to 1 hour for the file to appear
        counter=0
        while [ ! -f "$output_file" ] && [ $counter -lt 3600 ]; do
          sleep 10  # Check every 10 seconds
          counter=$((counter + 10))
          echo $counter
        done

        # Move the file if it exists
        if [ -f "$output_file" ]; then
          echo "Found file: $output_file, moving to $output_dir/"
          mv "$output_file" "$output_dir/"
        else
          echo "Warning: Expected output file $output_file not found after 1 hour!"
        fi
    done
done
echo "All tests completed. Outputs moved to $output_dir"
