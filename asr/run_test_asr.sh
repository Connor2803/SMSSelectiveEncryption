#!/bin/bash

# Define possible values for parameters
strategies=(1 2 3)
datasets=(1 2)
targets=(1 2)
uniqueATDbool=0
encryptionRatios=(0 20 40 60 80 100)
maxHouseholdsNumber=80

# Create output directory
output_dir="avg_asr_tests_text"
mkdir -p "$output_dir"
sleep 5  # Sleep for 5 seconds before starting

# Run Go program for each combination
for dataset in "${datasets[@]}"; do
  for strategy in "${strategies[@]}"; do
    for target in "${targets[@]}"; do
      for encryptionRatio in "${encryptionRatios[@]}"; do
        echo "Running: go run ./test_asr.go $strategy $dataset $target $uniqueATDbool $encryptionRatio $maxHouseholdsNumber"
        go run "./test_asr.go" "$strategy" "$dataset" "$target" "$uniqueATDbool" "$encryptionRatio" "$maxHouseholdsNumber"

        # Define expected output file name (in parent directory)
        output_file="../test_avg_asr_time_${strategy}_${dataset}_${target}_${uniqueATDbool}_${encryptionRatio}_${maxHouseholdsNumber}.txt"

        # Wait for up to 1 hour for the file to appear
        counter=0
        while [ ! -f "$output_file" ] && [ $counter -lt 3600 ]; do
          sleep 10  # Check every 10 seconds
          counter=$((counter + 10))
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
  done
done

echo "All tests completed. Outputs moved to $output_dir"
