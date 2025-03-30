#!/bin/bash

# Define possible values for parameters
datasets=(1 2)
strategies=(1 2 3)
maxHouseholdsNumber=80

# Create output directory
output_dir="block_tests_text"
mkdir -p "$output_dir"
sleep 5
# Run Go program for each combination
for dataset in "${datasets[@]}"; do
  for strategy in "${strategies[@]}"; do
    echo "Running: go run ./blocksize_choices/test_blocks.go $strategy $dataset $maxHouseholdsNumber"
    go run "./blocksize_choices/test_blocks.go" "$strategy" "$dataset" "$maxHouseholdsNumber"
    sleep 15
    # Move the expected output file to block_tests folder
    output_file="test_blocks_${strategy}_${dataset}_${maxHouseholdsNumber}.txt"
    if [ -f "$output_file" ]; then
      mv "$output_file" "$output_dir/"
    else
      echo "Warning: Expected output file $output_file not found!"
    fi
  done
done

echo "All tests completed. Outputs moved to $output_dir"
