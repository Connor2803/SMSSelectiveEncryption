#!/bin/bash

# Define possible values for parameters
strategies=(1 2 3)
datasets=(1 2)
targets=(1 2)
maxHouseholdsNumber=80

# Create output directory
output_dir="metrics_tests_text"
mkdir -p "$output_dir"
sleep 5
# Run Go program for each combination
for dataset in "${datasets[@]}"; do
  for strategy in "${strategies[@]}"; do
    for target in "${targets[@]}"; do
      echo "Running: go run ./privacy_metrics/test_metrics.go $strategy $dataset $target $maxHouseholdsNumber"
      go run "./privacy_metrics/test_metrics.go" "$strategy" "$dataset" "$target" "$maxHouseholdsNumber"
      sleep 15
      # Move the expected output file to block_tests folder
      output_file="test_metrics_${strategy}_${dataset}_${target}_${maxHouseholdsNumber}.txt"
      if [ -f "$output_file" ]; then
        mv "$output_file" "$output_dir/"
      else
        echo "Warning: Expected output file $output_file not found!"
      fi
    done
  done
done

echo "All tests completed. Outputs moved to $output_dir"
