#!/bin/bash

# List of variables to substitute
vars=(
  "bicycle"
)

# Iterate over each variable
for var in "${vars[@]}"; do
    echo "Starting compression for: $var"
    
    python compress.py \
      --model_path "/d01/luis/datasets/models/$var" \
      --data_device "cuda" \
      --output_vq "/d01/luis/outputs_preliminares_sensitivity/$var/"
      
    echo "Finished: $var"
    echo "-----------------------------------"
done
