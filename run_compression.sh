#!/bin/bash

# List of variables to substitute
vars=(
  "bicycle"
  "bonsai"
  "counter"
  "drjohnson"
  "flowers"
  "garden"
  "kitchen"
  "playroom"
  "room"
  "stump"
  "train"
  "treehill"
  "truck"
)

# Iterate over each variable
for var in "${vars[@]}"; do
    echo "Starting compression for: $var"
    
    python compress.py \
      --model_path "/d01/luis/datasets/models/$var" \
      --data_device "cuda" \
      --output_vq "/d01/luis/outputs_preliminares_sensitivity_softmax/$var/"
      
    echo "Finished: $var"
    echo "-----------------------------------"
done
