#!/bin/bash

DEVICE_IDS=(0 1 2 3)


for i in "${!DEVICE_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES=${DEVICE_IDS[$i]} python eval/run_attention_visualization.py &
done

wait
echo "All task finish."