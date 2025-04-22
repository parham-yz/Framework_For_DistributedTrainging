#!/bin/bash

# Experiment script: CIFAR-100 ResNet34 for various K values
Ks=(0 2 5 10 20)
STEP_SIZE="5e-05"
BATCH_SIZE="128"
ROUNDS="5000"
DATASET="cifar100"
REPORT_SAMPLING_RATE="20"
MEASUREMENT_SAMPLING_RATE="399"
CUDA_CORE="0"
COMMUNICATION_DELAY="0"
N_WORKERS="1"

for K in "${Ks[@]}"; do
    if [ "$K" -eq 0 ]; then
        MODE="entire"
    else
        MODE="blockwise_sequential"
    fi
    echo "Running ResNet34 experiment: K=$K, mode=$MODE"
    python3 -m src.dol1 \
        --step_size "$STEP_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --rounds "$ROUNDS" \
        --dataset_name "$DATASET" \
        --cuda_core "$CUDA_CORE" \
        --training_mode "$MODE" \
        --model ResNet34 \
        --report_sampling_rate "$REPORT_SAMPLING_RATE" \
        --measurement_sampling_rate "$MEASUREMENT_SAMPLING_RATE" \
        --K "$K" \
        --communication_delay "$COMMUNICATION_DELAY" \
        --n_workers "$N_WORKERS"
done
echo "All ResNet34 experiments completed."