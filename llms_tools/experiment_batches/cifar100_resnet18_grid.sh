#!/usr/bin/env bash
# Grid experiments for CIFAR-100 on ResNet18
# K values: 1,2,5,10,50 with 5 step sizes each
# Batches separated by wait: each K's 5 experiments run in parallel then wait

set -e

# Hyperparameters
KS=(1 2 5 10 50)
STEPS=(0.01 0.001 0.0001 0.00001 0.000001 0.0000001)
TOTAL=$(( ${#KS[@]} * ${#STEPS[@]} ))
COUNTER=0
ROUNDS=5000
BATCH_SIZE=128
REPORT_RATE=20

echo "Cleaning up reports directory..."
mkdir -p reports
rm -rf reports/*

echo "Starting CIFAR-100/ResNet18 grid: total $TOTAL experiments"

for K in "${KS[@]}"; do
  PIDS=()
  MODE="blockwise_sequential"
  if [ "$K" -eq 1 ]; then MODE="entire"; fi
  echo "-- Batch for K=$K (mode=$MODE) --"
  for STEP in "${STEPS[@]}"; do
    # Distribute jobs across GPUs 0 and 1
    GPU=$(( COUNTER % 2 ))
    COUNTER=$(( COUNTER + 1 ))
    echo "[$COUNTER/$TOTAL] Launching K=$K, step_size=$STEP on GPU $GPU"
    python3 -m src.dol1 \
      --model ResNet18 \
      --dataset_name cifar100 \
      --training_mode $MODE \
      --step_size $STEP \
      --batch_size $BATCH_SIZE \
      --rounds $ROUNDS \
      --K $K \
      --cuda_core $GPU \
      --report_sampling_rate $REPORT_RATE &
    PIDS+=("$!")
  done
  # Wait for all processes in this batch to finish
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
  echo "Batch K=$K complete ($COUNTER/$TOTAL done)"
done

echo "All $COUNTER/$TOTAL CIFAR-100/ResNet18 experiments completed."
echo "Reports are in the reports/ directory."